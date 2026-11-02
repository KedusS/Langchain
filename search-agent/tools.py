"""The agent's two tools: retrieve official rules and look up card records.

Shared paths and Ollama settings live here so ingestion and chat use the same
embedding model, collection name, and database folder.
"""

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader

load_dotenv()

# Resolve paths relative to this file, so commands work from different folders.
PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_DIR / "chroma"
COLLECTION_NAME = "optcg_comprehensive_rules"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
RULES_PAGE_URL = "https://en.onepiece-cardgame.com/rules/"
CARD_API_URL = "https://optcgapi.com/api/sets/card/{card_id}/"
# Regex patterns recognize IDs such as OP01-001, rule numbers such as 10-1-2-1,
# and headings such as "10-1-2. [Double Attack]". (?m) lets ^ and $ match the
# start and end of each line. The r prefix keeps backslashes in regex strings.
CARD_ID_PATTERN = re.compile(r"^[A-Z0-9]{1,6}-\d{3}$")
RELEVANCE_THRESHOLD = 0.25
RULE_SECTION_PATTERN = re.compile(r"(?m)^\s*(\d+(?:-\d+)+)\.")
KEYWORD_HEADING_PATTERN = re.compile(r"(?m)^[ \t]*(\d+(?:-\d+)+)\.[ \t]*\[([^\]\n]+)\][ \t]*$")

# Keep only the API fields useful to this assistant; ignore unrelated response data.
CARD_FIELDS = (
    "card_set_id",
    "card_name",
    "card_type",
    "card_color",
    "card_cost",
    "card_power",
    "life",
    "attribute",
    "sub_types",
    "counter_amount",
    "rarity",
    "set_name",
    "card_text",
)


def get_embeddings() -> OllamaEmbeddings:
    """Create an Ollama embedding client for indexing and searching text.

    Embeddings are lists of numbers representing text meaning. Queries must use
    the same embedding model as the documents stored in Chroma.
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


@lru_cache(maxsize=1)
def _get_rules_vector_store() -> Chroma:
    """Open the local rules collection and reuse that connection between calls.

    lru_cache stores the returned Chroma object, so it is not reopened for every
    question. The ingestion script must create the index before it can be searched.
    """
    if not CHROMA_DIR.is_dir():
        raise RuntimeError(
            "The local OPTCG rules index is missing. Run "
            "`uv run python ingestion.py` before asking rules questions."
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


# @tool exposes the function to LangChain. Its existing docstring is also the
# description the model reads when deciding how to call it.
@tool
def search_rules(query: str) -> str:
    """Search official OPTCG rules and Rules Q&A. Use the full question for a ruling, or a keyword alone for its definition."""
    if not query.strip():
        raise ValueError("A non-empty rules question is required.")
    # Example: turn "What is Double Attack?" into "Double Attack" for matching.
    definition = re.fullmatch(r"what (?:does (.+?) do|is (.+?))\??", query.strip(), re.IGNORECASE)
    if definition:
        query = next(part for part in definition.groups() if part).strip().rstrip("?")

    store = _get_rules_vector_store()
    # Chroma returns parallel lists: each text chunk has metadata at the same index.
    # Metadata records its source URL, PDF page, and document title.
    indexed_rules = store.get(
        where={"title": "Comprehensive Rules"}, include=["documents", "metadatas"]
    )
    # Definitions have exact headings; don't rely on a similarity rank to find them.
    matches = []
    for text, metadata in zip(indexed_rules.get("documents", []), indexed_rules.get("metadatas", [])):
        if text and any(
            keyword.casefold() == query.strip("[] ").casefold()
            for _, keyword in KEYWORD_HEADING_PATTERN.findall(text)
        ):
            # None is an exact match marker, not an embedding similarity score.
            matches.append((Document(page_content=text, metadata=metadata or {}), None))
    pdf_path = PROJECT_DIR / "data" / "rule_comprehensive.pdf"
    if matches and pdf_path.is_file():
        # A chunk can stop halfway through a definition. Recover its complete
        # PDF page before extracting the specific rule below.
        reader = PdfReader(pdf_path)
        for document, _ in matches:
            document.page_content = reader.pages[int(document.metadata["page"])].extract_text()

    search_options = {"filter": {"title": "Comprehensive Rules"}} if definition or len(query.split()) == 1 else {}
    if not matches:
        # No exact heading: compare the question's embedding with stored chunks.
        # k=4 retrieves up to four candidate passages; the filter narrows the source.
        matches = store.similarity_search_with_relevance_scores(query, k=4, **search_options)
    relevant = [
        (document, score)
        for document, score in matches
        if score is None or score >= RELEVANCE_THRESHOLD
    ]
    if not relevant:
        return (
            "No sufficiently relevant passage was found in the indexed official "
            "Comprehensive Rules. Do not infer a ruling from this result."
        )

    # Prefer an exact Q&A question over passages about a similar situation.
    ignored_words = {"a", "an", "the", "i", "you", "your", "my", "of", "to", "can", "do", "does", "is", "are", "be", "already"}

    def question_words(text):
        """Compare question wording while ignoring common words like 'the'."""
        return set(re.findall(r"[a-z]+", text.lower())) - ignored_words

    exact_qa = []
    for document, score in relevant:
        if document.metadata.get("title") != "Rules Q&A":
            continue
        # Find question-shaped text ending in '?' within a Q&A chunk.
        for question in re.findall(r"(?mi)^[ \t]*((?:Can|Do|Does|What|How|When|Is|Are|May|Must)\b[^?]*\?)", document.page_content):
            if question_words(question) == question_words(query):
                exact_qa.append((document, score))
                break
    if exact_qa:
        relevant = exact_qa

    # A keyword definition should not be mixed with neighbouring abilities.
    keyword_matches = []
    for document, score in relevant:
        for heading in KEYWORD_HEADING_PATTERN.finditer(document.page_content):
            section, keyword = heading.groups()
            if keyword.casefold() != query.strip("[] ").casefold():
                continue
            end = len(document.page_content)
            for following in RULE_SECTION_PATTERN.finditer(document.page_content, heading.end()):
                # Keep child rules (10-1-2-1 under 10-1-2), then stop when a
                # different section begins, rather than including other abilities.
                if not following.group(1).startswith(section + "-"):
                    end = following.start()
                    break
            passage = document.model_copy(update={
                "page_content": document.page_content[heading.start():end].strip()
            })
            keyword_matches.append((passage, score))
    if keyword_matches:
        relevant = keyword_matches

    passages = []
    for document, score in relevant:
        # PDF libraries count pages from 0; citations shown to people start at 1.
        page = int(document.metadata["page"]) + 1
        source = document.metadata["source"]
        title = document.metadata.get("title", "Comprehensive Rules")
        rule_sections = sorted(set(RULE_SECTION_PATTERN.findall(document.page_content)))
        sections = ", ".join(rule_sections) if rule_sections else "not identified"
        match_label = "exact keyword match" if score is None else f"relevance {score:.2f}"
        # Return source information along with the actual text. The agent reads
        # this string, and main.py can use the metadata line to build source links.
        passages.append(
            f"{title} p. {page} ({match_label}; PDF: {source})\n"
            f"Rules on page: {sections}\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(passages)


@tool
def lookup_card(card_id: str, field: str = "") -> str:
    """Look up an OPTCG card by ID, such as OP01-001.

    For a single stat, set field to power, cost, name, color, life, or text.
    Omit field when you need the full card for a rules interaction.
    """
    # Normalize user input, then validate it before constructing the API URL.
    normalized_id = card_id.strip().upper()
    if not CARD_ID_PATTERN.fullmatch(normalized_id):
        raise ValueError(
            "Use a card ID in set-number format, such as OP01-001 or P-001."
        )

    selected_field = field.strip().lower()
    # Accept either a friendly field name (power) or the API name (card_power).
    if selected_field and selected_field not in CARD_FIELDS:
        selected_field = f"card_{selected_field}"
    if selected_field and selected_field not in CARD_FIELDS:
        raise ValueError("Choose a valid card field, such as power, cost, name, or text.")

    url = CARD_API_URL.format(card_id=normalized_id)
    response = requests.get(url, timeout=15)
    if response.status_code == 404:
        return f"The OPTCG card API returned no card for ID {normalized_id}."
    # Other HTTP failures should raise an error instead of becoming fake card data.
    response.raise_for_status()

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as error:
        raise RuntimeError("The OPTCG card API returned invalid JSON.") from error

    # The API may return one record or several alternate-art records. Normalize
    # both shapes to a list so the remaining code has only one case to handle.
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        records = payload
    else:
        raise RuntimeError("The OPTCG card API returned an unexpected response.")

    # A successful HTTP response is not enough: verify the returned card ID too.
    matches = [
        record
        for record in records
        if str(record.get("card_set_id", "")).upper() == normalized_id
    ]
    if not matches:
        return f"The OPTCG card API returned no card for ID {normalized_id}."

    fields = ("card_set_id", selected_field) if selected_field else CARD_FIELDS
    # dict.get() keeps missing values as None (JSON null), rather than inventing 0.
    cards = [
        {key: record.get(key) for key in fields}
        for record in matches
    ]
    power_note = ""
    if selected_field in ("", "card_power") and all(
        str(record.get("card_type", "")).lower() == "event" and record.get("card_power") is None
        for record in matches
    ):
        # An effect saying "+1000 power" is not the Event card's own printed power.
        power_note = "Printed power: not applicable. This is an Event card, which has no printed power.\n"
    # This is reference text for the agent, not a final answer shown directly in chat.
    return (
        f"Card records for {normalized_id} from the OPTCG API "
        f"(parallel/alternate-art entries may be listed separately):\n"
        f"{json.dumps(cards, ensure_ascii=False, indent=2)}\n"
        f"{power_note}"
        f"API source: {url}"
    )
