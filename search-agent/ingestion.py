"""Build the local rules index before starting the chat application.

Run with `uv run python ingestion.py`. This downloads the official PDFs,
extracts text, splits it into chunks, and stores their embeddings in Chroma.
"""

import re
from io import BytesIO
from urllib.parse import urljoin

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from tools import CHROMA_DIR, COLLECTION_NAME, PROJECT_DIR, RULES_PAGE_URL, get_embeddings

# Page URL, PDF filename, document title, chunk size, overlap.
# Sizes are characters, not model tokens. Smaller Q&A chunks keep each short
# question close to its answer; overlap repeats text across chunk boundaries.
SOURCES = [
    (RULES_PAGE_URL, "rule_comprehensive.pdf", "Comprehensive Rules", 1000, 150),
    (urljoin(RULES_PAGE_URL, "faq/"), "qa_rules.pdf", "Rules Q&A", 300, 50),
]


def download_official_pdf(page_url: str, filename: str) -> tuple[bytes, str]:
    """Find a PDF link on an official page and return (PDF bytes, source URL)."""
    page = requests.get(page_url, timeout=20)
    page.raise_for_status()
    # Escape the filename so characters such as '.' are literal in the regex.
    pattern = r"""href=["']([^"']*""" + re.escape(filename) + r"""(?:\?[^"']*)?)["']"""
    match = re.search(pattern, page.text, re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"No {filename} link was found on {page_url}.")
    # urljoin also handles relative links such as ../pdf/qa_rules.pdf.
    pdf_url = urljoin(page_url, match.group(1))
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    # Reject an HTML error page even if the server labels it as a successful request.
    if not response.content.startswith(b"%PDF-"):
        raise RuntimeError(f"The official download was not a PDF: {pdf_url}")
    return response.content, pdf_url


def ingest_rules() -> int:
    """Rebuild the collection from both PDFs and return the number of chunks.

    This is a setup/refresh operation, not something the chat runs for each turn.
    The saved PDFs are also used to read complete keyword definitions later.
    """
    chunks, downloads = [], []
    for page_url, filename, title, chunk_size, overlap in SOURCES:
        content, source = download_official_pdf(page_url, filename)
        # BytesIO lets PdfReader read the downloaded bytes as an in-memory file.
        reader = PdfReader(BytesIO(content))
        # Each Document holds page text plus its original page number and source.
        # := assigns extracted text once; pages with no readable text are skipped.
        pages = [
            Document(page_content=text, metadata={"page": number, "source": source, "title": title})
            for number, page in enumerate(reader.pages)
            if (text := page.extract_text()) and text.strip()
        ]
        if not pages:
            raise RuntimeError(f"No readable text was found in {title}.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True,
        )
        chunks.extend(splitter.split_documents(pages))
        # Keep the bytes until indexing finishes, then save the source PDFs locally.
        downloads.append((filename, content, source))
    if not chunks:
        raise RuntimeError("The official rules PDF produced no searchable text.")

    embeddings = get_embeddings()
    if CHROMA_DIR.exists():
        # Replace this named collection so a refresh does not duplicate old chunks.
        store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        store.delete_collection()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # Chroma uses the embedding client to turn chunks into vectors and persist them.
    Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        collection_name=COLLECTION_NAME, persist_directory=str(CHROMA_DIR),
    )
    data_dir = PROJECT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, content, _ in downloads:
        (data_dir / filename).write_bytes(content)
    print(f"Indexed {len(chunks)} chunks from " + " and ".join(source for _, _, source in downloads))
    print(f"Local Chroma index: {CHROMA_DIR}")
    return len(chunks)


if __name__ == "__main__":
    # Importing this module does not rebuild the index; running the file does.
    ingest_rules()
