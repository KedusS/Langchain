"""Create the agent and handle one chat turn.

Start reading at invoke_agent(): retrieve references -> run the agent -> check
the response -> return answer and history. Names starting with _ are helpers
used inside this file; Streamlit calls invoke_agent().
"""

import json
import os
import re
from typing import Sequence

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from tools import CARD_API_URL, OLLAMA_BASE_URL, RULE_SECTION_PATTERN, lookup_card, search_rules

# Read optional settings from .env; os.getenv() below supplies defaults.
load_dotenv()

# These are instructions sent to the model, not Python code that enforces rules.
SYSTEM_PROMPT = """You are an OPTCG rules assistant. Use the retrieved references
provided with the current question. Card API data supplies printed card facts;
official rules explain game mechanics. Always Call tools.
- For a specific ruling, prefer an explicit official Rules Q&A answer when returned.
  Pass the full user's question to search_rules for a specific ruling.
  State that Q&A answer and its source; do not add a reason the Q&A does not give.
- Printed card stats or text: call lookup_card with the card ID. For a single
  stat, pass field, for example field="power". If no card is found, say so.
  Copy printed stats from their API fields; do not derive them from card effects.
  A null field means no value is provided, not zero. Explain this in ordinary
  language, not as "power is null". Event cards have no printed power.
- Card interactions: use both tools.
For follow-ups, you may reuse tool results already in this conversation if they
answer the question. Otherwise call a tool. When reusing rules, cite the exact
rule section from that tool result in your answer.
Use the tool results, not memory or previous answers. Explain how the exact
rule's conditions and required actions apply. Timing and activation actions are
separate requirements: meeting the timing does not waive an action or cost.
Already being in an action's resulting state is not performing that action.
For state-change questions, also search for the rules governing that state change.
Do not invent rules or URLs.
Cite the returned rule section, PDF page, and source link.
For Rules Q&A, cite the PDF page and source link without inventing a rule number.
For a keyword definition, give at most two sentences describing that keyword; do not add other abilities
or infer exceptions. Read its meaning from the supplied definition, not its name.
Do not name example cards unless lookup_card has verified them. For card stats,
give only the requested stat and API source link. Keep answers brief.
If the tools do not settle the question, say you cannot verify the ruling."""

llm = ChatOllama(
    model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:1.7b"),
    base_url=OLLAMA_BASE_URL,
    temperature=0,       # Reduce randomness; this does not guarantee correctness.
    reasoning=False,     # Disable Ollama's separate thinking-output mode.
    num_ctx=8192,         # Context budget for messages and reference text, in tokens.
    num_predict=384,      # Limit the length of each model response, in tokens.
)

agent = create_agent(
    model=llm,
    tools=[search_rules, lookup_card],
    system_prompt=SYSTEM_PROMPT,
)


def _is_tool_request_text(text: str) -> bool:
    """Return True if reply text contains JSON shaped like a tool request.

    A real tool call belongs in AIMessage.tool_calls. Sometimes the small model
    prints that request as text instead; it should not be shown as an answer.
    """
    def is_call(value):
        """Recognize a request directly, inside a list, or inside a wrapper."""
        if isinstance(value, list):
            return any(is_call(item) for item in value)
        if not isinstance(value, dict):
            return False
        if "tool_calls" in value:
            return is_call(value["tool_calls"])
        call = value.get("function", value)
        return isinstance(call, dict) and isinstance(call.get("name"), str) and any(
            key in call for key in ("arguments", "args", "parameters")
        )

    decoder = json.JSONDecoder()
    # Look inside prose or code fences too, not just at the start of the reply.
    for start in re.finditer(r"[\[{]", text):
        try:
            # raw_decode reads one JSON value without requiring the reply to end there.
            payload, _ = decoder.raw_decode(text[start.start():])
        except ValueError:
            continue
        if is_call(payload):
            return True
    return False


# A regular expression (text pattern) that captures a source's title, page, and
# URL from search_rules output. Parentheses in the pattern create those groups.
PDF_REFERENCES = re.compile(
    r"(Comprehensive Rules|Rules Q&A) p\. (\d+) \([^\n]*?PDF: (https://en\.onepiece-cardgame\.com/[^\s)]+)\)"
)


def _run_agent(messages: list[BaseMessage], question_index: int) -> list[BaseMessage]:
    """Return the agent's completed message history, with one format retry.

    question_index identifies the latest user message so a correction can be
    added there. LangChain handles real tool calls inside agent.invoke().
    """
    for attempt in range(2):
        output = agent.invoke({"messages": messages}).get("messages", [])
        final = output[-1] if output else None
        if not isinstance(final, AIMessage) or final.tool_calls or not final.text.strip():
            raise RuntimeError("The agent did not finish its answer. Please try again.")
        if not _is_tool_request_text(final.text):
            return output
        if attempt:
            raise RuntimeError("The model returned a tool request instead of an answer. The request was not displayed; please try again.")
        # Copy the list and message instead of modifying the caller's history.
        # The second attempt still uses this same agent and the supplied sources.
        messages = list(messages)
        messages[question_index] = messages[question_index].model_copy(update={"content": (
            messages[question_index].text + "\n\nThe card and rules references are already "
            "provided above. Give the final answer in plain language. Do not print a tool "
            "request or tool-call JSON. Use actual tool calling only if more evidence is needed."
        )})


def _cites_rule(answer: str, reference: str) -> bool:
    """Check for a rule number or Q&A link that exists in the reference text.

    This checks the citation, not whether the model interpreted the rule correctly.
    """
    return any(
        # The boundaries prevent 10-1-4 from matching an invented 10-1-4-99.
        re.search(r"(?<![\d-])" + re.escape(section) + r"(?![\d-])", answer)
        for section in RULE_SECTION_PATTERN.findall(reference)
    ) or any(
        source in answer and "/qa_rules.pdf" in source
        for title, _, source in PDF_REFERENCES.findall(reference) if title == "Rules Q&A"
    )


def invoke_agent(messages: Sequence[BaseMessage]) -> tuple[str, list[BaseMessage]]:
    """Answer the newest user question in a LangChain conversation.

    messages contains HumanMessage, AIMessage, and any ToolMessage results.
    Returns (answer_text, updated_messages): the UI displays the text and saves
    updated_messages for the next turn. It does not save just the answer text.
    """
    # Search backwards for the most recent user message; -1 means none was found.
    question_index = next((i for i in range(len(messages) - 1, -1, -1)
                           if isinstance(messages[i], HumanMessage)), -1)
    if question_index < 0:
        raise ValueError("Ask an OPTCG question first.")
    question = messages[question_index].text
    # The set removes duplicate IDs if the same card is mentioned more than once.
    card_ids = set(re.findall(r"\b[A-Z0-9]{1,6}-\d{3}\b", question.upper()))
    cards = {}
    # Retrieve evidence up front because the small model can skip tool calls.
    # .invoke({...}) is how Python calls functions wrapped with LangChain @tool.
    for card_id in sorted(card_ids):
        content = lookup_card.invoke({"card_id": card_id})
        if content:
            cards[card_id] = content
    rules = search_rules.invoke({"query": question})
    card_evidence = "\n\n".join(cards.values()) or "No card ID was specified."
    model_messages = list(messages)
    # Give the model the question plus its references without changing the input
    # history. model_copy() creates a new message with only content replaced.
    model_messages[question_index] = messages[question_index].model_copy(update={"content": (
        f"Question: {question}\n\nCard API reference retrieved for this question:\n{card_evidence}\n\n"
        f"Official reference retrieved for this question:\n{rules}"
    )})
    output = _run_agent(model_messages, question_index)
    # Store the original question, not the large reference block. Actual agent
    # tool calls and ToolMessage results remain in the returned history.
    if question_index < len(output) and isinstance(output[question_index], HumanMessage):
        output[question_index] = output[question_index].model_copy(
            update={"content": messages[question_index].content}
        )
    answer = output[-1].text
    # Failed tools are not evidence. fresh_tools contains only this turn's results;
    # tools also includes successful results kept from earlier conversation turns.
    tools = [m for m in output if isinstance(m, ToolMessage) and m.status != "error"]
    fresh_tools = [m for m in output[len(messages):] if isinstance(m, ToolMessage) and m.status != "error"]
    if not (cards or fresh_tools or PDF_REFERENCES.search(rules)):
        # If no new reference was available, require a citation to a prior rule.
        cited_rule = any(_cites_rule(answer, m.text) for m in tools if m.name == "search_rules")
        if not tools or (not card_ids and not cited_rule):
            raise RuntimeError(
                "The agent did not consult a source or cite a previously retrieved rule, "
                "so its answer was not shown. Please try again."
            )
    card_results = list(cards.values()) + [m.text for m in tools if m.name == "lookup_card"]
    for card_id in card_ids:
        # An API "not found" result takes precedence over any invented model stat.
        missing = f"The OPTCG card API returned no card for ID {card_id}."
        if any(missing in result for result in card_results):
            output[-1] = output[-1].model_copy(update={"content": missing})
            return missing, output
        source = f"API source: {CARD_API_URL.format(card_id=card_id)}"
        if not any(source in result for result in card_results):
            raise RuntimeError(f"The agent did not verify {card_id} with lookup_card, so its answer was not shown.")
    # Attach links from verified lookup results when the model leaves them out.
    for card_id in sorted(card_ids):
        source = CARD_API_URL.format(card_id=card_id)
        if source not in answer:
            answer += f"\n\n[{card_id} card data]({source})"
    references = ([rules] if not card_ids else []) + [m.text for m in fresh_tools if m.name == "search_rules"]
    for reference in references:
        for title, page, source in PDF_REFERENCES.findall(reference):
            if source not in answer:
                answer += f"\n\n[{title}, p. {page}]({source}#page={page})"
    # Keep the final answer, including added source links, in agent history too.
    output[-1] = output[-1].model_copy(update={"content": answer})
    return answer, output


def main() -> None:
    """Run one question in the terminal instead of opening the Streamlit UI."""
    question = input("Ask an OPTCG rules or card question: ")
    # _ means this terminal entry point does not need the returned chat history.
    answer, _ = invoke_agent([HumanMessage(content=question)])
    print(answer)


if __name__ == "__main__":
    # Runs for `python main.py`, but not when Streamlit imports invoke_agent.
    main()
