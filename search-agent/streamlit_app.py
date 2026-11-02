"""Display the chat and pass conversation history to main.invoke_agent().

Streamlit reruns this file after interactions. session_state keeps the chat
between reruns, rather than starting over on the second question.
"""

import streamlit as st
from httpx import ConnectError, HTTPError
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from requests.exceptions import RequestException

from main import invoke_agent
from tools import CHROMA_DIR, OLLAMA_BASE_URL, RULES_PAGE_URL

st.set_page_config(page_title="OPTCG Rules Assistant", page_icon="🏴‍☠️")
st.title("OPTCG Rules Assistant")
st.caption("Explain a One Piece Card Game situation. The agent checks the official rules and relevant card text.")

if st.sidebar.button("New conversation"):
    # Clear both the displayed chat and the model's full message history.
    st.session_state.messages = []
    st.session_state.agent_messages = []
    st.rerun()

if not CHROMA_DIR.is_dir():
    st.sidebar.warning("Rules index missing. Run `uv run python ingestion.py` first.")
st.sidebar.markdown(f"[Official rules page]({RULES_PAGE_URL})")

# messages stores simple display rows. agent_messages stores LangChain message
# objects, including tool calls and their results, which are needed for follow-ups.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

# Redraw previous messages whenever Streamlit reruns the script.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["answer"])

question = st.chat_input("For example: What is blocker?")
if question:
    st.session_state.messages.append({"role": "user", "answer": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build a temporary history with the new question. Save the agent's returned
    # history only after a successful answer, not before the request finishes.
    history = [*st.session_state.agent_messages, HumanMessage(content=question)]
    with st.chat_message("assistant"):
        with st.spinner("Checking the rules and card database..."):
            try:
                answer, agent_messages = invoke_agent(history)
            except ConnectError:
                # Ollama uses httpx; the separate public card API uses requests.
                st.error(
                    f"Can't connect to Ollama at `{OLLAMA_BASE_URL}`. "
                    "Start `ollama serve`, or check your OLLAMA_BASE_URL setting."
                )
                st.stop()  # End this rerun so code below does not use a missing answer.
            except (HTTPError, RequestException, RuntimeError, ValueError, GraphRecursionError) as error:
                st.error(f"Unable to complete the lookup: {error}")
                st.stop()
        st.markdown(answer)

    # The display needs the answer text; the next agent turn needs every message.
    st.session_state.messages.append({"role": "assistant", "answer": answer})
    st.session_state.agent_messages = agent_messages
