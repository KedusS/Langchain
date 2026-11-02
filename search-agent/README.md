# OPTCG Rules Assistant

One LangChain agent using **qwen3:1.7b**, official OPTCG rules, and card data.
The application has four files:

- `main.py` creates the agent, supplies references, and returns its answer.
- `tools.py` contains `search_rules` and `lookup_card`.
- `ingestion.py` downloads the official PDFs and builds the Chroma index.
- `streamlit_app.py` displays the chat and keeps conversation history.

Start with `invoke_agent()` in `main.py` to follow one question from input to
answer. Then read its tool calls in `tools.py`. `ingestion.py` prepares the data,
and `streamlit_app.py` connects the agent to the browser. Each file includes
comments explaining the steps and function inputs and outputs.

This follows the model-and-tools approach in
[Eden Marco's course](https://github.com/emarco177/langchain-course/blob/project/search-agent/main.py).

## Run

Requires Python 3.11+, uv, and Ollama. From the repository root:

```bash
cd search-agent
uv sync
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
ollama serve
```

Keep Ollama running. In another terminal, from `search-agent`:

```bash
uv run python ingestion.py
uv run streamlit run streamlit_app.py
```

Open port **8501**. For a terminal-only question, use `uv run python main.py`.

## How it works

Each question receives official rule passages and card records for any card IDs
it names. The agent answers from those references and can call either tool for
more evidence. Keyword definitions use the complete source rule; situation
questions search the indexed Comprehensive Rules and Rules Q&A. Card records
come from the public [OPTCG API](https://optcgapi.com/documentation).

The app keeps tool calls and results in chat history. **New conversation** clears
it. If the model prints a tool request instead of an answer, it gets one
correction attempt. Missing cards are reported from the API response.

Run ingestion again to refresh the official PDFs. Generated PDFs, the Chroma
index, `.env`, and `.venv` are excluded from version control.

## Configuration

Optional `.env` settings:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:1.7b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Ollama must run where this app can reach it. In Codespaces, `localhost` refers
to the Codespace, not your personal computer.

## Checks

```bash
uv run python -m unittest test_rules_grounding.py
```

These eight offline tests cover card data, rule retrieval, chat history, and
response handling. They do not
prove that every model-generated ruling is correct; check cited official rules.
