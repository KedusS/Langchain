# LangChain OPTCG Rules Assistant

This repository adapts the LangChain search-agent example into an AI assistant
for One Piece Card Game rules and card lookups.

See [search-agent/README.md](search-agent/README.md) for setup and usage. The
assistant uses LangChain `create_agent`, local Chroma RAG over the official
Comprehensive Rules PDF, the public OPTCG card API, and a minimal Streamlit chat
interface.
