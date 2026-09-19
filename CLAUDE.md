# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RAG Chatbot Application** — Retrieval-Augmented Generation system built with Streamlit, Llama Index, and local Ollama LLM. Users upload PDFs and ask questions about document content. The system retrieves relevant chunks and answers using an LLM.

### Tech Stack
- **Framework**: Streamlit (web UI)
- **RAG Pipeline**: Llama Index (llama-index-core, llama-index-embeddings-huggingface)
- **LLM**: Ollama (local, llama3.2:1b model)
- **Embeddings**: HuggingFace (BAAI/bge-large-en-v1.5)
- **Vector Store**: Qdrant (available but not actively used in current code)
- **Runtime**: Python 3.13, uv package manager

## Getting Started

### Prerequisites
Ensure Ollama is installed and running with the llama3.2:1b model available:
```bash
# Install ollama if needed (https://ollama.ai)
ollama pull llama3.2:1b
ollama serve  # Run in a separate terminal before starting the app
```

### Setup & Run
```bash
# Activate virtual environment
source .venv-1/bin/activate

# Install dependencies (if missing)
pip install streamlit llama-index llama-index-core llama-index-llms-ollama \
  llama-index-embeddings-huggingface torch huggingface_hub

# Start the app
streamlit run app.py
```

The app runs on `http://localhost:8501` by default.

## Architecture & Code Flow

### `app.py` — Main Application
**Flow:**
1. **PDF Upload** (sidebar) → Temp file storage
2. **Document Processing** → Load with `SimpleDirectoryReader`, build vector index
3. **LLM & Embeddings Setup** → Initialize Ollama LLM and HuggingFace embeddings
4. **Query Engine Creation** → Streaming-enabled query engine with custom prompt
5. **Chat Loop** → Accept user queries, stream responses, store in session state

**Key Components:**
- `load_llm()` — Cached resource that loads Ollama model (120s timeout)
- `reset_chat()` — Clears session messages and context
- `display_pdf()` — Renders uploaded PDF in iframe
- Session state caching (`st.session_state.file_cache`) — Avoids re-indexing same PDF

**Customization Point:**
- Line 100–109: Custom QA prompt template. Modify to change answer style/format.

### `notebook.ipynb` — Exploration/Development
Jupyter notebook for testing RAG pipeline components before integration into the app.

## Important Notes

### Session State & Caching
- Files are cached per session (`session_id + filename`) to avoid redundant indexing
- LLM is cached with `@st.cache_resource` to reuse across reruns
- Chat history stored in `st.session_state.messages`

### PDF Processing
- Only `.pdf` files supported (filtered in uploader)
- Uses `SimpleDirectoryReader` to load PDFs from temp directory
- Full document is indexed; no chunking parameters are exposed

### Streaming Behavior
- Query engine configured with `streaming=True`
- Response is built character-by-character for real-time display
- Cursor animation (▌) shown during streaming

### Known Limitations & TODOs
- No explicit document chunking strategy (uses llama-index defaults)
- No reranking currently enabled (code references "Cohere reranker" in comment but not implemented)
- No chat persistence across sessions (session cache only)
- No authentication or multi-user isolation
- Vector store (Qdrant) imported but not used

## Debugging

**App won't start:**
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Verify `.venv-1` exists and has required packages

**Slow PDF indexing:**
- Embedding generation is the bottleneck (BAAI/bge-large-en-v1.5)
- First run will download the model (~1GB)

**Out-of-memory issues:**
- Reduce document size or use a smaller embedding model
- `gc.collect()` is called on chat reset but may not be sufficient for large PDFs

## File Structure
```
rag-project/
├── app.py                          # Main Streamlit application
├── notebook.ipynb                  # Exploration notebook
├── docs/
│   └── dspy.pdf                   # Sample reference document
└── .venv-1/                        # Python virtual environment (Python 3.13)
```
## after modification
make sure to implement new or update existing test whenever required