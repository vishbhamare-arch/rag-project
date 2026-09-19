# RAG Chatbot with Llama 3.2 & PDF Documents

Conversational AI application that answers questions about PDF documents using Retrieval-Augmented Generation (RAG). Upload a PDF, ask questions, and get intelligent answers powered by Llama 3.2 running locally via Ollama.

## Features

- **PDF Upload & Processing** — Upload PDF files and automatically build searchable indexes
- **Real-time Streaming** — Watch responses stream as they're generated character-by-character
- **Local LLM** — Runs Llama 3.2 (1B parameter) entirely on your machine via Ollama
- **Session Persistence** — Chat history maintained within a session, PDFs cached to avoid re-indexing
- **Smart Retrieval** — Uses HuggingFace embeddings (BAAI/bge-large-en-v1.5) for semantic search
- **Conversational Interface** — Clean web UI built with Streamlit
- **Token Usage Tracking** — Real-time token metrics for embeddings, prompts, and responses (see [TOKEN_TRACKING.md](TOKEN_TRACKING.md))

## Quick Start

### Prerequisites
- Python 3.13+
- [Ollama](https://ollama.ai) installed and running locally
- ~3-4GB available disk space (for models and embeddings)

### Installation

1. **Install Ollama and pull the model:**
   ```bash
   # If you don't have Ollama installed
   # Visit https://ollama.ai and install
   
   # Pull the Llama 3.2 1B model
   ollama pull llama3.2:1b
   
   # Start Ollama server (run in a separate terminal)
   ollama serve
   ```

2. **Clone/setup the project:**
   ```bash
   cd rag-project
   source .venv-1/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit llama-index llama-index-core llama-index-llms-ollama \
     llama-index-embeddings-huggingface torch huggingface_hub
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   
   The app will open at `http://localhost:8501`

## Usage

1. **Upload a PDF** — Click "Choose your `.pdf` file" in the sidebar and select a document
2. **Wait for indexing** — The system will process the PDF and build a vector index (first run may take 1-2 minutes)
3. **Ask questions** — Type questions in the chat box and press Enter
4. **View responses** — Responses stream in real-time, displayed in the chat interface
5. **Clear chat** — Click the "Clear ↺" button to reset conversation and free memory

## How It Works

1. **Document Processing**
   - PDF is loaded and split into chunks
   - Chunks are embedded using HuggingFace's BAAI/bge-large-en-v1.5 model
   - Embeddings are stored in a vector index

2. **Query Processing**
   - User question is embedded
   - Similar document chunks are retrieved (semantic search)
   - Retrieved context + question is sent to Llama 3.2

3. **Response Generation**
   - Llama 3.2 generates answers based on context
   - Response streams to UI in real-time

## Project Structure

```
rag-project/
├── app.py                 # Main Streamlit application
├── notebook.ipynb         # Jupyter notebook for exploration/testing
├── docs/
│   └── dspy.pdf          # Sample documentation (example)
├── .venv-1/              # Python virtual environment
├── CLAUDE.md             # Developer guidance (for Claude Code)
└── README.md             # This file
```

## Configuration

### Custom Prompt Template
Edit the QA prompt in `app.py` (lines 100-109) to customize answer style:

```python
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above I want you to think step by step...\n"
    "Query: {query_str}\n"
    "Answer: "
)
```

### Models & Parameters
- **LLM Model:** `llama3.2:1b` (configured in `load_llm()`)
- **Embedding Model:** `BAAI/bge-large-en-v1.5` 
- **LLM Timeout:** 120 seconds (can increase for slower systems)

To use a different Ollama model, edit line 33 in `app.py`:
```python
llm = Ollama(model="your-model-name:tag", request_timeout=120.0)
```

## Troubleshooting

### "Connection refused" Error
**Issue:** App can't connect to Ollama  
**Solution:** 
```bash
# Make sure Ollama is running in another terminal
ollama serve

# Verify it's accessible
curl http://localhost:11434/api/tags
```

### Slow PDF Indexing
**Issue:** First PDF takes too long to process  
**Solution:**
- Embedding model download (~1GB) happens on first run
- Subsequent PDFs will be faster
- Large PDFs (100+ pages) naturally take longer

### Out of Memory Errors
**Issue:** App crashes when uploading large PDFs  
**Solution:**
- Split the PDF into smaller documents
- Use a smaller embedding model
- Clear chat history to free memory

### Streamlit Not Found
**Issue:** `ModuleNotFoundError: No module named 'streamlit'`  
**Solution:**
```bash
source .venv-1/bin/activate
pip install streamlit
```

## Performance Notes

- **First run:** ~2-3 minutes (embedding model download + initialization)
- **PDF indexing:** 1-5 minutes depending on file size
- **Query response:** 3-10 seconds (LLM generation time)
- **Embedding generation:** Main bottleneck (GPU acceleration not configured)

## Testing

### Run Tests
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_token_counter.py -v
pytest tests/test_token_tracker.py -v
pytest tests/test_integration.py -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html
```

**Test Coverage:**
- 60 total tests
- 19 unit tests for token counting
- 28 unit tests for token aggregation
- 13 integration tests for workflows
- All tests passing ✓

For detailed token tracking feature documentation, see [TOKEN_TRACKING.md](TOKEN_TRACKING.md).

## Limitations

- Single PDF per session (can be extended to support multiple)
- No persistent storage (chat history lost after session ends)
- No multi-user support or authentication
- Responses may hallucinate if context doesn't contain relevant information
- Vector store (Qdrant) is imported but not actively used

## Future Enhancements

- [ ] Multiple PDF support in single session
- [ ] Chat persistence to database
- [ ] Document management UI (delete, update indexes)
- [ ] GPU acceleration for embeddings
- [ ] Advanced retrieval (hybrid search, reranking)
- [ ] User authentication and workspaces

## Requirements

See `CLAUDE.md` for development setup and architecture details.

**Runtime Requirements:**
- Python 3.13+
- Ollama with llama3.2:1b
- ~6GB RAM (4GB for models + 2GB for app)
- Internet connection (for initial model/embedding downloads)

## License

This project is provided as-is for educational and personal use.

## Support

For issues or questions:
1. Check the "Troubleshooting" section above
2. Review `CLAUDE.md` for developer documentation
3. Ensure Ollama is running and accessible
4. Check system resources (RAM, disk space)
