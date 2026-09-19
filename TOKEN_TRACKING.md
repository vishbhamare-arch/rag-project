# Token Usage Tracking Feature

Comprehensive token usage tracking system for the RAG chatbot application. Tracks embedding, prompt, and response tokens across the entire session.

## Overview

The token tracking feature monitors token consumption at three key points:

1. **Embedding Tokens** — Generated when PDFs are indexed
2. **Prompt Tokens** — Generated from user queries
3. **Response Tokens** — Generated from LLM responses

All metrics are aggregated per-session and displayed in the UI.

## Architecture

### Components

#### 1. TokenCounter (`token_counter.py`)
Responsible for counting tokens in text using HuggingFace transformers tokenizer.

**Key Methods:**
- `count_tokens(text: str) -> int` — Count tokens in a single text
- `count_tokens_batch(texts: List[str]) -> List[int]` — Count tokens in multiple texts
- `estimate_embedding_tokens(num_docs, avg_tokens) -> int` — Estimate tokens for embedding
- `get_avg_tokens_per_doc(texts) -> float` — Calculate average tokens per document

**Example:**
```python
counter = TokenCounter()
tokens = counter.count_tokens("Hello world")
print(tokens)  # Output: number of tokens
```

#### 2. TokenTracker (`token_tracker.py`)
Aggregates token usage across the session and provides analytics.

**Key Methods:**
- `add_embedding_tokens(count, doc_name)` — Track embedding tokens
- `add_prompt_tokens(count, query)` — Track query tokens
- `add_response_tokens(count, response)` — Track response tokens
- `add_message_tokens(msg_id, role, content, prompt_tokens, response_tokens)` — Track full message
- `get_summary() -> Dict` — Get aggregated statistics
- `get_per_message_summary() -> List[Dict]` — Get per-message breakdown
- `reset()` — Reset all counters
- `to_dict() / from_dict()` — Serialize/deserialize state

**Example:**
```python
tracker = TokenTracker()
tracker.add_embedding_tokens(1000, "document.pdf")
tracker.add_prompt_tokens(50, "What is this?")
tracker.add_response_tokens(200, "The answer is...")
summary = tracker.get_summary()
print(summary)
# Output:
# {
#   'total_tokens': 1250,
#   'embedding_tokens': 1000,
#   'prompt_tokens': 50,
#   'response_tokens': 200,
#   'message_count': 1,
#   'operations': {...},
#   'uptime_seconds': 12.5
# }
```

### Integration with App

**Session Initialization (`app.py` lines ~27-30):**
```python
if "id" not in st.session_state:
    st.session_state.token_tracker = TokenTracker()
    st.session_state.token_counter = TokenCounter()
```

**PDF Upload & Indexing (`app.py` lines ~90-102):**
```python
doc_tokens = counter.estimate_embedding_tokens(
    len(docs),
    counter.get_avg_tokens_per_doc(doc_texts)
)
tracker.add_embedding_tokens(int(doc_tokens), uploaded_file.name)
```

**Query Processing (`app.py` lines ~163-169):**
```python
prompt_tokens = counter.count_tokens(prompt)
tracker.add_prompt_tokens(prompt_tokens, prompt)
# ... query execution ...
response_tokens = counter.count_tokens(full_response)
tracker.add_response_tokens(response_tokens, full_response)
tracker.add_message_tokens("msg1", "assistant", full_response, 
                          prompt_tokens, response_tokens)
```

**UI Display (`app.py` lines ~144-158):**
```python
with st.sidebar:
    st.subheader("📊 Token Usage")
    summary = st.session_state.token_tracker.get_summary()
    col1_metric, col2_metric = st.columns(2)
    with col1_metric:
        st.metric("Total Tokens", summary['total_tokens'])
        st.metric("Embedding Tokens", summary['embedding_tokens'])
    with col2_metric:
        st.metric("Prompt Tokens", summary['prompt_tokens'])
        st.metric("Response Tokens", summary['response_tokens'])
```

## Data Flow

```
User Upload PDF
    ↓
count_tokens(pdf_text) via TokenCounter
    ↓
add_embedding_tokens() → TokenTracker
    ↓
User Query
    ↓
count_tokens(query) via TokenCounter
    ↓
add_prompt_tokens() → TokenTracker
    ↓
LLM Response
    ↓
count_tokens(response) via TokenCounter
    ↓
add_response_tokens() → TokenTracker
    ↓
Display metrics in sidebar
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_token_counter.py -v
pytest tests/test_token_tracker.py -v
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Coverage

**test_token_counter.py** (19 tests):
- Token counting accuracy
- Batch processing
- Edge cases (empty, special chars, unicode, very long text)
- Average calculation

**test_token_tracker.py** (28 tests):
- Token aggregation
- Message tracking
- Persistence (to_dict/from_dict)
- Summary generation
- Reset functionality
- Edge cases (large counts, negative values, multiple operations)

**test_integration.py** (13 tests):
- Full RAG workflow simulation
- Multi-turn conversations
- Data persistence
- Complex scenarios
- Session lifecycle

**Total: 60 tests, all passing**

## Usage Example

### Complete Workflow

```python
from token_counter import TokenCounter
from token_tracker import TokenTracker

# Initialize
counter = TokenCounter()
tracker = TokenTracker()

# Simulate document upload
pdf_content = "This is a document about machine learning..." * 50
pdf_tokens = counter.count_tokens(pdf_content)
tracker.add_embedding_tokens(pdf_tokens, "ml_guide.pdf")

# Simulate user query
query = "What is machine learning?"
query_tokens = counter.count_tokens(query)
tracker.add_prompt_tokens(query_tokens, query)

# Simulate LLM response
response = "Machine learning is a subset of AI that..."
response_tokens = counter.count_tokens(response)
tracker.add_response_tokens(response_tokens, response)

# Add to message history
tracker.add_message_tokens(
    message_id="msg1",
    role="assistant",
    content=response,
    prompt_tokens=query_tokens,
    response_tokens=response_tokens
)

# Get analytics
summary = tracker.get_summary()
print(f"Total tokens used: {summary['total_tokens']}")
print(f"Embedding tokens: {summary['embedding_tokens']}")
print(f"Prompt tokens: {summary['prompt_tokens']}")
print(f"Response tokens: {summary['response_tokens']}")

# Get per-message breakdown
breakdown = tracker.get_per_message_summary()
for msg in breakdown:
    print(f"{msg['role'].upper()}: {msg['tokens']} tokens")
```

## Metrics Available

### Summary Metrics
- `total_tokens` — Sum of all token usage
- `embedding_tokens` — Tokens from document indexing
- `prompt_tokens` — Tokens from user queries
- `response_tokens` — Tokens from LLM responses
- `message_count` — Number of messages processed
- `operations` — Breakdown by operation type
- `uptime_seconds` — Session duration

### Per-Message Metrics
- `message_id` — Unique message identifier
- `role` — "user" or "assistant"
- `content` — Message preview (first 100 chars)
- `prompt_tokens` — Query tokens (for assistant messages)
- `response_tokens` — Response tokens (for assistant messages)
- `total` — Sum of prompt + response tokens
- `timestamp` — ISO format timestamp

## Performance Considerations

1. **Token Counting Cost**
   - ~50-100ms per document (depends on size)
   - Batch counting available for efficiency

2. **Memory Usage**
   - Minimal (< 1KB per message)
   - Message history stores 100-char previews

3. **Tokenizer Loading**
   - HuggingFace tokenizer loaded once at app startup
   - ~500MB download on first run

## Future Enhancements

- [ ] Token cost calculation (dollars/cents)
- [ ] Token usage quotas and alerts
- [ ] Historical tracking across sessions
- [ ] Export token metrics (CSV/JSON)
- [ ] Token efficiency analysis
- [ ] Multi-user token aggregation

## Troubleshooting

### TokenCounter Not Loading
```python
# Falls back to word splitting if tokenizer fails
counter = TokenCounter()  # Uses word count if transformers unavailable
```

### Incorrect Token Counts
- Different tokenizers may give slightly different counts
- Use consistent tokenizer throughout session
- Ensure transformers library is up-to-date

### Memory Usage Rising
```python
# Reset tracker to clear message history
st.session_state.token_tracker.reset()
```

## References

- TokenCounter uses HuggingFace `transformers` library
- Compatible with Llama 2/3 tokenizers
- Supports custom tokenizer models

## Files

- `token_counter.py` — Tokenization utility
- `token_tracker.py` — Token aggregation and analytics
- `tests/test_token_counter.py` — Unit tests for counter
- `tests/test_token_tracker.py` — Unit tests for tracker
- `tests/test_integration.py` — Integration tests
- `pytest.ini` — Pytest configuration
- `requirements-test.txt` — Test dependencies
