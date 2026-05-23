# Setup and Environment — Document Summarizer

## Project structure

```
document-summarizer/
├── app.py              ← FastAPI application
├── summarizer.py       ← Map-reduce summarization logic
├── eval.py             ← RAGAS + custom evaluation
├── requirements.txt
├── .env.example
└── test_docs/          ← Sample documents for testing
    ├── sample_report.pdf
    └── sample_article.md
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```
# requirements.txt
openai==1.51.0
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.0
pymupdf==1.24.11
ragas==0.2.3
langchain-openai==0.2.3
langsmith==0.1.147
python-dotenv==1.0.1
httpx==0.27.2
datasets==3.0.1
```

```bash
# .env.example
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls-...
LANGSMITH_PROJECT=document-summarizer
```

## Token counting utility

Long documents must be chunked before summarization. Know the token count before you start:

```python
# tokens.py
import tiktoken

_enc = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

def chunk_by_tokens(text: str, max_tokens: int = 3000, overlap_tokens: int = 200) -> list[str]:
    """Split text into chunks of at most max_tokens tokens with overlap."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens))
        start += max_tokens - overlap_tokens
    return chunks
```

Test your setup:

```python
from tokens import count_tokens, chunk_by_tokens

text = open("test_docs/sample_article.md").read()
print(f"Total tokens: {count_tokens(text)}")
chunks = chunk_by_tokens(text)
print(f"Chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i}: {count_tokens(chunk)} tokens")
```

---

[[README]] | [[02-implementation]]
