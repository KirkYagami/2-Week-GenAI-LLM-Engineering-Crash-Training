# Setup and Environment — RAG Q&A Chatbot

## Project structure

```
rag-chatbot/
├── app.py              ← FastAPI application
├── ingest.py           ← Document ingestion pipeline
├── eval.py             ← RAGAS evaluation script
├── requirements.txt
├── .env.example
├── Dockerfile
├── fly.toml
└── docs/               ← Source documents to ingest
    ├── faq.md
    └── product-guide.pdf
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```
# requirements.txt
openai==1.51.0
chromadb==0.5.15
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.0
pymupdf==1.24.11
sentence-transformers==3.1.1
httpx==0.27.2
ragas==0.2.3
langchain-openai==0.2.3
python-dotenv==1.0.1
```

```bash
# .env.example
OPENAI_API_KEY=sk-...
CHROMA_PERSIST_PATH=./chroma_db
COLLECTION_NAME=documents
```

## Document ingestion

```python
# ingest.py
import os
import pathlib
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

try:
    import pymupdf as fitz
except ImportError:
    import fitz

load_dotenv()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]

def extract_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n\n".join(page.get_text() for page in doc)

def ingest(docs_dir: str = "./docs") -> None:
    chroma = chromadb.PersistentClient(path=os.getenv("CHROMA_PERSIST_PATH", "./chroma_db"))
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    collection = chroma.get_or_create_collection(
        os.getenv("COLLECTION_NAME", "documents"),
        embedding_function=embedding_fn,
    )

    documents, ids, metadatas = [], [], []
    for path in pathlib.Path(docs_dir).rglob("*"):
        if path.suffix == ".pdf":
            text = extract_pdf(str(path))
        elif path.suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
        else:
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{path.stem}_{i}"
            if chunk_id not in {doc_id for doc_id in collection.get()["ids"]}:
                documents.append(chunk)
                ids.append(chunk_id)
                metadatas.append({"source": path.name, "chunk_index": i})

    if documents:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
        print(f"Added {len(documents)} chunks. Total: {collection.count()}")
    else:
        print("No new documents to ingest.")

if __name__ == "__main__":
    ingest()
```

Run ingestion:

```bash
OPENAI_API_KEY=sk-... python ingest.py
# Output: Added 47 chunks. Total: 47
```

Verify the index:

```python
import chromadb
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_collection("documents")
print(f"Documents indexed: {collection.count()}")
results = collection.query(query_texts=["refund policy"], n_results=3)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[{meta['source']}] {doc[:100]}...")
```

> [!tip] Use overlapping chunks to avoid splitting answers across boundaries
> A chunk boundary that falls mid-sentence means the answer to a question might be split between two chunks. A 150-token overlap between chunks means each sentence appears in at least two chunks, preventing this.

> [!warning] Re-ingesting the same document creates duplicate chunks
> The `ingest.py` above checks existing IDs to avoid duplicates. If you need a full re-index, delete the `chroma_db/` directory and run `ingest.py` again.

---

[[README]] | [[02-implementation]]
