# RAG Pipeline End-to-End

This note builds a complete, production-ready RAG pipeline: ingest a PDF, chunk it, embed it, store it in ChromaDB, and answer questions with citations.

## Learning objectives

- Ingest PDF documents into a RAG pipeline
- Store and query embeddings with ChromaDB
- Handle multi-document corpora with metadata
- Measure and log pipeline performance

---

## Full pipeline implementation

```python
import os
import hashlib
import json
import time
from dataclasses import dataclass, field
from openai import OpenAI
from anthropic import Anthropic
import chromadb
from chromadb.utils import embedding_functions
import tiktoken

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
enc = tiktoken.get_encoding("cl100k_base")

@dataclass
class RAGConfig:
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "claude-sonnet-4-6"
    chunk_size: int = 512          # tokens
    chunk_overlap: int = 64        # tokens
    retrieval_k: int = 4
    min_relevance_score: float = 0.5
    max_context_tokens: int = 4000
```

---

## Stage 1: Document ingestion

```python
def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path: str) -> str:
    """Extract text from PDF using PyPDF2."""
    try:
        import PyPDF2
        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text.strip())
        return "\n\n".join(text_parts)
    except ImportError:
        raise ImportError("Install PyPDF2: pip install PyPDF2")

def load_document(path: str) -> tuple[str, str]:
    """Returns (text, source_name)."""
    import os
    ext = os.path.splitext(path)[1].lower()
    source = os.path.basename(path)

    if ext == ".pdf":
        return load_pdf(path), source
    elif ext in {".txt", ".md"}:
        return load_text_file(path), source
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_document(text: str, source: str, config: RAGConfig) -> list[dict]:
    """Fixed-size token chunking with overlap."""
    tokens = enc.encode(text)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + config.chunk_size]
        chunk_text = enc.decode(chunk_tokens).strip()

        if chunk_text:
            # Stable ID based on content hash
            chunk_id = hashlib.md5(f"{source}-{i}-{chunk_text[:50]}".encode()).hexdigest()[:16]
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source": source,
                "chunk_index": len(chunks),
                "token_count": len(chunk_tokens),
                "char_start": len(enc.decode(tokens[:i]))
            })

        i += config.chunk_size - config.chunk_overlap

    return chunks
```

---

## Stage 2: Embedding and storage

```python
def get_chroma_collection(name: str = "rag_collection") -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    return chroma_client.get_or_create_collection(
        name=name,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )

def ingest_chunks(
    chunks: list[dict],
    collection: chromadb.Collection,
    batch_size: int = 100
) -> dict:
    """Upsert chunks into ChromaDB in batches."""
    start_time = time.time()
    added = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"]
            } for c in batch]
        )
        added += len(batch)
        print(f"  Upserted {added}/{len(chunks)} chunks...")

    elapsed = time.time() - start_time
    return {"chunks_added": added, "elapsed_seconds": elapsed}

def ingest_document(path: str, collection: chromadb.Collection, config: RAGConfig) -> dict:
    """Full ingestion pipeline for a single document."""
    print(f"Loading: {path}")
    text, source = load_document(path)
    print(f"  Loaded {len(text):,} chars")

    chunks = chunk_document(text, source, config)
    print(f"  Created {len(chunks)} chunks")

    result = ingest_chunks(chunks, collection)
    result["source"] = source
    result["chunks_created"] = len(chunks)
    return result
```

---

## Stage 3: Retrieval

```python
def retrieve(
    query: str,
    collection: chromadb.Collection,
    config: RAGConfig,
    source_filter: str | None = None
) -> list[dict]:
    """Query ChromaDB and return formatted results."""
    where_clause = {"source": source_filter} if source_filter else None

    results = collection.query(
        query_texts=[query],
        n_results=config.retrieval_k,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )

    if not results["ids"][0]:
        return []

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - distance/2 for cosine space
        similarity = 1 - dist / 2
        if similarity >= config.min_relevance_score:
            retrieved.append({
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "similarity": similarity
            })

    return retrieved
```

---

## Stage 4: Generation

```python
SYSTEM_PROMPT = """You are a precise Q&A assistant. Answer questions using ONLY the provided document context.

Instructions:
- If the answer is in the context, give it clearly and cite the source document.
- If the answer is not in the context, respond exactly: "I don't have information about this in the provided documents."
- Do not use your general knowledge — only the provided context.
- Cite sources using the format: [Source: filename]"""

def generate_answer(
    question: str,
    retrieved: list[dict],
    config: RAGConfig
) -> dict:
    if not retrieved:
        return {
            "answer": "I don't have information about this in the provided documents.",
            "sources": [],
            "retrieved_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0
        }

    # Assemble context
    context_parts = []
    sources = []
    for item in retrieved:
        context_parts.append(f"[Source: {item['source']}]\n{item['text']}")
        if item["source"] not in sources:
            sources.append(item["source"])

    context = "\n\n---\n\n".join(context_parts)

    response = anthropic_client.messages.create(
        model=config.generation_model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"
        }]
    )

    cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000

    return {
        "answer": response.content[0].text,
        "sources": sources,
        "retrieved_count": len(retrieved),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": cost
    }
```

---

## Putting it all together

```python
class RAGPipeline:
    def __init__(self, collection_name: str = "rag_demo", config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.collection = get_chroma_collection(collection_name)

    def ingest(self, path: str) -> dict:
        return ingest_document(path, self.collection, self.config)

    def ask(self, question: str, source_filter: str | None = None) -> dict:
        retrieved = retrieve(question, self.collection, self.config, source_filter)
        result = generate_answer(question, retrieved, self.config)
        result["question"] = question
        return result

    def stats(self) -> dict:
        count = self.collection.count()
        return {"total_chunks": count, "collection": self.collection.name}

# Demo with sample text files (no real PDFs needed)
import tempfile, os

# Create sample "documents"
sample_docs = {
    "company-policy.txt": """
Vacation Policy: All employees receive 20 days of paid time off annually.
Unused PTO rolls over up to 10 days into the next calendar year.
PTO requests must be submitted through the HR portal with 2 weeks notice.

Remote Work: Employees may work remotely up to 3 days per week.
Full remote approval requires manager sign-off and a minimum of 6 months tenure.
Core hours are 10am–3pm in the employee's local timezone.
    """,
    "product-faq.txt": """
Q: How do I reset my password?
A: Click 'Forgot Password' on the login page. You will receive a reset link within 5 minutes.

Q: What payment methods do you accept?
A: We accept all major credit cards, PayPal, and bank transfers for enterprise accounts.

Q: Is there a free trial?
A: Yes, all plans include a 14-day free trial with no credit card required.

Q: How do I cancel my subscription?
A: You can cancel at any time from Settings > Billing. Your access continues until the end of the billing period.
    """
}

# Write to temp files and ingest
pipeline = RAGPipeline(collection_name="demo", config=RAGConfig(retrieval_k=3))

with tempfile.TemporaryDirectory() as tmpdir:
    for filename, content in sample_docs.items():
        path = os.path.join(tmpdir, filename)
        with open(path, "w") as f:
            f.write(content.strip())
        pipeline.ingest(path)

print(f"\nPipeline stats: {pipeline.stats()}")

# Ask questions
questions = [
    "How many vacation days do employees get?",
    "Can I work from home every day?",
    "How do I cancel my subscription?",
    "What is the company's stock price?"  # out of scope
]

for q in questions:
    result = pipeline.ask(q)
    print(f"\nQ: {q}")
    print(f"A: {result['answer']}")
    print(f"   Sources: {result['sources']} | Tokens: {result['input_tokens']}/{result['output_tokens']} | ${result['cost_usd']:.5f}")
```

---

## Performance logging

```python
import csv
from datetime import datetime

def log_query(result: dict, log_file: str = "rag_queries.csv") -> None:
    fieldnames = ["timestamp", "question", "retrieved_count", "sources",
                  "input_tokens", "output_tokens", "cost_usd"]
    row = {
        "timestamp": datetime.now().isoformat(),
        "question": result["question"],
        "retrieved_count": result["retrieved_count"],
        "sources": "; ".join(result["sources"]),
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost_usd": result["cost_usd"]
    }

    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
```

> [!tip] Log everything in production
> Every query, retrieved chunk, and generated answer should be logged. This gives you: (1) a dataset for evaluating retrieval quality, (2) cost tracking per query type, (3) the ability to replay queries after prompt improvements, and (4) a paper trail for auditing AI-generated answers.

---

[[03-retrieval-and-augmentation]] | [[05-rag-vs-fine-tuning]]
