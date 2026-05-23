# Agenda — Vector Databases

FAISS is great for a laptop. Production RAG needs a vector database: persistence, filtering, CRUD updates, and multi-tenant isolation. Today you learn the three databases you'll encounter most: ChromaDB, Pinecone, and Qdrant.

## Learning objectives

By the end of this session you will be able to:

- Create collections and ingest vectors in ChromaDB, Pinecone, and Qdrant
- Use metadata filters alongside vector search
- Implement hybrid search (dense + sparse)
- Choose the right vector database for a given production scenario

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:20 | Vector DB overview — capabilities, tradeoffs, pricing | [[01-vector-db-overview]] |
| 0:20 – 0:50 | ChromaDB — local dev, persistent storage, collections | [[02-chromadb]] |
| 0:50 – 1:20 | Pinecone — serverless, namespaces, sparse-dense | [[03-pinecone]] |
| 1:20 – 1:50 | Qdrant — self-hosted, payloads, filtering | [[04-qdrant]] |
| 1:50 – 2:20 | Indexing and filtering deep dive | [[05-indexing-and-filtering]] |
| 2:20 – 2:50 | Hybrid search — BM25 + dense, RRF | [[06-hybrid-search]] |
| 2:50 – 3:00 | Practice exercises | [[07-practice-exercises]] |

## Setup

```bash
pip install chromadb pinecone-client qdrant-client sentence-transformers
```

```python
import os
import chromadb
from pinecone import Pinecone
from qdrant_client import QdrantClient

# ChromaDB — works without any API key
chroma = chromadb.Client()

# Pinecone — requires PINECONE_API_KEY
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Qdrant — in-memory (no server needed for local dev)
qdrant = QdrantClient(":memory:")
```

[[../Day-03-Part-1-RAG-Basics/07-interview-questions|← Day 03 Part 1]] | [[01-vector-db-overview|Start →]]
