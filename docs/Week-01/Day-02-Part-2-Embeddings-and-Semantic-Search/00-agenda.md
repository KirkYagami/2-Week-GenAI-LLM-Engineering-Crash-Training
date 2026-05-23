# Agenda — Embeddings and Semantic Search

Embeddings turn language into geometry. Once text lives in vector space, you can search it with math instead of keywords — enabling a retrieval system that understands meaning, not just spelling.

## Learning objectives

By the end of this session you will be able to:

- Explain what embeddings are and why they represent semantic meaning
- Call the OpenAI and Sentence Transformers embedding APIs
- Compute cosine similarity between vectors
- Build a working in-memory semantic search engine
- Construct a full embedding → retrieval → generation pipeline

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:25 | What are embeddings? — the intuition and the math | [[01-what-are-embeddings]] |
| 0:25 – 0:55 | Embedding models — OpenAI, Sentence Transformers, MTEB | [[02-embedding-models]] |
| 0:55 – 1:20 | Cosine similarity — distance, dot product, L2 norm | [[03-cosine-similarity]] |
| 1:20 – 1:50 | Vector search — brute force, FAISS, approximate nearest neighbor | [[04-vector-search]] |
| 1:50 – 2:30 | Semantic search pipeline — chunking, indexing, retrieval, reranking | [[05-semantic-search-pipeline]] |
| 2:30 – 3:00 | Practice exercises + Q&A | [[06-practice-exercises]] |

## Setup

```python
pip install openai sentence-transformers faiss-cpu numpy
```

```python
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Quick smoke test
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Hello, embeddings!"
)
print(f"Embedding dimension: {len(response.data[0].embedding)}")
# Output: Embedding dimension: 1536
```

[[../Day-02-Part-1-OpenAI-and-Anthropic-APIs/07-interview-questions|← Day 02 Part 1]] | [[01-what-are-embeddings|Start →]]
