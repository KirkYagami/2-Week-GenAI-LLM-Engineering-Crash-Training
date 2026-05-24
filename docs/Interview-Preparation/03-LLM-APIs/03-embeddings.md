# Embeddings

Embeddings underpin semantic search, RAG, clustering, and classification. Interviewers test whether you know how to choose an embedding model and reason about the vector space you're working in.

---

## Q1: What's the difference between `text-embedding-3-small` and `text-embedding-3-large`?

??? "Show answer"
    | | `text-embedding-3-small` | `text-embedding-3-large` |
    |-|--------------------------|--------------------------|
    | **Dimensions** | 1536 (or truncated to 512) | 3072 (or truncated) |
    | **MTEB score** | 62.3 | 64.6 |
    | **Cost** | $0.02 / 1M tokens | $0.13 / 1M tokens |
    | **Best for** | Most RAG use cases | High-precision retrieval where 2% accuracy gain is worth 6× cost |

    The `dimensions` parameter lets you truncate both models. `text-embedding-3-small` at 512 dimensions is often the right default — it's 3× cheaper than `text-embedding-3-large` at full dimensions with only ~2% MTEB difference.

    ```python
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    embedding = client.embeddings.create(
        input="What is the return policy?",
        model="text-embedding-3-small",
        dimensions=512,   # truncate from 1536 to save storage and compute
    ).data[0].embedding
    ```

---

## Q2: What is cosine similarity and why use it instead of Euclidean distance for embeddings?

??? "Show answer"
    **Cosine similarity** measures the angle between two vectors, not their magnitude. Two vectors that point in the same direction have cosine similarity = 1.0, regardless of their length.

    ```python
    import numpy as np

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    ```

    **Why not Euclidean distance?** Embedding models don't normalize magnitudes consistently — two semantically identical sentences may produce vectors of different magnitudes depending on phrasing length. Cosine similarity is magnitude-invariant.

    If vectors are L2-normalized (length = 1), cosine similarity and Euclidean distance produce identical rankings — most embedding APIs return normalized vectors, so either metric works in that case.

---

## Q3: What is the MTEB benchmark and how do you use it to choose an embedding model?

??? "Show answer"
    **MTEB (Massive Text Embedding Benchmark)** evaluates embedding models across 56 tasks: retrieval, clustering, classification, reranking, and semantic similarity. The leaderboard is at `huggingface.co/spaces/mteb/leaderboard`.

    How to use it:
    1. Filter by task type — if you're building a RAG system, filter to **Retrieval** tasks
    2. Filter by model size — if you need to run locally, filter to models that fit your hardware
    3. Compare cost — API models (OpenAI, Cohere) have per-token pricing; open-source models have compute cost

    For RAG specifically, the most relevant MTEB subtask is `BEIR` (a suite of retrieval benchmarks). A model that ranks high on BEIR will generally retrieve relevant chunks reliably.

    Strong open-source alternatives to OpenAI:
    - `BAAI/bge-m3` — multilingual, strong retrieval, free to self-host
    - `sentence-transformers/all-MiniLM-L6-v2` — fast, small, good for high-volume low-latency use cases

---

## Q4: How do you embed documents in batch efficiently?

??? "Show answer"
    The OpenAI embeddings API accepts a list of strings, so batch in a single call rather than looping.

    ```python
    import os
    from openai import AsyncOpenAI
    import asyncio

    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await aclient.embeddings.create(
                input=batch,
                model="text-embedding-3-small",
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings
    ```

    The API limit is 2048 inputs per request and 8191 tokens per input. For large ingestion jobs, parallelise batches with `asyncio.gather` and respect rate limits with a semaphore.

    ```python
    async def embed_all(texts: list[str]) -> list[list[float]]:
        batches = [texts[i:i+100] for i in range(0, len(texts), 100)]
        sem = asyncio.Semaphore(5)  # max 5 concurrent requests

        async def embed_with_sem(batch):
            async with sem:
                return await embed_batch(batch, batch_size=len(batch))

        results = await asyncio.gather(*[embed_with_sem(b) for b in batches])
        return [emb for batch in results for emb in batch]
    ```

---

## Q5: What causes embedding drift and how do you handle it in production?

??? "Show answer"
    **Embedding drift** occurs when you upgrade your embedding model. Embeddings from `text-embedding-ada-002` and `text-embedding-3-small` are not comparable — you cannot mix them in the same index.

    If you upgrade the embedding model:
    1. Re-embed all documents with the new model
    2. Rebuild the vector index from scratch
    3. Test that Recall@k improves before cutting over

    In production, track the embedding model name as metadata on each document:

    ```python
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"embedding_model": "text-embedding-3-small", "version": "1"} for _ in chunks],
    )
    ```

    If you need to migrate incrementally (for large corpora), run both models in parallel, route new queries to the new index, and migrate old documents in background batches.

---

*Previous: [Anthropic API](02-anthropic-api.md) | Next: [Agent Patterns](../04-Agents-and-LangGraph/01-agent-patterns.md)*
