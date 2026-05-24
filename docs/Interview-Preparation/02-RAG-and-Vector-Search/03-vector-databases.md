# Vector Databases

Interviewers ask about vector databases to test whether you've actually thought about production storage — persistence, updates, filtering, and scale — not just in-memory retrieval.

---

## Q1: What's the difference between ChromaDB, Pinecone, and Qdrant? When do you choose each?

??? "Show answer"
    | | ChromaDB | Pinecone | Qdrant |
    |-|----------|----------|--------|
    | **Hosting** | Local / self-hosted | Fully managed cloud | Self-hosted or cloud |
    | **Setup** | `pip install chromadb` — zero config | API key, serverless or pod | Docker or binary |
    | **Best for** | Development, prototyping, small prod | Managed scale, no infra ops | High-performance self-hosted |
    | **Filtering** | Metadata filter operators | Metadata filters | Rich payload filters |
    | **Cost** | Free (self-hosted) | Paid per vector stored | Free (self-hosted) |

    **Choose ChromaDB** for development and small-scale production (< 1M vectors) where you control the host.

    **Choose Pinecone** when you need managed scale, don't want to operate infrastructure, and have budget for it.

    **Choose Qdrant** when you need self-hosted production performance with rich filtering — it supports on-disk indexing for large corpora and has strong payload filter operators.

---

## Q2: How do metadata filters work alongside vector search?

??? "Show answer"
    Metadata filters narrow the search space to a subset of documents before (or during) vector similarity ranking. This lets you scope retrieval to a specific user, date range, document type, or tenant.

    ```python
    import chromadb

    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_collection("docs")

    # Filter to only docs from Q4 2024
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={
            "$and": [
                {"year": {"$eq": 2024}},
                {"quarter": {"$eq": "Q4"}},
                {"doc_type": {"$in": ["policy", "faq"]}},
            ]
        }
    )
    ```

    ChromaDB filter operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`.

    > [!warning] Pre-filter vs post-filter
    > Some databases filter *before* ANN search (pre-filter), which is accurate but slower. Others filter *after* retrieval (post-filter), which is faster but may return fewer than `n_results` items. Know which your database uses — Qdrant pre-filters, which is more accurate for small filtered sets.

---

## Q3: What is HNSW and how do its parameters affect retrieval?

??? "Show answer"
    **HNSW (Hierarchical Navigable Small World)** is the graph-based index most vector databases use for approximate nearest neighbour (ANN) search. It trades a small accuracy loss for dramatically faster search than brute force.

    Key parameters:
    - **`M`** — number of edges per node in the graph. Higher M = better recall, more memory. Typical: 16–64.
    - **`ef_construction`** — search width during index build. Higher = better quality index, slower build. Typical: 100–200.
    - **`ef` (ef_search)** — search width at query time. Higher = better recall, slower queries. Typical: 50–200.

    In practice: start with defaults, measure recall at your target latency, then tune `ef` first (query-time only, no index rebuild needed).

    For most RAG use cases the defaults are fine — HNSW recall > 0.95 at reasonable latency for corpora up to tens of millions of vectors.

---

## Q4: When would you use hybrid search over pure vector search?

??? "Show answer"
    Use hybrid search when your queries contain:
    - **Exact identifiers** — product codes, SKUs, names, version numbers, error codes
    - **Technical terminology** that may not appear in the embedding model's training data
    - **Rare proper nouns** — a new product name that hasn't been semanticized in the embedding space yet

    In these cases, BM25 (sparse retrieval) finds the exact match that the embedding model might miss, while dense retrieval catches semantic variants.

    The standard merge is **Reciprocal Rank Fusion (RRF)**:

    ```python
    def rrf_merge(dense_ids: list[str], sparse_ids: list[str], k: int = 60) -> list[str]:
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(dense_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, doc_id in enumerate(sparse_ids):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)
    ```

    Both Pinecone and Qdrant have native sparse-dense support. ChromaDB requires a separate BM25 implementation.

---

## Q5: How do you keep a vector index up to date as source documents change?

??? "Show answer"
    This is the operational problem most candidates forget to mention. Three strategies:

    **1. Upsert-based updates** — when a document changes, delete its chunks and re-embed:

    ```python
    def update_document(collection, doc_id: str, new_text: str):
        # Delete old chunks for this document
        collection.delete(where={"source_doc_id": {"$eq": doc_id}})

        # Re-chunk and re-embed
        chunks = splitter.split_text(new_text)
        embeddings = embed_batch(chunks)
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source_doc_id": doc_id} for _ in chunks]

        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    ```

    **2. Soft delete + version field** — mark old chunks as inactive with a metadata flag; only retrieve `{"active": {"$eq": True}}`. Avoids immediate delete costs but bloats the index over time.

    **3. Rebuild on schedule** — for corpora that change in batch (nightly exports), rebuild the entire index periodically. Simpler operationally; adds latency between update and retrieval.

    Track a `last_modified` timestamp per document in metadata to detect stale chunks efficiently.

---

*Previous: [Chunking and Retrieval](02-chunking-and-retrieval.md) | Next: [OpenAI API](../03-LLM-APIs/01-openai-api.md)*
