# Interview Questions — Vector Databases

---

## Q1: What is the difference between ChromaDB, Pinecone, and Qdrant? When would you choose each?

??? "Show answer"
    **ChromaDB:** Open-source, runs in-process (no server required), great Python API. Choose for local development, prototyping, or small-scale production (< 1M vectors) where you want zero infrastructure overhead.

    **Pinecone:** Fully managed serverless cloud service. No servers to provision or tune. Automatically scales to zero. Choose when you want zero operational burden and are comfortable with the vendor lock-in and per-query pricing.

    **Qdrant:** Open-source, written in Rust, excellent performance, the most expressive metadata filter DSL. Runs locally (in-memory or file), via Docker, or in managed cloud. Choose when you need advanced filtering, self-hosted control, or production scale with open-source economics.

    **Decision framework:**
    - Prototype/dev → ChromaDB
    - Production, no ops team → Pinecone Serverless
    - Production, self-hosted or advanced filtering → Qdrant
    - Existing PostgreSQL infrastructure → pgvector

---

## Q2: What is HNSW and what are its key configuration parameters?

??? "Show answer"
    HNSW (Hierarchical Navigable Small World) is a graph-based Approximate Nearest Neighbor algorithm. It builds a multilayer graph where each node is connected to its approximate nearest neighbors. Queries traverse the graph greedily from a coarse top layer to a fine bottom layer.

    **Key parameters:**

    **`m` (connections per node, 8–64):** Number of bidirectional edges per node. Higher m = better recall, more memory (each edge costs ~8-16 bytes), slower build time. Default: 16 for most use cases.

    **`ef_construction` (build quality, 64–512):** Controls how many neighbors are evaluated when inserting each node. Higher = better quality graph, slower build. Default: 200.

    **`ef` (search beam width, 32–256):** Controls how many nodes are explored during a query. Higher = better recall, slower query. Tune at query time without rebuilding the index.

    **Memory:** With `m=16`, HNSW uses roughly 1.1× the raw vector storage in additional graph overhead.

    **Recall vs. latency tradeoff:** With `ef=64`, typical recall@10 ≈ 95–98%. With `ef=128`, recall approaches 99%. The latency difference is 2–4ms per query. Tune based on your recall requirements.

---

## Q3: Explain pre-filtering vs. post-filtering in vector search. Which does Qdrant use?

??? "Show answer"
    **Post-filtering:** Perform ANN search over all vectors, retrieve top-K, then apply metadata filter to the results. Simple but dangerous: if the filter is highly selective (1% of docs match), your top-1000 ANN results may only contain 10 matching documents — you effectively get Recall@10 instead of Recall@1000.

    **Pre-filtering:** Apply the metadata filter first to get a candidate set, then perform ANN search only within that candidate set. Accurate but requires either a payload index (fast) or a full scan of filtered vectors (slow).

    **Qdrant's approach:** Qdrant intelligently chooses between strategies based on filter selectivity and available payload indexes:
    - If the filtered set is large (> `full_scan_threshold`): use ANN within the filtered set
    - If the filtered set is small: use brute-force exact search within the filtered set
    - Without payload indexes, filtering requires scanning all payloads (slow)

    **Practical implication:** Always create payload indexes for fields used in filters. Without them, Qdrant falls back to slow full payload scans regardless of selectivity.

---

## Q4: What is Reciprocal Rank Fusion and why is it preferred over score-based merging?

??? "Show answer"
    RRF merges ranked lists from multiple retrieval systems using rank position rather than raw scores.

    **Formula:** `RRF(d) = Σ 1 / (k + rank_i(d))` where the sum is over all retrieval systems and k=60 is a constant that dampens the impact of top ranks.

    **Why rank-based instead of score-based:**
    Dense retrieval produces cosine similarities (0 to 1). BM25 produces term frequency scores (0 to 20+). These scales are incompatible — you can't directly add 0.87 (cosine) + 12.4 (BM25) and get a meaningful combined score. You'd need to normalize each score first.

    RRF sidesteps this: it only uses the rank (position in the sorted list), which is unit-free. Rank 1 from dense + rank 2 from BM25 produces a high combined RRF score regardless of the raw score values.

    **Why k=60:** The constant prevents rank 1 from completely dominating. Without it, rank 1 from either system scores infinity while rank 2 scores 1 — a single system's top result always wins. k=60 means rank 1 = 1/61, rank 2 = 1/62, so the difference is small and documents appearing in multiple lists accumulate scores.

---

## Q5: When would you choose pgvector over a dedicated vector database?

??? "Show answer"
    pgvector adds vector similarity search to PostgreSQL as an extension. It's the right choice when:

    **Already using PostgreSQL:** You keep vectors and metadata in the same database, join queries are trivial, transactions work naturally (update vector + metadata atomically), and you have one less system to operate.

    **Simpler filtering requirements:** pgvector can use full PostgreSQL WHERE clauses — join with other tables, use subqueries, leverage existing indexes. This is more powerful than any dedicated vector DB's filter DSL.

    **Small to medium scale (< 5M vectors):** pgvector with HNSW index handles this well. Above 10M vectors, dedicated vector databases typically outperform pgvector on query latency.

    **When NOT to use pgvector:**
    - Very high query throughput (> 1K QPS): PostgreSQL's connection overhead becomes a bottleneck
    - Frequent vector updates: HNSW index in pgvector doesn't yet support efficient incremental updates
    - Multi-model or multi-tenant with billions of vectors: dedicated systems scale better

    **2025 note:** pgvector 0.7+ added HNSW support and substantially improved performance. The gap with dedicated vector databases has narrowed significantly for most use cases.

---

## Q6: How do you handle vector index updates when documents change?

??? "Show answer"
    Vector updates are one of the operationally tricky parts of RAG systems. Three strategies:

    **Upsert (insert or update by ID):**
    - All major vector databases support upserting by document ID
    - For text changes: re-embed the new text and upsert with the same ID
    - Limitation: the old vector stays in the HNSW graph until a compaction — some databases (FAISS) don't support deletion well

    **Tombstone + rebuild:**
    - Mark deleted IDs in a separate set; filter them out post-retrieval
    - Periodically rebuild the index from scratch to clean up tombstoned vectors
    - Simplest operationally, but wastes storage and has latency during rebuild

    **Native deletion (Qdrant, Pinecone):**
    - Delete by ID or by filter condition
    - Qdrant marks deleted vectors in the HNSW graph and cleans up during optimization
    - Safe for CRUD operations at production scale

    **Practical pattern:**
    1. Store `doc_id → [chunk_ids]` mapping in your application database
    2. When a document updates: delete all old chunk IDs from the vector store, re-chunk, re-embed, re-upsert
    3. Use Qdrant or Pinecone for native deletion; avoid FAISS for frequently-updated collections

---

## Q7: What is vector quantization and when would you use it in production?

??? "Show answer"
    Vector quantization compresses float32 vectors to smaller representations to reduce memory and improve throughput.

    **Scalar quantization (int8):**
    - Compresses float32 (4 bytes/dim) to int8 (1 byte/dim) — 4× memory reduction
    - Clips values at a quantile (typically 99th percentile) and maps the range to [-128, 127]
    - Quality loss: < 1% on most NDCG benchmarks
    - Supported by: Qdrant, FAISS

    **Binary quantization:**
    - Compresses to 1 bit/dim — 32× memory reduction
    - Quality loss: 2–5% depending on embedding model and dataset
    - Best for: very large collections (100M+ vectors) where memory is the bottleneck
    - Supported by: Qdrant, FAISS

    **Product quantization (PQ):**
    - Splits each vector into subvectors, quantizes each subspace independently
    - 8–64× compression with tunable quality
    - Standard in FAISS IVF-PQ indexes

    **When to use:**
    - 1536-dim embeddings at 10M vectors = 57 GB float32; with int8 = 14 GB — fits in one A100
    - Use scalar quantization in production when you have > 1M vectors
    - Enable binary quantization only after benchmarking recall on your actual queries — some embedding models don't quantize well

---

## Q8: Design a multi-tenant vector database for a SaaS platform with 1000 customers.

??? "Show answer"
    A multi-tenant vector database must ensure: customer A's queries never return customer B's documents, per-customer deletion works correctly, and the system scales as customers grow.

    **Approach 1 — Namespace per customer (Pinecone):**
    One index, one namespace per customer. Queries are automatically isolated by namespace. Works well up to ~10K customers. Deletion by namespace is fast.

    **Approach 2 — Metadata filter per customer:**
    All customers share one collection. Each vector has `customer_id` in metadata. Every query includes `where={"customer_id": {"$eq": customer_id}}`. Works in ChromaDB and Qdrant. Risk: metadata filter bugs could expose cross-customer data.

    **Approach 3 — Collection per customer (Qdrant):**
    Each customer gets their own Qdrant collection. Complete isolation. Works well up to ~1000 customers. Beyond that, collection overhead becomes significant.

    **Approach 4 — Separate index per customer (Pinecone serverless):**
    Create a separate index per customer. Maximum isolation, but index creation takes 30–60s — not suitable for instant provisioning.

    **Recommendation for 1000 customers:**
    - Qdrant with one collection per customer (1000 collections scales fine)
    - Or Pinecone with namespace per customer (all in one serverless index)
    - Include `customer_id` in all metadata for audit trails, even with namespace isolation
    - Implement customer quota limits (max vectors per customer) at the application layer

---

[[07-practice-exercises]] | [[../Day-04-Part-1-LLM-Evaluation/00-agenda]]
