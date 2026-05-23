# Interview Questions — Embeddings and Semantic Search

These questions appear in ML engineering, search infrastructure, and AI product roles.

---

## Q1: What is an embedding and why does cosine similarity work for comparing them?

??? "Show answer"
    An embedding is a fixed-length dense vector that represents the semantic meaning of a piece of text. It's produced by passing text through a neural network — typically a transformer — and taking the final hidden state (or a pooled version of it).

    Cosine similarity works because embedding models are trained to map semantically similar texts to vectors that point in similar directions in the embedding space. The training signal (e.g., contrastive learning with positive/negative pairs) teaches the model that "dog" and "puppy" should produce nearby vectors while "dog" and "database" should produce distant ones.

    Cosine similarity measures the angle between vectors, ignoring magnitude. This is important because the same concept expressed in a short sentence vs. a long paragraph may produce vectors of different magnitudes. Cosine similarity normalizes this out, so comparison is purely about semantic direction.

---

## Q2: When would you use a bi-encoder vs. a cross-encoder?

??? "Show answer"
    **Bi-encoder:** Encodes the query and each document independently. The score is the dot product / cosine similarity of the two embeddings. Pros: documents can be pre-encoded and indexed offline; query-time cost is O(1) per document (one query embedding + index lookup). Cons: each document is encoded without seeing the query — less accurate.

    **Cross-encoder:** Takes a (query, document) pair as a single input and produces a relevance score. Pros: much higher accuracy because the model sees both query and document simultaneously. Cons: cannot pre-encode documents; every query-document pair requires a fresh forward pass — O(N) at query time.

    **The standard pattern:** Use a bi-encoder to retrieve top-k candidates quickly (e.g., k=50–100 from FAISS), then use a cross-encoder to rerank those k candidates. You get near-cross-encoder accuracy at near-bi-encoder speed.

---

## Q3: Explain the difference between `IndexFlatIP`, `IndexIVFFlat`, and `IndexHNSWFlat` in FAISS.

??? "Show answer"
    **`IndexFlatIP`** (Flat Inner Product): Brute-force exact search. Computes the dot product between the query and every stored vector. 100% recall. O(N) query time. Best for < 100K vectors when accuracy is critical.

    **`IndexIVFFlat`** (Inverted File Flat): Partitions the vector space into `nlist` Voronoi cells (clusters). At query time, only searches `nprobe` cells instead of all N vectors. Approximate — may miss some true nearest neighbors. Requires training (running k-means). O(nprobe × N/nlist) query time. Best for 100K–10M vectors. Set `nprobe` to trade accuracy for speed.

    **`IndexHNSWFlat`** (Hierarchical Navigable Small World): Graph-based ANN. Builds a multilayer graph where each node connects to its M nearest neighbors. At query time, traverses the graph greedily. No training required. Very fast queries, ~O(log N). Uses more memory than IVF. Best for latency-sensitive, general-purpose ANN. `efSearch` controls accuracy/speed tradeoff.

    **Rule of thumb:** Use Flat for prototypes, IVF for large-scale batch pipelines, HNSW for low-latency serving.

---

## Q4: What is the "lost in the middle" problem and how does it affect RAG?

??? "Show answer"
    Research shows that language models perform best at attending to information at the beginning and end of long contexts. Information buried in the middle of a long prompt is more likely to be missed.

    In RAG, if you concatenate 10 retrieved chunks and place the most relevant one in the middle of the context, the model may give a worse answer than if it were first or last.

    **Mitigations:**
    1. **Reranking:** Use a cross-encoder to rerank the top-k retrieved chunks and then place the highest-scored chunks at the start or end of the context.
    2. **Reduce k:** Send fewer, more relevant chunks rather than many loosely relevant ones.
    3. **Recency ordering:** Place the most relevant chunk first, then in descending relevance order.
    4. **Chunk size:** Smaller chunks with higher density are less likely to bury key information.

    The effect is more pronounced in models with context windows > 32K tokens and when chunks are densely packed.

---

## Q5: How would you evaluate a semantic search system's quality?

??? "Show answer"
    Semantic search evaluation requires labeled relevance data. The standard metrics are:

    **Recall@k:** Of all relevant documents for a query, what fraction appear in the top-k results? If a query has 3 relevant documents and your top-5 includes 2 of them, Recall@5 = 0.67.

    **Precision@k:** Of the k retrieved documents, what fraction are actually relevant? If top-5 has 2 relevant out of 5, Precision@5 = 0.40.

    **MRR (Mean Reciprocal Rank):** For each query, take 1 / rank_of_first_relevant_result. Average across queries. A first-position hit scores 1.0; second position scores 0.5. Good for single-answer queries.

    **NDCG (Normalized Discounted Cumulative Gain):** Accounts for graded relevance (not just relevant/irrelevant). Discounts results at lower ranks. Industry standard for search evaluation.

    **Practical process:**
    1. Build a test set of 50–200 queries with labeled relevant documents
    2. Run retrieval, compute Recall@5 and MRR
    3. Compare chunking strategies, embedding models, and reranking configurations
    4. Re-evaluate after any major change to the pipeline

---

## Q6: What is the difference between semantic search and keyword search? When would you use each?

??? "Show answer"
    **Keyword search (BM25, TF-IDF):**
    - Matches documents containing the exact query terms
    - Fast, no GPU needed, interpretable
    - Fails on synonyms ("ML" vs "machine learning"), paraphrases, misspellings
    - Works well when users use exact terminology (legal, medical, product codes)
    - Widely supported: Elasticsearch, PostgreSQL full-text search, Typesense

    **Semantic search (embedding-based):**
    - Matches documents by meaning, regardless of exact wording
    - Handles synonyms, paraphrases, and multilingual queries
    - Requires embedding model + vector index infrastructure
    - Fails at exact matching (product IDs, model numbers)

    **Hybrid search (the production choice):**
    Most production systems combine both. BM25 retrieves high-precision keyword matches; semantic search retrieves meaning-based matches. Results are merged with Reciprocal Rank Fusion (RRF) or a learned combination. This outperforms either approach alone on most benchmarks.

    **Decision:** Start with semantic search for most conversational/natural language queries. Add BM25 if users frequently search by exact terms (model numbers, names, codes). Use hybrid if you need both.

---

## Q7: How does dimension reduction affect embedding quality?

??? "Show answer"
    OpenAI's `text-embedding-3-*` models support the `dimensions` parameter to project embeddings to smaller sizes (e.g., 256 or 512 instead of 1536 or 3072). This uses Matryoshka Representation Learning (MRL) — the model is trained so that the first D dimensions are already a good embedding, not just the full-size one.

    **Benefits of smaller dimensions:**
    - Less memory: 256 float32 = 1KB vs. 3072 float32 = 12KB per vector
    - Faster dot products and index builds
    - Lower storage cost for large corpora

    **Quality tradeoff:**
    On MTEB retrieval benchmarks, `text-embedding-3-large` at 256 dimensions still outperforms `text-embedding-ada-002` at 1536 dimensions — the newer training is that much better.

    Reducing `text-embedding-3-small` from 1536 to 256 dims loses ~5–8% quality on average benchmarks. Whether that's acceptable depends on your use case. Benchmark on your own data before committing.

    **PCA as an alternative:**
    For models without MRL support, you can train a PCA projection on your corpus embeddings and reduce dimensions. This loses more quality than MRL but works with any embedding model.

---

## Q8: How do you handle embedding model updates in a production system?

??? "Show answer"
    Embedding models encode a specific notion of similarity based on their training. When you switch models (e.g., from `text-embedding-ada-002` to `text-embedding-3-small`), all existing embeddings in your index are invalid — they live in a different vector space.

    **Migration process:**
    1. Keep the old model/index serving traffic during migration
    2. Re-embed all documents with the new model into a new index
    3. Shadow-test: route a percentage of queries to both indexes, compare results
    4. Switch traffic to new index only after quality validation
    5. Delete old index after confidence period

    **Versioning:** Store the embedding model name with each vector index (in your metadata). Every chunk should have a `model_version` field so you can detect stale embeddings.

    **Cost:** Re-embedding is a one-time cost. For 1M documents averaging 200 tokens: 200M tokens × $0.02/1M = $4.00 with `text-embedding-3-small`. Usually worth doing for model upgrades.

---

## Q9: What is chunking overlap and why is it used?

??? "Show answer"
    Chunking overlap repeats the last N tokens of chunk K at the start of chunk K+1. For example, with a chunk size of 500 tokens and overlap of 50 tokens, the last 50 tokens of each chunk appear again at the start of the next chunk.

    **Why:** Key sentences, conclusions, and context often span paragraph boundaries. Without overlap, a sentence that starts at the end of one chunk and continues at the start of the next will be split — neither chunk has the full sentence. With overlap, both chunks contain the full boundary sentence.

    **Tradeoffs:**
    - More overlap = better boundary coverage, but more total tokens stored (index size grows)
    - 10–20% overlap is typical (50 tokens overlap on 500-token chunks)
    - Very large overlap (>50%) defeats the purpose of chunking and bloats the index

    **When overlap doesn't help:** For structured data (tables, code, JSONs) where boundaries are semantic (between functions, between rows) rather than mid-sentence. Overlap can introduce duplicate rows or split function signatures in a confusing way.

---

## Q10: How would you design a system that keeps an embedding index up to date as documents change?

??? "Show answer"
    A production embedding index needs an update strategy for three events:

    **Additions:** New documents need to be chunked, embedded, and added to the index. FAISS `index.add()` supports incremental additions without rebuilding.

    **Deletions:** FAISS `IndexFlatIP` doesn't support deletion natively. Solutions:
    - **Tombstone pattern:** Mark deleted chunk IDs in a separate set; filter them out of search results post-retrieval
    - **Rebuild periodically:** Batch deletions, then rebuild the full index nightly/weekly
    - **Use a database that supports deletion:** Qdrant, Pinecone, and Weaviate support native vector deletion

    **Updates:** An updated document means old chunks are invalid and new chunks are needed. Treat as delete + add. Use a `doc_id → chunk_ids` map to know which chunks to delete.

    **Practical architecture:**
    - Event queue (Kafka, SQS) captures document changes
    - Embedding worker consumes events, generates embeddings, writes to vector store
    - Vector store with native CRUD (Qdrant, Pinecone) handles incremental updates
    - Nightly rebuild for FAISS-based systems where deletions accumulate

    For < 1M documents changing at < 100 updates/day, a nightly rebuild is operationally simpler than real-time incremental updates.

---

[[06-practice-exercises]] | [[../Day-03-Part-1-RAG-Basics/00-agenda]]
