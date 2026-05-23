# Vector Database Overview

A vector database is a storage system optimized for embedding vectors. Beyond raw similarity search, production vector databases add: persistence, filtering by metadata, CRUD operations (delete/update), multi-tenancy, and monitoring.

## Learning objectives

- Compare the capabilities of ChromaDB, Pinecone, and Qdrant
- Understand the indexing algorithms (HNSW, IVF) each database uses
- Choose a database based on operational requirements

---

## Why not just FAISS?

FAISS is a library — it's excellent for similarity search but provides no:

- **Persistence:** You must save/load the index manually and keep metadata in a separate store
- **Filtering:** FAISS has no metadata filtering — you filter post-retrieval
- **CRUD:** Deleting a vector requires rebuilding the index
- **Multi-tenancy:** No concept of users or namespaces
- **Monitoring:** No built-in metrics or observability

Vector databases wrap an ANN library with these production features.

---

## 2025 production landscape

| Database | Deployment | Best for | Pricing model |
|----------|-----------|----------|---------------|
| **ChromaDB** | Local / self-hosted | Development, small projects | Free (open source) |
| **Pinecone** | Fully managed cloud | Teams who want zero ops | Per vector stored + queries |
| **Qdrant** | Self-hosted / managed cloud | Production, filtering-heavy | Free self-hosted / usage-based cloud |
| **Weaviate** | Self-hosted / managed | GraphQL API, multi-modal | Free self-hosted / usage-based |
| **Milvus** | Self-hosted (Kubernetes) | Enterprise, very large scale | Free open source |
| **pgvector** | PostgreSQL extension | Existing Postgres users | Free |

---

## Feature matrix

| Feature | ChromaDB | Pinecone | Qdrant |
|---------|----------|----------|--------|
| Persistent storage | ✅ (PersistentClient) | ✅ | ✅ |
| Metadata filtering | ✅ (basic) | ✅ | ✅ (advanced) |
| Hybrid search | ❌ | ✅ (sparse-dense) | ✅ (sparse + dense) |
| Deletions | ✅ | ✅ | ✅ |
| Namespaces/tenancy | ✅ (collections) | ✅ (namespaces) | ✅ (collections) |
| Self-hosted | ✅ | ❌ | ✅ |
| Serverless | ❌ | ✅ | ✅ (cloud) |
| Python SDK | ✅ | ✅ | ✅ |
| Index algorithm | HNSW | proprietary | HNSW |

---

## When to choose each

```
Starting a new project or prototyping?
    → ChromaDB (zero setup, great Python API)

Building for production with no ops team?
    → Pinecone Serverless (managed, scales automatically)

Need advanced filtering, self-hosted control, or open source?
    → Qdrant (best filtering language, Docker-friendly)

Already running PostgreSQL?
    → pgvector (add vector search to your existing DB)

Very large scale (100M+ vectors), enterprise?
    → Milvus or Weaviate
```

---

## Indexing algorithms

All major vector databases use HNSW (Hierarchical Navigable Small World) as their primary ANN algorithm:

```
HNSW key parameters:
- m: number of bidirectional links per node (16–64)
  Higher = better recall, more memory, slower build
- ef_construction: quality of graph during build (64–512)
  Higher = better quality, slower build
- ef (at query time): search beam width (32–256)
  Higher = better recall, slower query
```

**Typical production settings:**
- `m = 16` — good default for most use cases
- `ef_construction = 200` — build once, build well
- `ef = 64` — tune for your recall/latency target

---

## Distance metrics

Most vector databases support multiple distance functions:

| Metric | Formula | When to use |
|--------|---------|-------------|
| **Cosine** | 1 - (A·B)/(‖A‖‖B‖) | Default for text embeddings |
| **Dot product** | -A·B | Pre-normalized embeddings (fastest) |
| **Euclidean (L2)** | ‖A-B‖₂ | Image features, when magnitude matters |
| **Manhattan (L1)** | Σ|Aᵢ-Bᵢ| | Sparse features |

> [!tip] Use cosine for text, pre-normalize for speed
> Normalizing embeddings to unit length and using dot product is mathematically equivalent to cosine similarity but faster at query time (no division). Do this pre-normalization at ingest time.

---

[[00-agenda]] | [[02-chromadb]]
