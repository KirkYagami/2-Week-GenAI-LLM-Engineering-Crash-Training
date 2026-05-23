# Pinecone

Pinecone is a fully managed vector database — no servers to provision, no indexes to tune. The serverless tier scales to zero and charges per query and storage, making it cost-effective for variable workloads.

## Learning objectives

- Create and manage Pinecone serverless indexes
- Upsert vectors with metadata and namespace isolation
- Query with metadata filters and namespace routing
- Implement sparse-dense (hybrid) search

---

## Setup

```python
import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
```

---

## Creating a serverless index

```python
index_name = "my-rag-index"

# Create index (only if it doesn't exist)
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,           # must match your embedding model
        metric="cosine",          # "cosine", "euclidean", or "dotproduct"
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"   # check pinecone.io for available regions
        )
    )
    print(f"Created index: {index_name}")

# Connect to the index
index = pc.Index(index_name)

# Check index stats
stats = index.describe_index_stats()
print(f"Vectors: {stats.total_vector_count}")
print(f"Namespaces: {stats.namespaces}")
```

---

## Upserting vectors

```python
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]

# Prepare vectors in Pinecone format: list of (id, values, metadata) tuples
documents = [
    {"id": "doc-001", "text": "Python is a high-level programming language", "source": "wiki", "category": "tech", "year": 2024},
    {"id": "doc-002", "text": "FastAPI builds APIs quickly with Python", "source": "docs", "category": "tech", "year": 2024},
    {"id": "doc-003", "text": "The Eiffel Tower stands 330m in Paris", "source": "wiki", "category": "travel", "year": 2023},
    {"id": "doc-004", "text": "Machine learning uses data to build models", "source": "textbook", "category": "tech", "year": 2024},
]

# Batch upsert (max 100 per upsert for efficiency)
texts = [d["text"] for d in documents]
embeddings = embed(texts)

vectors = [
    {
        "id": doc["id"],
        "values": emb,
        "metadata": {k: v for k, v in doc.items() if k not in {"id", "text"}}
    }
    for doc, emb in zip(documents, embeddings)
]

# Upsert to default namespace
index.upsert(vectors=vectors, namespace="default")
print(f"Upserted {len(vectors)} vectors")

# Check updated stats (may take a moment to reflect)
import time; time.sleep(2)
stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")
```

---

## Querying

```python
def search_pinecone(
    query: str,
    k: int = 5,
    namespace: str = "default",
    filter: dict | None = None,
    min_score: float = 0.0
) -> list[dict]:
    query_emb = embed([query])[0]

    kwargs = {
        "vector": query_emb,
        "top_k": k,
        "namespace": namespace,
        "include_metadata": True
    }
    if filter:
        kwargs["filter"] = filter

    results = index.query(**kwargs)

    return [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata
        }
        for match in results.matches
        if match.score >= min_score
    ]

# Basic search
results = search_pinecone("Python web frameworks", k=3)
for r in results:
    print(f"[{r['score']:.3f}] {r['id']} — {r['metadata']}")

# Filtered search — only return tech documents from 2024
filtered = search_pinecone(
    "programming language",
    k=5,
    filter={
        "$and": [
            {"category": {"$eq": "tech"}},
            {"year": {"$eq": 2024}}
        ]
    }
)
```

### Pinecone filter operators

| Operator | Example |
|----------|---------|
| `$eq`, `$ne` | `{"status": {"$eq": "active"}}` |
| `$gt`, `$gte`, `$lt`, `$lte` | `{"year": {"$gte": 2024}}` |
| `$in`, `$nin` | `{"tag": {"$in": ["ml", "ai"]}}` |
| `$and`, `$or` | `{"$and": [...]}` |
| `$exists` | `{"optional_field": {"$exists": True}}` |

---

## Namespaces for multi-tenancy

Namespaces isolate vectors — a query to namespace A never sees vectors from namespace B.

```python
# Add vectors for different tenants
for tenant_id in ["tenant-A", "tenant-B"]:
    vectors = [
        {
            "id": f"{tenant_id}-doc-1",
            "values": embed([f"Document for {tenant_id}"])[0],
            "metadata": {"tenant": tenant_id}
        }
    ]
    index.upsert(vectors=vectors, namespace=tenant_id)

# Queries are isolated per namespace
results_a = index.query(
    vector=embed(["document"])[0],
    top_k=5,
    namespace="tenant-A",
    include_metadata=True
)

results_b = index.query(
    vector=embed(["document"])[0],
    top_k=5,
    namespace="tenant-B",
    include_metadata=True
)

print(f"Tenant A results: {[m.id for m in results_a.matches]}")
print(f"Tenant B results: {[m.id for m in results_b.matches]}")

# Delete all vectors in a namespace
index.delete(delete_all=True, namespace="tenant-A")
```

> [!tip] Namespace per user or per document collection
> Use namespaces to separate different customers (multi-tenancy) or different document collections (e.g., "support-docs" vs. "blog-posts"). This prevents cross-contamination and enables isolated delete operations.

---

## CRUD operations

```python
# Fetch specific vectors by ID
fetch_result = index.fetch(ids=["doc-001", "doc-002"], namespace="default")
for id, vector in fetch_result.vectors.items():
    print(f"{id}: dim={len(vector.values)}, metadata={vector.metadata}")

# Update metadata (re-upsert with same ID)
updated_metadata = {
    "source": "wiki-updated",
    "category": "tech",
    "year": 2025,
    "verified": True
}
index.upsert(
    vectors=[{"id": "doc-001", "values": embed(["Python is a high-level programming language"])[0],
               "metadata": updated_metadata}],
    namespace="default"
)

# Delete specific vectors
index.delete(ids=["doc-003"], namespace="default")

# Delete by metadata filter
index.delete(
    filter={"category": {"$eq": "travel"}},
    namespace="default"
)
```

---

## Sparse-dense hybrid search

Pinecone supports native sparse vectors for hybrid BM25+dense search.

```python
from pinecone_text.sparse import BM25Encoder

# Initialize BM25 encoder (fit on your corpus)
corpus = [d["text"] for d in documents]
bm25 = BM25Encoder()
bm25.fit(corpus)

# Create hybrid index (must specify metric="dotproduct" for hybrid)
hybrid_index_name = "hybrid-rag"
if hybrid_index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=hybrid_index_name,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

hybrid_index = pc.Index(hybrid_index_name)

# Upsert with both dense and sparse vectors
for doc in documents:
    dense_vec = embed([doc["text"]])[0]
    sparse_vec = bm25.encode_documents([doc["text"]])[0]

    hybrid_index.upsert(
        vectors=[{
            "id": doc["id"],
            "values": dense_vec,
            "sparse_values": {"indices": sparse_vec["indices"], "values": sparse_vec["values"]},
            "metadata": {k: v for k, v in doc.items() if k not in {"id", "text"}}
        }]
    )

# Hybrid query
query_text = "Python programming"
dense_query = embed([query_text])[0]
sparse_query = bm25.encode_queries([query_text])[0]

hybrid_results = hybrid_index.query(
    vector=dense_query,
    sparse_vector={"indices": sparse_query["indices"], "values": sparse_query["values"]},
    top_k=3,
    include_metadata=True,
    alpha=0.5  # 0 = pure sparse, 1 = pure dense
)
```

---

## Cost estimation

```python
def estimate_pinecone_cost(
    num_vectors: int,
    dim: int = 1536,
    daily_queries: int = 1000,
    bytes_per_float: int = 4
) -> dict:
    # Pinecone serverless pricing (approximate, May 2025)
    # Storage: $0.033 per GB per month
    # Read: $4 per 1M read units (1 read unit = 1 query for up to 5 results, ~20 for top-100)
    # Write: $2 per 1M write units

    gb_storage = (num_vectors * dim * bytes_per_float) / (1024**3)
    monthly_storage_cost = gb_storage * 0.033

    monthly_queries = daily_queries * 30
    read_units = monthly_queries  # roughly 1 RU per query for small k
    monthly_read_cost = read_units * 4 / 1_000_000

    return {
        "vectors": num_vectors,
        "storage_gb": gb_storage,
        "monthly_storage_usd": monthly_storage_cost,
        "monthly_read_usd": monthly_read_cost,
        "monthly_total_usd": monthly_storage_cost + monthly_read_cost
    }

for size in [10_000, 100_000, 1_000_000]:
    cost = estimate_pinecone_cost(size, daily_queries=5000)
    print(f"{size:>8,} vectors: ${cost['monthly_total_usd']:>8.2f}/month "
          f"(storage: ${cost['monthly_storage_usd']:.2f}, reads: ${cost['monthly_read_usd']:.2f})")
```

---

[[02-chromadb]] | [[04-qdrant]]
