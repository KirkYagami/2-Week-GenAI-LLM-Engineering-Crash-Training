# Qdrant

Qdrant is an open-source vector database written in Rust. It offers the most expressive metadata filtering language of any vector database, runs locally in-memory or via Docker, and has a managed cloud option. It's the go-to choice for teams that need advanced filtering or self-hosted control.

## Learning objectives

- Create Qdrant collections with custom HNSW configuration
- Upsert points with structured payloads
- Write complex metadata filters using Qdrant's filter DSL
- Use Qdrant's named vectors for multi-vector collections

---

## Setup

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range, MatchAny
)

# In-memory (no Docker needed, great for testing)
client = QdrantClient(":memory:")

# Local persistent storage
# client = QdrantClient(path="./qdrant_storage")

# Remote (Docker or cloud)
# client = QdrantClient(url="http://localhost:6333")
# client = QdrantClient(url="https://your-cluster.qdrant.io", api_key=os.getenv("QDRANT_API_KEY"))
```

---

## Creating collections

```python
from qdrant_client.models import HnswConfigDiff

# Basic collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,           # dimension of your embeddings
        distance=Distance.COSINE
    )
)

# Collection with custom HNSW settings
client.recreate_collection(
    collection_name="documents_hnsw",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE,
        hnsw_config=HnswConfigDiff(
            m=16,                    # connections per node
            ef_construct=200,        # construction quality
            full_scan_threshold=10_000  # use brute force below this size
        )
    )
)

# Check collection info
info = client.get_collection("documents")
print(f"Collection: {info.config.params.vectors}")
print(f"Points: {info.points_count}")
```

---

## Upserting points

In Qdrant, stored items are called **points**. Each point has an ID, a vector, and an optional payload (metadata).

```python
import os
from openai import OpenAI
from qdrant_client.models import PointStruct

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in resp.data]

documents = [
    {"text": "Python is a high-level programming language", "source": "wiki.txt", "category": "tech", "year": 2024, "rating": 4.8},
    {"text": "FastAPI builds APIs with Python and type hints", "source": "docs.txt", "category": "tech", "year": 2024, "rating": 4.5},
    {"text": "The Eiffel Tower is 330m tall in Paris", "source": "travel.txt", "category": "travel", "year": 2023, "rating": 4.9},
    {"text": "RAG combines retrieval with generation", "source": "ai-guide.txt", "category": "ai", "year": 2024, "rating": 4.7},
    {"text": "Machine learning enables statistical modeling", "source": "ml-book.txt", "category": "ai", "year": 2023, "rating": 4.3},
]

embeddings = embed([d["text"] for d in documents])

# Upsert points (integer IDs required in Qdrant, or use UUIDs)
points = [
    PointStruct(
        id=i,
        vector=emb,
        payload={k: v for k, v in doc.items() if k != "text"}
        # payload can contain any JSON-serializable data
    )
    for i, (doc, emb) in enumerate(zip(documents, embeddings))
]

client.upsert(collection_name="documents", points=points)
print(f"Upserted {len(points)} points")
```

---

## Querying

```python
def search_qdrant(
    query: str,
    collection: str = "documents",
    k: int = 5,
    query_filter: Filter | None = None,
    score_threshold: float | None = None
) -> list[dict]:
    query_emb = embed([query])[0]

    results = client.search(
        collection_name=collection,
        query_vector=query_emb,
        limit=k,
        query_filter=query_filter,
        score_threshold=score_threshold,
        with_payload=True
    )

    return [
        {"id": r.id, "score": r.score, **r.payload}
        for r in results
    ]

# Basic search
results = search_qdrant("Python programming", k=3)
for r in results:
    print(f"[{r['score']:.3f}] id={r['id']}, source={r['source']}")

# Search with score threshold (no results below this)
high_quality = search_qdrant(
    "machine learning AI",
    score_threshold=0.7
)
```

---

## Qdrant filter language

Qdrant's filter DSL is the most expressive of any vector database.

```python
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny,
    Range, IsNullCondition, IsEmptyCondition
)

# Simple equality
tech_filter = Filter(
    must=[FieldCondition(key="category", match=MatchValue(value="tech"))]
)

# Multiple conditions (AND)
filtered_results = search_qdrant(
    "Python tutorial",
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="tech")),
            FieldCondition(key="year", range=Range(gte=2024))
        ]
    )
)

# OR conditions
multi_category = Filter(
    should=[
        FieldCondition(key="category", match=MatchValue(value="tech")),
        FieldCondition(key="category", match=MatchValue(value="ai"))
    ],
    minimum_should_match=1
)

# Match any in list
tag_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchAny(any=["tech", "ai"])
        )
    ]
)

# Range filter
high_rated = Filter(
    must=[
        FieldCondition(key="rating", range=Range(gte=4.5, lt=5.0))
    ]
)

# Combined example: category="ai" AND year>=2024 AND rating>=4.5
combined = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="ai")),
        FieldCondition(key="year", range=Range(gte=2024)),
        FieldCondition(key="rating", range=Range(gte=4.5))
    ]
)

results = search_qdrant("retrieval augmented generation", query_filter=combined)
print(f"Combined filter results: {len(results)}")
```

---

## CRUD operations

```python
from qdrant_client.models import PointIdsList, PayloadSelector, SetPayload

# Fetch points by ID
fetched = client.retrieve(
    collection_name="documents",
    ids=[0, 1, 2],
    with_payload=True,
    with_vectors=False
)
for point in fetched:
    print(f"id={point.id}: {point.payload}")

# Update payload only (without re-embedding)
client.set_payload(
    collection_name="documents",
    payload={"status": "verified", "updated_at": "2025-05-23"},
    points=[0, 1]
)

# Delete specific points
client.delete(
    collection_name="documents",
    points_selector=PointIdsList(points=[4])
)

# Delete by filter
client.delete(
    collection_name="documents",
    points_selector=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="travel"))]
    )
)

print(f"Remaining: {client.count('documents').count}")

# Scroll through all points (useful for data export)
all_points, offset = [], None
while True:
    batch, offset = client.scroll(
        collection_name="documents",
        limit=100,
        offset=offset,
        with_payload=True
    )
    all_points.extend(batch)
    if offset is None:
        break

print(f"All points: {len(all_points)}")
```

---

## Payload indexing for fast filtering

By default, Qdrant scans all points for metadata filters. For large collections (>100K points), create payload indexes for frequently filtered fields.

```python
from qdrant_client.models import PayloadSchemaType

# Index a keyword field for fast exact matching
client.create_payload_index(
    collection_name="documents",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

# Index a numeric field for fast range queries
client.create_payload_index(
    collection_name="documents",
    field_name="year",
    field_schema=PayloadSchemaType.INTEGER
)

# Index a float field for rating queries
client.create_payload_index(
    collection_name="documents",
    field_name="rating",
    field_schema=PayloadSchemaType.FLOAT
)

# List indexes
info = client.get_collection("documents")
print(f"Payload schema: {info.payload_schema}")
```

> [!tip] Always index high-cardinality filter fields
> For a collection of 500K documents with `category` filter used on 90% of queries, a keyword index reduces filter time from O(N) to O(log N). The index is built once and maintained incrementally.

---

## Named vectors (multi-vector search)

Qdrant supports multiple vectors per point — useful when you want to embed different representations of the same document (e.g., title + body separately).

```python
from qdrant_client.models import VectorParams, Distance

# Create collection with multiple named vector fields
client.recreate_collection(
    collection_name="multi_vec_docs",
    vectors_config={
        "title": VectorParams(size=1536, distance=Distance.COSINE),
        "body": VectorParams(size=1536, distance=Distance.COSINE),
    }
)

# Upsert with named vectors
title_embs = embed(["Python language overview"])
body_embs = embed(["Python is a high-level language. It supports OOP, functional, and procedural paradigms..."])

client.upsert(
    collection_name="multi_vec_docs",
    points=[
        PointStruct(
            id=1,
            vector={"title": title_embs[0], "body": body_embs[0]},
            payload={"source": "python-wiki.txt"}
        )
    ]
)

# Search by a specific named vector
results = client.search(
    collection_name="multi_vec_docs",
    query_vector=("body", embed(["programming paradigms"])[0]),
    limit=3
)
```

---

[[03-pinecone]] | [[05-indexing-and-filtering]]
