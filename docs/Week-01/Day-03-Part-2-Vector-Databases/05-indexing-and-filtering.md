# Indexing and Filtering

Understanding how vector databases index and filter data lets you design schemas that are fast at production scale. The wrong schema causes slow queries, high memory usage, or missed results under filters.

## Learning objectives

- Design metadata schemas for efficient filtering
- Understand pre-filtering vs. post-filtering and when each applies
- Implement payload indexing in Qdrant
- Choose index parameters for your scale

---

## Pre-filtering vs. post-filtering

There are two strategies for combining vector search with metadata filters:

**Post-filtering:** Search all vectors, get top-k, then filter the results by metadata. Simple but wasteful — if your filter is selective (only 1% of documents match), you may retrieve many irrelevant results and end up with fewer than k final results.

**Pre-filtering:** Apply the metadata filter first to get a candidate set, then perform ANN search only within that set. More efficient but requires payload indexing.

```
Post-filtering:
  ANN search → top-1000 → filter by category="tech" → return top-5
  Problem: if only 1% are "tech", your top-1000 contains ~10 "tech" docs — you only return 5 out of a possible large set

Pre-filtering (Qdrant's approach):
  Filter by category="tech" → candidate set of 10,000 docs → ANN on those 10,000 → top-5
  Better: ANN searches only relevant documents, returns accurate results
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(":memory:")

# Qdrant automatically chooses pre or post filtering based on:
# - Selectivity of the filter (what fraction of points match)
# - Whether payload indexes exist for the filter fields
# - Full scan threshold setting

# To force pre-filtering: ensure payload indexes exist
```

---

## Metadata schema design

Your metadata schema determines what filters are possible and how fast they run.

```python
# Good schema — designed for typical queries
DOCUMENT_PAYLOAD = {
    # String fields for exact matching
    "source": "policy-handbook.pdf",          # filename or document ID
    "department": "engineering",              # allows dept-based filtering
    "doc_type": "policy",                     # "policy", "faq", "procedure"
    "language": "en",                         # for multilingual systems

    # Numeric fields for range queries
    "year": 2024,
    "page_number": 12,
    "chunk_index": 3,                         # position within document
    "token_count": 485,

    # Boolean fields
    "is_public": True,                        # public vs. internal
    "is_archived": False,

    # Array fields for multi-value matching
    "tags": ["compliance", "hr", "leave"],    # multi-tag documents

    # Timestamp (stored as Unix epoch for range queries)
    "updated_at": 1716480000                  # Unix timestamp
}

# Anti-patterns to avoid:
BAD_SCHEMA = {
    # Don't store the full text in metadata — use the document field instead
    "full_text": "very long content...",      # waste of memory

    # Don't use nested objects for frequently filtered fields
    "nested": {"deeply": {"nested": "value"}},  # poor filter performance

    # Don't encode multiple values as a string
    "comma_tags": "ml, ai, nlp",             # use array instead
}
```

---

## Payload indexing in Qdrant

Without indexes, metadata filters require scanning all points. With indexes, filtering is O(log N).

```python
from qdrant_client.models import PayloadSchemaType, TextIndexParams, TokenizerType

# Create the collection first
from qdrant_client.models import VectorParams, Distance
client.recreate_collection(
    collection_name="enterprise_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Keyword index — for exact string matching (category, department, doc_type)
client.create_payload_index(
    collection_name="enterprise_docs",
    field_name="department",
    field_schema=PayloadSchemaType.KEYWORD
)

# Integer index — for numeric comparisons and range queries
client.create_payload_index(
    collection_name="enterprise_docs",
    field_name="year",
    field_schema=PayloadSchemaType.INTEGER
)

# Float index — for floating-point range queries (scores, ratings)
client.create_payload_index(
    collection_name="enterprise_docs",
    field_name="relevance_score",
    field_schema=PayloadSchemaType.FLOAT
)

# Boolean index — for is_archived, is_public etc.
client.create_payload_index(
    collection_name="enterprise_docs",
    field_name="is_public",
    field_schema=PayloadSchemaType.BOOL
)

# Full-text index — for keyword search within the document field
client.create_payload_index(
    collection_name="enterprise_docs",
    field_name="text",
    field_schema=TextIndexParams(
        type="text",
        tokenizer=TokenizerType.WORD,
        min_token_len=2,
        max_token_len=20,
        lowercase=True
    )
)

# Check indexes
info = client.get_collection("enterprise_docs")
print(f"Payload schema: {info.payload_schema}")
```

---

## Index selection by scale

| Collection size | Recommended approach |
|----------------|---------------------|
| < 1,000 vectors | No index needed — flat brute-force is fine |
| 1K – 100K vectors | HNSW with default settings, no payload index needed |
| 100K – 1M vectors | HNSW + payload indexes for frequently filtered fields |
| > 1M vectors | HNSW + all filter field indexes + quantization |

---

## Quantization for memory reduction

Vector quantization reduces memory usage by compressing float32 vectors.

```python
from qdrant_client.models import (
    ScalarQuantizationConfig, BinaryQuantizationConfig,
    QuantizationType, ScalarType
)

# Scalar quantization: float32 → int8 (4× memory reduction)
client.recreate_collection(
    collection_name="quantized_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ScalarQuantizationConfig(
        type=QuantizationType.SCALAR,
        quantile=0.99,     # clip outliers at 99th percentile
        always_ram=True    # keep quantized index in RAM for speed
    )
)

# Binary quantization: float32 → 1 bit (32× memory reduction)
# Best for high-dimensional vectors (1536+) — small quality loss
client.recreate_collection(
    collection_name="binary_quantized",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=BinaryQuantizationConfig(
        type=QuantizationType.BINARY,
        always_ram=True
    )
)
```

**Memory comparison for 1M vectors at 1536 dimensions:**
- float32: 1M × 1536 × 4 bytes = **5.9 GB**
- int8 (scalar quantization): ~**1.5 GB** (4× reduction)
- 1-bit (binary quantization): ~**185 MB** (32× reduction, some quality loss)

> [!tip] Use scalar quantization in production
> Scalar quantization (int8) provides 4× memory reduction with < 1% quality loss on most benchmarks. Binary quantization provides 32× reduction but loses 2–5% on NDCG — acceptable for some use cases. Always benchmark on your data before enabling quantization.

---

## ChromaDB filtering deep dive

ChromaDB's filter language, while simpler than Qdrant's, covers the common cases:

```python
import chromadb
from chromadb.utils import embedding_functions
import os

client = chromadb.Client()
ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
collection = client.get_or_create_collection("filtered_docs", embedding_function=ef)

collection.add(
    ids=["1", "2", "3", "4", "5"],
    documents=[
        "Python programming tutorial",
        "Machine learning basics",
        "Paris travel guide",
        "Advanced Python patterns",
        "Deep learning architectures"
    ],
    metadatas=[
        {"category": "tech", "difficulty": "beginner", "year": 2024, "rating": 4.2},
        {"category": "ai", "difficulty": "intermediate", "year": 2023, "rating": 4.5},
        {"category": "travel", "difficulty": "beginner", "year": 2024, "rating": 4.8},
        {"category": "tech", "difficulty": "advanced", "year": 2024, "rating": 4.7},
        {"category": "ai", "difficulty": "advanced", "year": 2024, "rating": 4.9},
    ]
)

# Complex filter: (category=ai OR category=tech) AND year=2024 AND rating>=4.5
results = collection.query(
    query_texts=["machine learning Python"],
    n_results=5,
    where={
        "$and": [
            {"$or": [
                {"category": {"$eq": "ai"}},
                {"category": {"$eq": "tech"}}
            ]},
            {"year": {"$eq": 2024}},
            {"rating": {"$gte": 4.5}}
        ]
    },
    include=["documents", "metadatas", "distances"]
)

print(f"Found {len(results['ids'][0])} results matching filter")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  {doc} | {meta}")
```

---

[[04-qdrant]] | [[06-hybrid-search]]
