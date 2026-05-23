# ChromaDB Cheat Sheet

Quick reference for ChromaDB collection operations.

---

## Setup

```python
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# In-memory (testing)
client = chromadb.Client()

# Persistent (production)
client = chromadb.PersistentClient(path="./chroma_db")

# Embedding function
embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)
```

## Collection operations

```python
# Create or get existing
collection = client.get_or_create_collection(
    name="my_docs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},  # cosine or l2 (default: l2)
)

# Add documents
collection.add(
    documents=["First document text", "Second document text"],
    ids=["doc1", "doc2"],
    metadatas=[{"source": "web"}, {"source": "pdf"}],  # optional
)

# Query (auto-embeds query text)
results = collection.query(
    query_texts=["What is Python?"],
    n_results=5,
    where={"source": {"$eq": "pdf"}},           # Metadata filter
    where_document={"$contains": "machine learning"},  # Text filter
    include=["documents", "metadatas", "distances"],
)

docs = results["documents"][0]      # list[str]
ids = results["ids"][0]             # list[str]
distances = results["distances"][0] # list[float] (smaller = more similar for cosine)
metadatas = results["metadatas"][0] # list[dict]

# Query with pre-computed embeddings (skip re-embedding)
collection.query(
    query_embeddings=[[0.1, 0.2, ...]],  # Pre-computed vector
    n_results=5,
)

# Update
collection.update(ids=["doc1"], documents=["Updated text"])
collection.upsert(ids=["doc3"], documents=["New doc"])

# Delete
collection.delete(ids=["doc1"])
collection.delete(where={"source": {"$eq": "old"}})

# Count
print(collection.count())

# Get all (small collections only)
all_docs = collection.get(include=["documents", "metadatas"])
```

## Metadata filter operators

```python
# Equality
where={"category": {"$eq": "faq"}}

# Not equal
where={"category": {"$ne": "draft"}}

# Comparison (for numeric metadata)
where={"year": {"$gte": 2023}}
where={"year": {"$lt": 2024}}

# In a list
where={"category": {"$in": ["faq", "guide"]}}

# Logical AND/OR
where={"$and": [
    {"category": {"$eq": "faq"}},
    {"year": {"$gte": 2023}},
]}
```

## Distance vs similarity

ChromaDB returns `distances` — lower is more similar for cosine space.

```python
# Convert distance to similarity score (0–1, higher = more similar)
similarity = 1 - distance  # For cosine distance
```

## Managing collections

```python
# List all collections
client.list_collections()

# Delete a collection
client.delete_collection("my_docs")

# Reset (delete everything — irreversible)
client.reset()
```
