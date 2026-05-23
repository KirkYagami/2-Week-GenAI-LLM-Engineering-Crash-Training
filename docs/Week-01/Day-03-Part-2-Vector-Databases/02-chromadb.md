# ChromaDB

ChromaDB is the fastest way to get a vector database running. It requires no server, no API keys, and no configuration — ideal for local development, prototyping, and small-to-medium production deployments.

## Learning objectives

- Use both in-memory and persistent ChromaDB clients
- Create collections with custom embedding functions
- Perform filtered and unfiltered queries
- Manage collections (add, update, delete, list)

---

## Client modes

```python
import chromadb
from chromadb.utils import embedding_functions

# 1. In-memory (data lost on process exit)
client = chromadb.Client()

# 2. Persistent (data saved to disk)
client = chromadb.PersistentClient(path="./my_chroma_db")

# 3. HTTP client (connect to a running ChromaDB server)
# client = chromadb.HttpClient(host="localhost", port=8000)
```

---

## Creating collections

```python
import os

# Built-in OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Built-in Sentence Transformers (no API key needed)
st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create collection with distance metric
collection = client.create_collection(
    name="my_documents",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}   # "cosine", "l2", or "ip"
)

# Get existing or create new
collection = client.get_or_create_collection(
    name="my_documents",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}
)

print(f"Collection: {collection.name}, Count: {collection.count()}")
```

---

## Adding documents

```python
# Simple text add — ChromaDB calls the embedding function automatically
collection.add(
    ids=["doc-1", "doc-2", "doc-3"],
    documents=[
        "Python is a versatile programming language",
        "FastAPI is a modern web framework for Python",
        "The Eiffel Tower is 330 meters tall"
    ],
    metadatas=[
        {"source": "python-wiki", "category": "programming", "year": 2024},
        {"source": "fastapi-docs", "category": "programming", "year": 2024},
        {"source": "eiffel-wiki", "category": "tourism", "year": 2023}
    ]
)

# Add with pre-computed embeddings (skip the embedding function)
import numpy as np
my_embeddings = np.random.rand(2, 1536).tolist()  # your pre-computed embeddings

collection.add(
    ids=["custom-1", "custom-2"],
    embeddings=my_embeddings,
    documents=["document text 1", "document text 2"],
    metadatas=[{"source": "custom"}, {"source": "custom"}]
)

print(f"Total documents: {collection.count()}")
```

---

## Querying

```python
# Basic semantic search
results = collection.query(
    query_texts=["web development in Python"],
    n_results=2,
    include=["documents", "metadatas", "distances"]
)

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    similarity = 1 - dist  # for cosine space: distance = 1 - similarity
    print(f"[{similarity:.3f}] {doc[:60]}... | source: {meta['source']}")

# Filtered query — only return programming documents
filtered_results = collection.query(
    query_texts=["Python frameworks"],
    n_results=3,
    where={"category": {"$eq": "programming"}},  # metadata filter
    include=["documents", "metadatas", "distances"]
)

# Filter by year range
recent_results = collection.query(
    query_texts=["programming"],
    n_results=5,
    where={"year": {"$gte": 2024}},
    include=["documents", "metadatas"]
)

# Full-text search (keyword search within documents)
keyword_results = collection.query(
    query_texts=["tower"],
    n_results=3,
    where_document={"$contains": "tall"},  # document must contain "tall"
    include=["documents", "metadatas"]
)
```

### ChromaDB filter operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equal | `{"category": {"$eq": "tech"}}` |
| `$ne` | Not equal | `{"status": {"$ne": "deleted"}}` |
| `$gt`, `$gte` | Greater than (or equal) | `{"year": {"$gte": 2023}}` |
| `$lt`, `$lte` | Less than (or equal) | `{"score": {"$lt": 0.5}}` |
| `$in` | Value in list | `{"tag": {"$in": ["ml", "ai"]}}` |
| `$nin` | Value not in list | `{"tag": {"$nin": ["spam"]}}` |
| `$and` | Logical AND | `{"$and": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}` |
| `$or` | Logical OR | `{"$or": [{"cat": "ml"}, {"cat": "ai"}]}` |

---

## CRUD operations

```python
# Get specific documents by ID
get_results = collection.get(
    ids=["doc-1", "doc-2"],
    include=["documents", "metadatas"]
)
print(get_results)

# Update document text and/or metadata
collection.update(
    ids=["doc-1"],
    documents=["Python is a high-level, versatile programming language created by Guido van Rossum"],
    metadatas=[{"source": "python-wiki", "category": "programming", "year": 2025}]
)

# Upsert (insert if new, update if exists)
collection.upsert(
    ids=["doc-1", "doc-new"],
    documents=["Updated Python doc", "Brand new document"],
    metadatas=[{"source": "wiki", "year": 2025}, {"source": "new", "year": 2025}]
)

# Delete by ID
collection.delete(ids=["custom-1", "custom-2"])

# Delete by metadata filter
collection.delete(where={"category": {"$eq": "tourism"}})

print(f"After deletes: {collection.count()}")

# List all collections
all_collections = client.list_collections()
print(f"Collections: {[c.name for c in all_collections]}")

# Delete a collection
client.delete_collection("my_documents")
```

---

## Building a document store with ChromaDB

```python
import os
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

class ChromaDocStore:
    def __init__(self, collection_name: str, persist_path: str = "./chroma"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def add(self, docs: list[dict]) -> int:
        """Add documents. Each doc must have: id, text, and optional metadata."""
        self.collection.upsert(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[{k: v for k, v in d.items() if k not in {"id", "text"}} for d in docs]
        )
        return len(docs)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None,
        min_similarity: float = 0.5
    ) -> list[dict]:
        kwargs = {"query_texts": [query], "n_results": k,
                  "include": ["documents", "metadatas", "distances"]}
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)
        output = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            sim = 1 - dist
            if sim >= min_similarity:
                output.append({"text": doc, "similarity": sim, **meta})
        return output

    def answer(self, question: str, k: int = 4, filters: dict | None = None) -> str:
        retrieved = self.search(question, k=k, filters=filters)
        if not retrieved:
            return "I don't have relevant information to answer this question."

        context = "\n\n".join(f"[{r.get('source', 'unknown')}]: {r['text']}" for r in retrieved)
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer based only on the context. Cite sources. Say 'I don't know' if not in context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQ: {question}"}
            ],
            max_tokens=400, temperature=0.0
        )
        return response.choices[0].message.content

# Usage
store = ChromaDocStore("company_knowledge")
store.add([
    {"id": "p1", "text": "Remote work is permitted 3 days/week.", "source": "policy.pdf", "dept": "hr"},
    {"id": "p2", "text": "PTO accrues at 1.67 days/month.", "source": "policy.pdf", "dept": "hr"},
    {"id": "e1", "text": "Engineering uses Jira for project tracking.", "source": "eng-guide.pdf", "dept": "engineering"},
])

print(store.answer("How many remote days can I work?"))
print()
print(store.answer("What tools does engineering use?", filters={"dept": {"$eq": "engineering"}}))
```

---

## Common mistakes

> [!warning] ChromaDB distance is NOT similarity
> ChromaDB returns `distances`, not similarities. For cosine space: `similarity = 1 - distance`. A distance of 0 = identical, distance of 2 = opposite. Don't use the raw distance value as a quality threshold — convert to similarity first.

> [!warning] In-memory client loses data on restart
> `chromadb.Client()` stores data in memory only. Always use `chromadb.PersistentClient(path="...")` for anything you need to keep between runs.

> [!warning] `n_results` cannot exceed the collection size
> If you request `n_results=5` but only have 3 documents, ChromaDB raises an error. Guard with `min(k, collection.count())`.

---

[[01-vector-db-overview]] | [[03-pinecone]]
