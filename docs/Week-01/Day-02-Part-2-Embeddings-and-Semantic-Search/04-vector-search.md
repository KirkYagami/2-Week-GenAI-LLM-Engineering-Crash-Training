# Vector Search

Brute-force cosine similarity works for thousands of documents. At millions, you need approximate nearest neighbor (ANN) search — trading a small amount of accuracy for 100–1000× faster retrieval. FAISS is the standard library for this.

## Learning objectives

- Implement brute-force similarity search from scratch
- Build a FAISS index for approximate nearest neighbor search
- Understand the IVF and HNSW index types
- Add metadata filtering to vector search

---

## Brute-force search (the baseline)

For small collections (< 100K docs), brute-force exact search is fine and requires no index.

```python
import os
import numpy as np
from dataclasses import dataclass, field
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

class BruteForceVectorStore:
    def __init__(self):
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return np.array([item.embedding for item in response.data])

    def add_documents(self, documents: list[Document]) -> None:
        texts = [doc.text for doc in documents]
        new_embeddings = self._embed(texts)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(documents)
        print(f"Index size: {len(self.documents)} documents")

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        if self.embeddings is None or len(self.documents) == 0:
            return []

        query_emb = self._embed([query])[0]

        # Normalize for cosine similarity via dot product
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        doc_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        docs_normalized = self.embeddings / (doc_norms + 1e-8)

        scores = docs_normalized @ query_norm  # (N,)
        top_k_indices = np.argsort(scores)[::-1][:k]

        return [(self.documents[i], float(scores[i])) for i in top_k_indices]

# Example usage
store = BruteForceVectorStore()
store.add_documents([
    Document("1", "Python is a versatile programming language used for web, data science, and AI"),
    Document("2", "FastAPI is a modern, fast web framework for Python"),
    Document("3", "Pandas provides data structures and analysis tools for Python"),
    Document("4", "Neural networks learn by adjusting weights through backpropagation"),
    Document("5", "The Eiffel Tower stands 330 meters tall in Paris"),
    Document("6", "RAG combines retrieval with language model generation"),
])

results = store.search("what Python libraries exist for data analysis?", k=3)
for doc, score in results:
    print(f"  [{score:.3f}] {doc.text[:60]}...")
```

---

## FAISS for scalable search

FAISS (Facebook AI Similarity Search) handles millions of vectors efficiently using approximate nearest neighbor algorithms.

```python
import faiss
import numpy as np
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings_batch(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    response = client.embeddings.create(model=model, input=texts)
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
    # Normalize for inner product search (equivalent to cosine similarity)
    faiss.normalize_L2(embeddings)
    return embeddings

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Natural language processing handles text understanding",
    "Computer vision processes images and video",
    "Reinforcement learning trains agents through rewards",
    "Transfer learning reuses pretrained model weights",
    "BERT is a transformer model for text understanding",
    "GPT models generate text autoregressively",
]

# Build FAISS index
doc_embeddings = get_embeddings_batch(documents)
dimension = doc_embeddings.shape[1]  # 1536

# IndexFlatIP = Flat (brute-force) + Inner Product (cosine if normalized)
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)
print(f"Index contains {index.ntotal} vectors of dimension {dimension}")

# Search
def search_faiss(query: str, k: int = 3) -> list[tuple[str, float]]:
    query_emb = get_embeddings_batch([query])
    scores, indices = index.search(query_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:  # -1 means no result
            results.append((documents[idx], float(score)))
    return results

results = search_faiss("How do transformers understand language?", k=3)
for text, score in results:
    print(f"  [{score:.3f}] {text}")
```

---

## Index types

FAISS offers several index types with different speed/accuracy tradeoffs:

```python
# 1. IndexFlatIP — exact search, no compression
# Best for: < 100K vectors, when accuracy is critical
flat_index = faiss.IndexFlatIP(dimension)

# 2. IndexIVFFlat — inverted file index, approximate
# Splits space into 'nlist' Voronoi cells, searches 'nprobe' cells
# Best for: 100K–10M vectors
nlist = 100  # number of clusters (typical: sqrt(N))
ivf_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, nlist, faiss.METRIC_INNER_PRODUCT)
ivf_index.train(doc_embeddings)  # must train before adding
ivf_index.add(doc_embeddings)
ivf_index.nprobe = 10  # search 10 clusters (higher = more accurate, slower)

# 3. IndexHNSWFlat — Hierarchical Navigable Small World graph
# No training required, very fast at query time, more memory
# Best for: general purpose, latency-sensitive applications
hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M (connections per node)
hnsw_index.hnsw.efConstruction = 40  # build quality (higher = better, slower to build)
hnsw_index.add(doc_embeddings)
hnsw_index.hnsw.efSearch = 16  # query quality (higher = more accurate)

def benchmark_indices(query: str) -> None:
    import time
    query_emb = get_embeddings_batch([query])

    for name, idx in [("Flat", flat_index), ("IVF", ivf_index), ("HNSW", hnsw_index)]:
        start = time.perf_counter()
        scores, indices = idx.search(query_emb, 3)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{name}: {elapsed:.2f}ms, top result: {documents[indices[0][0]][:40]}...")
```

| Index | Build time | Query time | Accuracy | Memory |
|-------|-----------|-----------|---------|--------|
| `IndexFlatIP` | Fast | O(N) | 100% | 4 bytes × D × N |
| `IndexIVFFlat` | Medium | O(√N) | 95–99% | 4 bytes × D × N + overhead |
| `IndexHNSWFlat` | Slow | O(log N) | 98–99% | ~2× flat memory |

> [!tip] Start with IndexFlatIP
> For a course project or prototype with < 50K documents, `IndexFlatIP` is exact and simple. Move to IVF or HNSW only when query latency or memory becomes a constraint.

---

## Persisting and loading an index

```python
# Save index to disk
faiss.write_index(index, "my_index.faiss")

# Load later
loaded_index = faiss.read_index("my_index.faiss")
print(f"Loaded index with {loaded_index.ntotal} vectors")

# Save document store separately (FAISS only stores vectors, not metadata)
import json
with open("my_docs.json", "w") as f:
    json.dump(documents, f)

with open("my_docs.json") as f:
    loaded_docs = json.load(f)
```

---

## Metadata filtering

FAISS doesn't support metadata filters natively — you filter after retrieval.

```python
from dataclasses import dataclass, field

@dataclass
class IndexedDocument:
    id: str
    text: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)

class FilterableVectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[IndexedDocument] = []

    def add(self, doc: IndexedDocument) -> None:
        emb = doc.embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(emb)
        self.index.add(emb)
        self.documents.append(doc)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        metadata_filter: dict | None = None,
        overretrieve_factor: int = 3
    ) -> list[tuple[IndexedDocument, float]]:
        # Retrieve more results to account for filtering
        retrieve_k = k * overretrieve_factor if metadata_filter else k

        query_emb = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, min(retrieve_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            if metadata_filter:
                if not all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
            results.append((doc, float(score)))
            if len(results) == k:
                break

        return results

# Example: filter by category
store = FilterableVectorStore(dimension=1536)
# Add docs with metadata...
# results = store.search(query_emb, k=5, metadata_filter={"category": "science", "year": 2024})
```

> [!warning] Overretrieve when filtering
> If you retrieve k=5 and filter aggressively, you may end up with 0–2 results. Retrieve `k × overretrieve_factor` candidates before filtering. A factor of 3–10 is typical depending on how selective the filter is.

---

## What's next

Brute-force and FAISS give you the retrieval layer. The next step is combining retrieval with generation in a full semantic search pipeline.

[[03-cosine-similarity]] | [[05-semantic-search-pipeline]]
