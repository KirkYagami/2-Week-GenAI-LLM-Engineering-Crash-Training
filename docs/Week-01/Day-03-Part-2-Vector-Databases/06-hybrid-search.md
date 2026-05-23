# Hybrid Search

Dense semantic search misses exact term matches. Sparse keyword search misses synonyms and paraphrases. Hybrid search combines both — it consistently outperforms either approach alone on BEIR benchmarks by 5–15%.

## Learning objectives

- Implement BM25 sparse retrieval from scratch
- Combine dense and sparse scores with Reciprocal Rank Fusion
- Use Qdrant's built-in sparse vector support
- Choose the right alpha (dense/sparse balance) for your domain

---

## Why hybrid search

```
Query: "What is gpt-4o's context window?"

Dense search finds:
- "GPT-4o supports conversations up to 128K tokens long"  ← correct
- "Context windows determine how much text the model can process"  ← related

Sparse (BM25) search finds:
- "gpt-4o-2024-11-20 has context_window=128000"  ← exact match
- "The GPT-4o API supports max 128000 tokens per request"  ← exact match

Hybrid (both):
- Gets the exact match AND the semantically related content
- More robust to variation in how users phrase queries
```

---

## BM25 from scratch

Understanding the algorithm before using the library:

```python
import math
from collections import Counter
import re

class BM25:
    """BM25 sparse retrieval — Robertson et al., 1994."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1   # term frequency saturation (higher = more TF weight)
        self.b = b     # length normalization (0 = no normalization, 1 = full)
        self.docs: list[list[str]] = []
        self.df: Counter = Counter()  # document frequency per term
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def fit(self, corpus: list[str]) -> None:
        self.docs = [self._tokenize(doc) for doc in corpus]
        self.avgdl = sum(len(doc) for doc in self.docs) / len(self.docs)

        # Document frequency
        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1

        # IDF scores
        N = len(self.docs)
        for term, freq in self.df.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query: str) -> list[float]:
        query_terms = self._tokenize(query)
        scores = []
        for doc_tokens in self.docs:
            tf = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0
            for term in query_terms:
                if term in self.idf:
                    idf = self.idf[term]
                    tf_normalized = (tf[term] * (self.k1 + 1)) / (
                        tf[term] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    )
                    score += idf * tf_normalized
            scores.append(score)
        return scores

# Test BM25
corpus = [
    "Python is a high-level programming language with simple syntax",
    "Machine learning uses statistical models to learn from data",
    "Deep learning neural networks have multiple hidden layers",
    "Python libraries like NumPy and pandas enable data analysis",
    "Natural language processing handles text classification tasks"
]

bm25 = BM25()
bm25.fit(corpus)

query = "Python data analysis libraries"
scores = bm25.score(query)
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
print(f"BM25 scores for '{query}':")
for idx, score in ranked:
    print(f"  [{score:.3f}] {corpus[idx][:60]}...")
```

---

## Reciprocal Rank Fusion (RRF)

RRF merges ranked lists from different retrieval systems without needing to calibrate score scales.

```python
from typing import Any

def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60   # RRF constant — 60 is the typical default
) -> list[tuple[str, float]]:
    """
    result_lists: list of [(doc_id, score), ...] from different retrieval systems.
    Returns merged ranking sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}

    for result_list in result_lists:
        # Sort by score descending to get rank
        ranked = sorted(result_list, key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# Example: merge dense and sparse results
dense_results = [
    ("doc-0", 0.92), ("doc-3", 0.87), ("doc-2", 0.71),
    ("doc-1", 0.65), ("doc-4", 0.58)
]
sparse_results = [
    ("doc-3", 12.4), ("doc-0", 9.1), ("doc-4", 7.8),
    ("doc-1", 5.2), ("doc-2", 3.1)
]

merged = reciprocal_rank_fusion([dense_results, sparse_results])
print("Merged RRF ranking:")
for doc_id, rrf_score in merged[:5]:
    print(f"  {doc_id}: {rrf_score:.4f}")
```

---

## Full hybrid search pipeline

```python
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([item.embedding for item in resp.data])

class HybridSearchEngine:
    def __init__(self, corpus: list[dict]):
        """corpus: list of {"id": str, "text": str, **metadata}"""
        self.corpus = corpus
        texts = [d["text"] for d in corpus]
        ids = [d["id"] for d in corpus]

        # Build BM25 index
        self.bm25 = BM25()
        self.bm25.fit(texts)

        # Build dense index
        print("Embedding corpus...")
        self.embeddings = embed(texts)
        self.ids = ids

    def dense_search(self, query: str, k: int) -> list[tuple[str, float]]:
        query_emb = embed([query])[0]
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1)
        q_norm = np.linalg.norm(query_emb)
        scores = (self.embeddings @ query_emb) / (norms * q_norm + 1e-8)
        top_k = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in top_k]

    def sparse_search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.score(query)
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(self.ids[i], score) for i, score in top_k if score > 0]

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> list[dict]:
        """
        alpha: weight for dense results in linear combination of scores.
               0 = pure sparse, 1 = pure dense, 0.5 = balanced.
               Use RRF when alpha=None.
        """
        retrieve_k = k * 3  # over-retrieve for merging

        dense = self.dense_search(query, k=retrieve_k)
        sparse = self.sparse_search(query, k=retrieve_k)

        if alpha is None:
            merged = reciprocal_rank_fusion([dense, sparse])
        else:
            # Linear combination (requires score normalization)
            def normalize(results: list[tuple[str, float]]) -> dict[str, float]:
                if not results:
                    return {}
                scores = [s for _, s in results]
                max_s, min_s = max(scores), min(scores)
                denom = max_s - min_s if max_s != min_s else 1.0
                return {id: (s - min_s) / denom for id, s in results}

            dense_norm = normalize(dense)
            sparse_norm = normalize(sparse)

            all_ids = set(dense_norm) | set(sparse_norm)
            combined = {
                id: alpha * dense_norm.get(id, 0) + (1 - alpha) * sparse_norm.get(id, 0)
                for id in all_ids
            }
            merged = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        # Return top-k with document info
        results = []
        for doc_id, score in merged[:k]:
            doc = next((d for d in self.corpus if d["id"] == doc_id), None)
            if doc:
                results.append({**doc, "hybrid_score": score})

        return results

# Test it
corpus = [
    {"id": "py-1", "text": "Python is a high-level programming language created by Guido van Rossum"},
    {"id": "py-2", "text": "Python libraries NumPy and pandas are used for data analysis"},
    {"id": "ml-1", "text": "Machine learning algorithms learn patterns from training data"},
    {"id": "ml-2", "text": "Deep learning uses neural networks with many hidden layers"},
    {"id": "api-1", "text": "FastAPI is a modern web framework for building APIs with Python"},
    {"id": "tower", "text": "The Eiffel Tower in Paris was built in 1889 by Gustave Eiffel"},
]

engine = HybridSearchEngine(corpus)

queries = [
    "Python data analysis",      # benefits from exact "Python" match
    "NumPy pandas libraries",    # exact keyword match important
    "how neural nets learn",     # benefits from semantic understanding
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = engine.hybrid_search(query, k=3, alpha=0.5)
    for r in results:
        print(f"  [{r['hybrid_score']:.3f}] {r['id']}: {r['text'][:60]}...")
```

---

## Choosing alpha

```python
def evaluate_alpha(
    engine: HybridSearchEngine,
    eval_set: list[tuple[str, str]],  # (query, expected_doc_id)
    alpha_values: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
) -> dict:
    """Find the best alpha for your dataset."""
    results = {}
    for alpha in alpha_values:
        hits = 0
        for query, expected_id in eval_set:
            top_ids = [r["id"] for r in engine.hybrid_search(query, k=3, alpha=alpha)]
            if expected_id in top_ids:
                hits += 1
        results[alpha] = hits / len(eval_set)
        print(f"alpha={alpha}: Recall@3 = {results[alpha]:.0%}")

    best_alpha = max(results, key=results.get)
    print(f"\nBest alpha: {best_alpha} (Recall@3 = {results[best_alpha]:.0%})")
    return results

# Example evaluation set
eval_set = [
    ("Python data analysis libraries", "py-2"),
    ("Who built the Eiffel Tower?", "tower"),
    ("web API framework Python", "api-1"),
    ("deep learning neural network", "ml-2"),
]

# evaluate_alpha(engine, eval_set)
```

> [!success] Rule of thumb for alpha
> - Queries with technical jargon, product names, version numbers → alpha closer to 0.25 (more sparse weight)
> - Natural language, conceptual questions → alpha closer to 0.75 (more dense weight)
> - Mixed user base → alpha=0.5 + RRF is a safe default

---

## Qdrant native sparse-dense

Qdrant 1.7+ supports sparse vectors natively, enabling true server-side hybrid search without client-side merging.

```python
from qdrant_client.models import SparseVector, SparseVectorParams

# Create collection with both dense and sparse vectors
client.recreate_collection(
    collection_name="hybrid_docs",
    vectors_config={
        "dense": VectorParams(size=1536, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams()
    }
)

# Upsert with both vector types
from pinecone_text.sparse import BM25Encoder
corpus_texts = [d["text"] for d in corpus]
bm25_enc = BM25Encoder().fit(corpus_texts)

for i, doc in enumerate(corpus):
    dense_emb = embed([doc["text"]])[0]
    sparse_emb = bm25_enc.encode_documents([doc["text"]])[0]

    client.upsert(
        collection_name="hybrid_docs",
        points=[PointStruct(
            id=i,
            vector={
                "dense": dense_emb,
                "sparse": SparseVector(
                    indices=sparse_emb["indices"],
                    values=sparse_emb["values"]
                )
            },
            payload={k: v for k, v in doc.items() if k != "text"}
        )]
    )
```

---

[[05-indexing-and-filtering]] | [[07-practice-exercises]]
