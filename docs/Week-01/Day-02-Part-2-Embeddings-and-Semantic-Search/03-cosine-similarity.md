# Cosine Similarity

Cosine similarity is the most common metric for comparing embedding vectors. Understanding it from first principles lets you choose the right metric for the right problem and debug retrieval failures.

## Learning objectives

- Compute cosine similarity from first principles with NumPy
- Explain why cosine similarity is preferred over Euclidean distance for embeddings
- Implement efficient batch similarity computation
- Know when to use dot product vs. cosine similarity vs. L2 distance

---

## The formula

Cosine similarity measures the angle between two vectors, ignoring their magnitude:

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

Where `A · B` is the dot product and `‖A‖` is the L2 norm (magnitude).

- Result range: -1 to +1
- 1.0 = identical direction (most similar)
- 0.0 = orthogonal (unrelated)
- -1.0 = opposite direction (most dissimilar)

For embeddings trained with cosine loss, scores typically fall between 0.0 and 1.0 in practice.

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

# Toy example
v1 = np.array([1.0, 0.5, 0.2])
v2 = np.array([0.9, 0.6, 0.1])
v3 = np.array([-1.0, -0.5, -0.2])  # opposite of v1

print(f"v1 vs v2: {cosine_similarity(v1, v2):.4f}")  # ~0.998 (very similar)
print(f"v1 vs v3: {cosine_similarity(v1, v3):.4f}")  # -1.000 (opposite)
print(f"v1 vs [0,1,0]: {cosine_similarity(v1, np.array([0,1,0])):.4f}")  # partial similarity
```

---

## Why cosine, not Euclidean distance?

Euclidean distance measures how far apart two points are in space. Cosine similarity measures the angle between them.

```python
# Problem with Euclidean distance on embeddings
v_short = np.array([0.5, 0.5])   # short vector, 45° angle
v_long  = np.array([5.0, 5.0])   # long vector, same 45° angle

# Euclidean: they look very different
print(f"Euclidean: {np.linalg.norm(v_short - v_long):.3f}")  # 6.364

# Cosine: they're identical (same direction)
print(f"Cosine: {cosine_similarity(v_short, v_long):.3f}")   # 1.000
```

Embedding models produce vectors of varying magnitudes — a long document may produce a larger-magnitude vector than a short one even if they cover the same topic. Cosine similarity normalizes this out.

> [!info] When magnitude matters
> Some models (like CLIP for images) are trained to encode confidence in the vector magnitude. For these, dot product (without normalization) can be better than cosine similarity. For sentence transformers and OpenAI embedding models, cosine similarity is correct.

---

## Dot product as a fast approximation

If embeddings are pre-normalized (L2 norm = 1), dot product equals cosine similarity — no division needed. This is much faster at scale.

```python
def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)

# Normalized embeddings: dot product == cosine similarity
v1_norm = normalize(v1.reshape(1, -1))[0]
v2_norm = normalize(v2.reshape(1, -1))[0]

dot_sim = float(np.dot(v1_norm, v2_norm))
cos_sim = cosine_similarity(v1, v2)
print(f"Dot product (normalized): {dot_sim:.6f}")
print(f"Cosine similarity:        {cos_sim:.6f}")
# Both produce the same result
```

> [!tip] Pre-normalize at index time
> Normalize all your document embeddings when you insert them into the index. At query time, normalize the query vector and use dot product. This halves the computation vs. computing full cosine similarity.

---

## Batch similarity computation

Computing similarity between one query and many documents efficiently:

```python
def top_k_similar(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    k: int = 5
) -> list[tuple[int, float]]:
    """Return indices and scores of top-k most similar documents."""
    # Normalize both
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    d_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    d_normalized = doc_embeddings / (d_norms + 1e-8)

    # Batch dot product: (1, D) @ (D, N) → (1, N)
    scores = np.dot(q_norm, d_normalized.T)

    # Get top-k indices (largest scores first)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]

# Example with real embeddings
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

docs = [
    "Python is a high-level programming language",
    "Django is a web framework for Python",
    "The Eiffel Tower is in Paris",
    "Machine learning requires large datasets",
    "Neural networks are inspired by the brain",
    "France is a country in Western Europe"
]

print("Embedding documents...")
doc_embeddings = np.array([get_embedding(doc) for doc in docs])

query = "deep learning and AI"
query_emb = get_embedding(query)

results = top_k_similar(query_emb, doc_embeddings, k=3)
print(f"\nTop 3 results for '{query}':")
for rank, (idx, score) in enumerate(results, 1):
    print(f"  {rank}. [{score:.3f}] {docs[idx]}")
```

---

## Similarity matrix for clustering

Computing all pairwise similarities between a set of documents:

```python
def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute N×N cosine similarity matrix."""
    normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return normalized @ normalized.T

# Find the most similar pair in a document set
def most_similar_pair(docs: list[str], embeddings: np.ndarray) -> tuple[str, str, float]:
    sim_matrix = similarity_matrix(embeddings)
    np.fill_diagonal(sim_matrix, -1)  # ignore self-similarity
    idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    return docs[idx[0]], docs[idx[1]], float(sim_matrix[idx])

# Find the most dissimilar pair (useful for diverse sampling)
def most_diverse_pair(docs: list[str], embeddings: np.ndarray) -> tuple[str, str, float]:
    sim_matrix = similarity_matrix(embeddings)
    idx = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
    return docs[idx[0]], docs[idx[1]], float(sim_matrix[idx])

sim_mat = similarity_matrix(doc_embeddings)
d1, d2, score = most_similar_pair(docs, doc_embeddings)
print(f"\nMost similar pair (score={score:.3f}):\n  '{d1}'\n  '{d2}'")
```

---

## Distance metrics comparison

| Metric | Formula | Range | Use case |
|--------|---------|-------|----------|
| Cosine similarity | A·B / (‖A‖‖B‖) | [-1, 1] | Semantic similarity (standard) |
| Dot product | A·B | (-∞, +∞) | Normalized vectors, recommendation |
| Euclidean (L2) | ‖A - B‖ | [0, +∞) | Image features, when magnitude matters |
| Manhattan (L1) | Σ|Aᵢ - Bᵢ| | [0, +∞) | Sparse features, robustness to outliers |

```python
def compare_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "cosine_similarity": cosine_similarity(a, b),
        "dot_product": float(np.dot(a, b)),
        "euclidean_distance": float(np.linalg.norm(a - b)),
        "manhattan_distance": float(np.sum(np.abs(a - b)))
    }

# Real embedding comparison
emb1 = get_embedding("The cat sat on the mat")
emb2 = get_embedding("A feline rested on a rug")
emb3 = get_embedding("The stock market fell today")

print("Similar sentences (cat/feline):")
print(compare_metrics(np.array(emb1), np.array(emb2)))

print("\nDissimilar sentences (cat/stock):")
print(compare_metrics(np.array(emb1), np.array(emb3)))
```

---

## Common mistakes

> [!warning] Comparing embeddings from different models
> Embeddings from `text-embedding-3-small` and `all-MiniLM-L6-v2` live in completely different vector spaces. Cosine similarity between them is meaningless. Always use the same model for query and document embeddings.

> [!warning] Don't use raw cosine distance scores as absolute thresholds
> A score of 0.8 from model A is not equivalent to 0.8 from model B. Calibrate thresholds on your own data (e.g., "for this model, scores above 0.75 indicate relevant results").

> [!warning] Zero vectors cause division by zero
> If an input produces a zero vector (rare but possible with empty strings or degenerate inputs), cosine similarity is undefined. Always guard with a small epsilon: `norm + 1e-8`.

---

[[02-embedding-models]] | [[04-vector-search]]
