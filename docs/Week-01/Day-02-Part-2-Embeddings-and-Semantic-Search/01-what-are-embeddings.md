# What Are Embeddings

An embedding is a fixed-length list of numbers that represents the semantic meaning of a piece of text. The model that generates them is trained so that semantically similar texts produce numerically similar vectors — text about dogs and text about puppies will be closer together than text about dogs and text about databases.

## Learning objectives

- Explain the relationship between tokens, embeddings, and meaning
- Understand why semantic similarity translates to geometric proximity
- Know the properties that make a good embedding space
- Describe how embeddings differ from one-hot encoding and bag-of-words

---

## From tokens to meaning

When a transformer processes text, it maps each token to an embedding vector. As the text passes through the attention layers, these vectors absorb context — the word "bank" in "river bank" ends up with a different vector than "bank" in "savings bank."

The final hidden state for each token captures its meaning in context. An embedding model distills the entire sequence down to a single vector by pooling these representations.

```
"The cat sat on the mat"
    ↓ tokenize
["The", "cat", "sat", "on", "the", "mat"]
    ↓ token embeddings (one vector per token)
[v_the, v_cat, v_sat, v_on, v_the, v_mat]  ← 6 × 1536-dim vectors (OpenAI small)
    ↓ transformer layers (attention over all tokens)
[c_the, c_cat, c_sat, c_on, c_the, c_mat]  ← context-aware vectors
    ↓ mean pooling
[0.12, -0.34, 0.87, ...]                    ← single 1536-dim sentence embedding
```

---

## The geometry of meaning

The key property: **direction encodes meaning, magnitude encodes confidence**. Vectors that point in similar directions represent similar concepts.

```python
import numpy as np

# Toy 2D embedding space (real embeddings are 768–3072 dims)
# Imagine each dimension captures a concept axis
embeddings = {
    "king":   np.array([0.95, 0.85]),   # high royalty, high male
    "queen":  np.array([0.95, -0.85]),  # high royalty, high female
    "man":    np.array([0.05, 0.90]),   # low royalty, high male
    "woman":  np.array([0.05, -0.90]),  # low royalty, high female
    "dog":    np.array([-0.80, 0.10]),  # unrelated to royalty/gender
}

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Classic analogy: king - man + woman ≈ queen
analogy = embeddings["king"] - embeddings["man"] + embeddings["woman"]
for word, vec in embeddings.items():
    sim = cosine_similarity(analogy, vec)
    print(f"  {word:8s}: {sim:.3f}")
# queen gets the highest score
```

---

## Why this is useful for search

Traditional keyword search: "python tutorial" matches documents containing "python" and "tutorial".

Semantic search: "python tutorial" matches documents about "learning to code in Python", "snake handling guide" (less similar), "beginner programming resources" (similar despite no overlapping words).

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

# Same concept, different words
query = "How do I connect Python to a database?"
candidates = [
    "SQLAlchemy tutorial for beginners",
    "Using psycopg2 to query PostgreSQL from Python",
    "French cooking techniques",
    "Database connection pooling in Python",
    "Python string formatting guide"
]

query_emb = embed(query)
for candidate in candidates:
    cand_emb = embed(candidate)
    sim = cosine_sim(query_emb, cand_emb)
    print(f"  {sim:.3f}  {candidate}")

# Expected output (approximate):
#   0.821  Using psycopg2 to query PostgreSQL from Python
#   0.798  Database connection pooling in Python
#   0.742  SQLAlchemy tutorial for beginners
#   0.541  Python string formatting guide
#   0.312  French cooking techniques
```

---

## Embedding dimensions and tradeoffs

| Dimension | Model | Properties |
|-----------|-------|------------|
| 384 | `all-MiniLM-L6-v2` | Fast, small, good for similarity |
| 768 | `all-mpnet-base-v2` | Better quality, still fast |
| 1,536 | `text-embedding-3-small` | OpenAI, good quality/cost |
| 3,072 | `text-embedding-3-large` | OpenAI, highest accuracy |

Higher dimensions capture more nuance but:
- Use more memory (3,072 float32 = 12KB per vector)
- Increase index build time
- Increase query latency

> [!tip] Dimension reduction
> OpenAI `text-embedding-3-*` models support `dimensions` parameter to reduce output size while preserving most semantic quality. `text-embedding-3-large` at `dimensions=256` outperforms `text-embedding-ada-002` (the old standard) at 1,536 dimensions — smaller and better.

```python
# Reduced-dimension embedding
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="Reduce me to 256 dimensions",
    dimensions=256
)
print(len(response.data[0].embedding))  # 256
```

---

## What embeddings are NOT good at

> [!warning] Embeddings don't encode recency
> "Latest iPhone model" has high semantic similarity to "iPhone 15 features" but the embedding doesn't know what "latest" means relative to today. For time-sensitive queries, combine semantic search with metadata filtering.

> [!warning] Embedding ≠ exact match
> "A equals B" and "A does not equal B" may have similar embeddings because the content words are the same. For negation-sensitive retrieval, use keyword filters or reranking.

> [!warning] Embeddings are not multilingual unless trained to be
> `text-embedding-3-small` handles multilingual text reasonably well. Monolingual models (like early sentence-transformers models) will cluster French text together and English text together regardless of meaning. Check your model's supported languages.

---

## Key numbers

```python
# How many tokens can you embed at once?
# OpenAI: max 8,191 tokens per request (for embedding models)
# Sentence Transformers: typically 256–512 tokens max

# How many embeddings can you batch?
# OpenAI: up to 2,048 texts per API call
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["text 1", "text 2", "text 3"]  # batch of 3
)
print(f"Got {len(response.data)} embeddings")
print(f"Total tokens: {response.usage.total_tokens}")
```

---

[[00-agenda]] | [[02-embedding-models]]
