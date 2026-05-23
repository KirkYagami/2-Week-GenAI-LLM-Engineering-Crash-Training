# Advanced Features — RAG Q&A Chatbot

## Cross-encoder reranking

The initial retrieval returns the top-N chunks by cosine similarity. A cross-encoder reranker re-scores the candidates by reading each (query, chunk) pair together — more accurate but slower.

```python
# reranker.py
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: list[str], source_ids: list[str], top_k: int = 3) -> tuple[list[str], list[str]]:
    """Rerank retrieved chunks and return top_k."""
    if not chunks:
        return [], []
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks, source_ids), reverse=True)
    top = ranked[:top_k]
    return [c for _, c, _ in top], [s for _, _, s in top]
```

Integrate into the chat endpoint:

```python
# In app.py, after collection.query():
from reranker import rerank

results = collection.query(query_texts=[request.question], n_results=10)  # Fetch more candidates
chunks = results["documents"][0]
source_ids = results["ids"][0]

# Rerank to get top 3
chunks, source_ids = rerank(request.question, chunks, source_ids, top_k=3)
```

> [!tip] Fetch 10, rerank to 3
> Reranking over 10 candidates typically improves precision without significantly increasing latency. The cross-encoder model runs locally — no API call — so it adds 50–150ms depending on hardware.

---

## Semantic caching

Exact-match caching misses on rephrased queries. Semantic caching checks cosine similarity between the current query and previously answered queries.

```python
# semantic_cache.py
import time
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SemanticCache:
    def __init__(self, threshold: float = 0.92, ttl: int = 3600):
        self.threshold = threshold
        self.ttl = ttl
        self._entries: list[dict] = []

    def _embed(self, text: str) -> np.ndarray:
        resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
        emb = np.array(resp.data[0].embedding)
        return emb / np.linalg.norm(emb)

    def get(self, query: str) -> dict | None:
        q_emb = self._embed(query)
        now = time.time()
        best_score, best = 0.0, None
        for entry in self._entries:
            if now - entry["ts"] > self.ttl:
                continue
            score = float(q_emb @ entry["emb"])
            if score > best_score:
                best_score, best = score, entry
        if best and best_score >= self.threshold:
            return best["response"]
        return None

    def set(self, query: str, response: dict) -> None:
        q_emb = self._embed(query)
        self._entries.append({"emb": q_emb, "response": response, "ts": time.time()})

sem_cache = SemanticCache(threshold=0.92)
```

---

## Metadata filtering

When documents have dates, categories, or authors, filter retrievals to only surface relevant metadata:

```python
# When ingesting, store rich metadata:
metadatas.append({
    "source": path.name,
    "category": "faq",         # or "guide", "policy", etc.
    "last_updated": "2024-01",
})

# When querying, filter by category:
results = collection.query(
    query_texts=[request.question],
    n_results=5,
    where={"category": {"$eq": "faq"}},  # ChromaDB filter syntax
)
```

---

## Context-aware conversation memory

For multi-turn conversations, include recent exchanges as context:

```python
class ConversationRequest(BaseModel):
    question: str
    history: list[dict] = Field(default_factory=list, description="List of {role, content} dicts")
    session_id: str = "default"

def _build_multi_turn_prompt(question: str, history: list[dict], chunks: list[str], source_ids: list[str]) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"[Source {i+1} — {sid}]\n{chunk}"
        for i, (sid, chunk) in enumerate(zip(source_ids, chunks))
    )
    messages = [
        {"role": "system", "content": f"Answer using only the context provided. Cite sources.\n\nContext:\n{context}"},
    ]
    # Include last 4 turns of history
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": question})
    return messages
```

---

[[02-implementation]] | [[04-evaluation]]
