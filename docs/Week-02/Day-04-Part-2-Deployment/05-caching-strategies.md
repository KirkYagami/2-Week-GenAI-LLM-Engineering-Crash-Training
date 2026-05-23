# Caching Strategies

LLM calls are expensive and slow. For identical or near-identical inputs, you can serve the cached result in milliseconds at near-zero cost. Two types of caching matter for LLM applications: exact-match caching for identical prompts, and semantic caching for similar queries.

## Learning objectives

- Implement exact-match caching with TTL
- Build a semantic cache using embeddings
- Add a cache layer to a FastAPI endpoint
- Measure cache hit rate and cost savings
- Know when caching is appropriate vs harmful

---

## Exact-match cache

```python
import os
import hashlib
import json
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ExactMatchCache:
    def __init__(self, ttl_seconds: int = 3600):
        self._store: dict[str, dict] = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _key(self, messages: list[dict], model: str, temperature: float) -> str:
        """Deterministic cache key from request parameters."""
        payload = json.dumps({"messages": messages, "model": model, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, messages: list[dict], model: str, temperature: float) -> str | None:
        key = self._key(messages, model, temperature)
        entry = self._store.get(key)
        if entry and time.time() - entry["timestamp"] < self.ttl:
            self.hits += 1
            return entry["response"]
        self.misses += 1
        return None

    def set(self, messages: list[dict], model: str, temperature: float, response: str) -> None:
        key = self._key(messages, model, temperature)
        self._store[key] = {"response": response, "timestamp": time.time()}

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

cache = ExactMatchCache(ttl_seconds=300)

def cached_completion(messages: list[dict], model: str = "gpt-4o-mini", temperature: float = 0.0) -> str:
    """Completion with exact-match caching. Cache only for deterministic calls (temperature=0)."""
    if temperature > 0:
        # Don't cache non-deterministic outputs
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return response.choices[0].message.content

    cached = cache.get(messages, model, temperature)
    if cached:
        return cached

    start = time.perf_counter()
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    latency = (time.perf_counter() - start) * 1000
    result = response.choices[0].message.content

    cache.set(messages, model, temperature, result)
    print(f"  [MISS] API call: {latency:.0f}ms | tokens: {response.usage.total_tokens}")
    return result

# Test caching
questions = [
    "What is the capital of France?",
    "What is the capital of France?",  # Identical — should hit cache
    "What is the capital of Germany?",
    "What is the capital of France?",  # Hit again
]

for q in questions:
    msg = [{"role": "user", "content": q}]
    result = cached_completion(msg)
    status = "HIT " if cache.hits > 0 and q == questions[0] else "MISS" if "France" in q else "MISS"
    print(f"  Q: {q[:40]} → {result[:30]}...")

print(f"\nCache hit rate: {cache.hit_rate:.0%} ({cache.hits} hits / {cache.misses} misses)")
```

---

## Semantic cache

Exact-match misses on "What is the capital of France?" vs "Tell me the capital city of France." Semantic caching handles this by checking if similar queries have cached answers.

```python
import os
import time
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92, ttl_seconds: int = 3600):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self._entries: list[dict] = []  # [{embedding, query, response, timestamp}]
        self.hits = 0
        self.misses = 0
        self.embedding_calls = 0

    def _embed(self, text: str) -> np.ndarray:
        self.embedding_calls += 1
        resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
        emb = np.array(resp.data[0].embedding)
        return emb / np.linalg.norm(emb)

    def get(self, query: str) -> str | None:
        q_emb = self._embed(query)
        now = time.time()

        # Find the most similar cached entry above threshold
        best_score = 0.0
        best_entry = None
        for entry in self._entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            score = float(q_emb @ entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            self.hits += 1
            print(f"  [SEMANTIC HIT] similarity={best_score:.3f} | matched: '{best_entry['query'][:50]}'")
            return best_entry["response"]

        self.misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        q_emb = self._embed(query)
        self._entries.append({
            "query": query, "response": response,
            "embedding": q_emb, "timestamp": time.time()
        })

sem_cache = SemanticCache(similarity_threshold=0.90)

def semantically_cached_completion(query: str) -> str:
    cached = sem_cache.get(query)
    if cached:
        return cached

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
        temperature=0.0,
    )
    result = response.choices[0].message.content
    sem_cache.set(query, result)
    return result

# Test semantic similarity matching
queries = [
    "What is the capital of France?",
    "Tell me the capital city of France.",          # Semantically similar → HIT
    "Which city serves as France's capital?",       # Semantically similar → HIT
    "What is the population of Paris?",              # Different topic → MISS
    "What is the capital of Germany?",              # Different country → MISS
]

for q in queries:
    result = semantically_cached_completion(q)
    print(f"Q: {q[:55]:<55} A: {result[:30]}...")

print(f"\nHit rate: {sem_cache.hit_rate:.0%} | Embedding calls: {sem_cache.embedding_calls}")
```

---

## Cache integration in FastAPI

```python
import os
import time
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI
import hashlib, json

app = FastAPI()
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_cache: dict[str, dict] = {}
CACHE_TTL = 300  # 5 minutes

def cache_key(message: str) -> str:
    return hashlib.md5(message.encode()).hexdigest()

class ChatRequest(BaseModel):
    message: str
    use_cache: bool = True

@app.post("/chat")
async def chat(request: ChatRequest):
    key = cache_key(request.message)

    if request.use_cache:
        entry = _cache.get(key)
        if entry and time.time() - entry["ts"] < CACHE_TTL:
            return {**entry["data"], "cached": True, "cache_age_s": int(time.time() - entry["ts"])}

    start = time.perf_counter()
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": request.message}],
        temperature=0.0,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    data = {
        "response": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "latency_ms": round(latency_ms, 0),
    }
    _cache[key] = {"data": data, "ts": time.time()}
    return {**data, "cached": False}
```

---

## When to cache and when not to

| Cache type | Good for | Avoid for |
|-----------|----------|-----------|
| Exact match | FAQ answers, fixed-template outputs | Personalized responses |
| Semantic | Variations of common questions | Creative tasks (temperature > 0) |
| Both | Static knowledge, documentation Q&A | Real-time data (prices, news) |
| Neither | User-specific, time-sensitive, stateful | Anything requiring freshness |

> [!warning] Never cache responses that include personal data or vary by user context
> A cached response to "What's my account balance?" that's served to a different user is a data breach. Scope your cache key to include user_id for any personalized endpoint.

---

[[04-serverless]] | [[06-practice-exercises]]
