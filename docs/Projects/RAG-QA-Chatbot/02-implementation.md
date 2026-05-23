# Implementation — RAG Q&A Chatbot

## Core application

```python
# app.py
import os
import time
import hashlib
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from openai import AsyncOpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

# ---- Globals initialized in lifespan ----
aclient: AsyncOpenAI = None
collection = None
_cache: dict[str, dict] = {}
_stats = {"requests": 0, "cache_hits": 0, "total_tokens": 0}
CACHE_TTL = 600  # 10 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    global aclient, collection
    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path=os.getenv("CHROMA_PERSIST_PATH", "./chroma_db"))
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    collection = chroma.get_or_create_collection(
        os.getenv("COLLECTION_NAME", "documents"),
        embedding_function=embedding_fn,
    )
    print(f"[startup] collection has {collection.count()} documents")
    yield
    await aclient.close()

app = FastAPI(title="RAG Q&A Chatbot", version="1.0.0", lifespan=lifespan)

# ---- Models ----

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    n_results: int = Field(5, ge=1, le=10)
    use_cache: bool = True
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    tokens: int
    latency_ms: float
    cached: bool

# ---- Cache helpers ----

def _cache_key(question: str, n_results: int) -> str:
    payload = json.dumps({"q": question, "n": n_results}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

def _build_prompt(question: str, chunks: list[str], source_ids: list[str]) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"[Source {i+1} — {sid}]\n{chunk}"
        for i, (sid, chunk) in enumerate(zip(source_ids, chunks))
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question using only the provided context. "
                "Cite source numbers in your answer (e.g., [Source 1]). "
                "If the context doesn't contain enough information, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

# ---- Endpoints ----

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    _stats["requests"] += 1
    key = _cache_key(request.question, request.n_results)

    if request.use_cache:
        entry = _cache.get(key)
        if entry and time.time() - entry["ts"] < CACHE_TTL:
            _stats["cache_hits"] += 1
            return ChatResponse(**entry["data"], cached=True)

    results = collection.query(query_texts=[request.question], n_results=request.n_results)
    chunks = results["documents"][0] if results["documents"] else []
    source_ids = results["ids"][0] if results["ids"] else []

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found for this question.")

    messages = _build_prompt(request.question, chunks, source_ids)

    start = time.perf_counter()
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=600,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    _stats["total_tokens"] += response.usage.total_tokens
    data = {
        "answer": response.choices[0].message.content,
        "sources": source_ids,
        "tokens": response.usage.total_tokens,
        "latency_ms": round(latency_ms, 1),
    }
    _cache[key] = {"data": data, "ts": time.time()}
    return ChatResponse(**data, cached=False)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    results = collection.query(query_texts=[request.question], n_results=request.n_results)
    chunks = results["documents"][0] if results["documents"] else []
    source_ids = results["ids"][0] if results["ids"] else []

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    messages = _build_prompt(request.question, chunks, source_ids)

    async def token_stream():
        yield f"data: {json.dumps({'event': 'sources', 'ids': source_ids})}\n\n"
        async for chunk in await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
            max_tokens=600,
            temperature=0.0,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'event': 'token', 'token': delta})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/health")
async def health():
    return {"status": "ok", "doc_count": collection.count()}

@app.get("/stats")
async def stats():
    total = _stats["requests"]
    hit_rate = _stats["cache_hits"] / total if total > 0 else 0.0
    return {
        **_stats,
        "cache_entries": len(_cache),
        "hit_rate": f"{hit_rate:.1%}",
    }
```

## Running locally

```bash
uvicorn app:app --reload
# Server starts at http://localhost:8000
```

Test with curl:

```bash
# Non-streaming
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is your refund policy?"}'

# Streaming
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset my password?"}'

# Cache stats
curl http://localhost:8000/stats
```

## Python client

```python
# client.py
import httpx
import json

def ask(question: str, base_url: str = "http://localhost:8000") -> dict:
    resp = httpx.post(f"{base_url}/chat", json={"question": question}, timeout=30.0)
    resp.raise_for_status()
    return resp.json()

def ask_streaming(question: str, base_url: str = "http://localhost:8000") -> str:
    full_text = ""
    with httpx.Client(timeout=30.0) as client:
        with client.stream("POST", f"{base_url}/chat/stream", json={"question": question}) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data["event"] == "token":
                    print(data["token"], end="", flush=True)
                    full_text += data["token"]
                elif data["event"] == "done":
                    break
    return full_text

if __name__ == "__main__":
    result = ask("What products do you offer?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Latency: {result['latency_ms']}ms | Cached: {result['cached']}")
```

> [!warning] ChromaDB embedding calls on every query add ~100ms
> The `collection.query()` call embeds your question before searching. To reduce latency, pre-embed the question with `aclient.embeddings.create()` and use `collection.query(query_embeddings=[...])` instead — this lets you time the embedding separately and reuse it if needed.

---

[[01-setup]] | [[03-advanced-features]]
