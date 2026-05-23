# Practice Exercises — Deployment

---

## Exercise 1 — FastAPI service with caching (Warm-up)

Build a complete FastAPI LLM service with exact-match caching, request validation, and a cache stats endpoint.

```python
# app.py
import os
import time
import hashlib
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from openai import AsyncOpenAI, RateLimitError, APIError

app = FastAPI(title="LLM Service", version="1.0.0")
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory cache
_cache: dict[str, dict] = {}
_cache_stats = {"hits": 0, "misses": 0, "total_tokens_saved": 0}
CACHE_TTL = 600  # 10 minutes

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=3000)
    system: Optional[str] = Field(None, max_length=1000)
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    cache: bool = Field(True, description="Whether to use response cache")

class ChatResponse(BaseModel):
    response: str
    tokens: int
    latency_ms: float
    cached: bool
    cost_usd: float

def make_cache_key(req: ChatRequest) -> str:
    payload = json.dumps({
        "message": req.message,
        "system": req.system or "",
        "temperature": req.temperature,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.temperature == 0.0 and request.cache:
        key = make_cache_key(request)
        entry = _cache.get(key)
        if entry and time.time() - entry["ts"] < CACHE_TTL:
            _cache_stats["hits"] += 1
            _cache_stats["total_tokens_saved"] += entry["tokens"]
            return ChatResponse(**entry["data"], cached=True)
        _cache_stats["misses"] += 1

    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.message})

    start = time.perf_counter()
    try:
        response = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=request.temperature,
            max_tokens=500,
        )
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    latency = (time.perf_counter() - start) * 1000
    usage = response.usage
    cost = (usage.prompt_tokens / 1e6 * 0.15) + (usage.completion_tokens / 1e6 * 0.60)

    data = {
        "response": response.choices[0].message.content,
        "tokens": usage.total_tokens,
        "latency_ms": round(latency, 1),
        "cost_usd": round(cost, 8),
    }

    if request.temperature == 0.0 and request.cache:
        _cache[key] = {"data": data, "ts": time.time(), "tokens": usage.total_tokens}

    return ChatResponse(**data, cached=False)

@app.get("/cache/stats")
async def cache_stats():
    total = _cache_stats["hits"] + _cache_stats["misses"]
    return {
        "entries": len(_cache),
        "hits": _cache_stats["hits"],
        "misses": _cache_stats["misses"],
        "hit_rate": f"{_cache_stats['hits']/total*100:.1f}%" if total > 0 else "0%",
        "tokens_saved": _cache_stats["total_tokens_saved"],
        "estimated_cost_saved_usd": round(_cache_stats["total_tokens_saved"] / 1e6 * 0.15, 6),
    }

@app.delete("/cache")
async def clear_cache():
    _cache.clear()
    return {"message": "Cache cleared"}

@app.get("/health")
async def health():
    return {"status": "ok", "cache_entries": len(_cache)}

# Test with:
# uvicorn app:app --reload
# curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
#   -d '{"message": "What is Python?"}'
# curl http://localhost:8000/cache/stats
```

---

## Exercise 2 — Streaming endpoint with client (Main)

Build a streaming FastAPI endpoint and a Python client that consumes it, measuring time-to-first-token.

```python
# streaming_service.py
import os
import json
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI()
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class StreamRequest(BaseModel):
    message: str
    max_tokens: int = 300

async def token_stream(message: str, max_tokens: int):
    yield f"data: {json.dumps({'event': 'start', 'ts': time.time()})}\n\n"
    token_count = 0
    try:
        async for chunk in await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
            stream=True,
            max_tokens=max_tokens,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                token_count += 1
                yield f"data: {json.dumps({'event': 'token', 'token': delta, 'n': token_count})}\n\n"
        yield f"data: {json.dumps({'event': 'done', 'total_tokens': token_count})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

@app.post("/stream")
async def stream(request: StreamRequest):
    return StreamingResponse(
        token_stream(request.message, request.max_tokens),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ------ Client (run separately) ------
# streaming_client.py
import httpx
import json
import time

def consume_stream(url: str, message: str) -> dict:
    start = time.perf_counter()
    first_token_time = None
    tokens = []

    with httpx.Client(timeout=30.0) as client:
        with client.stream("POST", url, json={"message": message, "max_tokens": 200}) as response:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data["event"] == "token":
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens.append(data["token"])
                    print(data["token"], end="", flush=True)
                elif data["event"] == "done":
                    break

    total_ms = (time.perf_counter() - start) * 1000
    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0

    print(f"\n\n--- Stream Metrics ---")
    print(f"Time to first token: {ttft_ms:.0f}ms")
    print(f"Total time:          {total_ms:.0f}ms")
    print(f"Tokens received:     {len(tokens)}")
    print(f"Tokens/second:       {len(tokens) / (total_ms/1000):.1f}")

    return {"ttft_ms": ttft_ms, "total_ms": total_ms, "tokens": len(tokens)}

# To test: start the server, then:
# result = consume_stream("http://localhost:8000/stream", "Explain transformers in detail.")
```

---

[[05-caching-strategies]] | [[07-interview-questions]]
