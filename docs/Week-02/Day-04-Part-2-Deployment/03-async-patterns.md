# Async Patterns

FastAPI is async-first, but LLM applications often fall into sync traps: blocking the event loop with sequential API calls, spinning up a new HTTP client per request, or forgetting to await coroutines. This note covers the patterns that keep a FastAPI LLM service responsive under load.

## Learning objectives

- Use `AsyncOpenAI` to avoid blocking the event loop
- Implement background tasks for post-request work
- Handle concurrent requests with proper resource management
- Use connection pooling for external APIs
- Write async context managers for lifecycle management

---

## AsyncOpenAI: don't block the event loop

```python
import os
import asyncio
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

app = FastAPI()
# Share one async client across all requests (thread-safe, connection-pooled)
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.0

@app.post("/chat")
async def chat(request: ChatRequest):
    # ✓ Correct: awaited async call — doesn't block the event loop
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": request.message}],
        temperature=request.temperature,
    )
    return {"response": response.choices[0].message.content}

# ✗ Wrong: Using synchronous client in an async endpoint
# from openai import OpenAI
# sync_client = OpenAI(...)
# response = sync_client.chat.completions.create(...)  # Blocks the event loop!
```

---

## Concurrent multi-call patterns

```python
import os
import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pattern 1: Run N calls in parallel (all independent)
async def parallel_analysis(text: str) -> dict:
    """Analyze text for sentiment, topics, and summary simultaneously."""

    async def get_sentiment():
        r = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Sentiment (one word): {text}"}],
            max_tokens=5, temperature=0.0,
        )
        return r.choices[0].message.content.strip()

    async def get_topics():
        r = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"List 3 topics (comma-separated): {text}"}],
            max_tokens=30, temperature=0.0,
        )
        return [t.strip() for t in r.choices[0].message.content.split(",")]

    async def get_summary():
        r = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarize in one sentence: {text}"}],
            max_tokens=50, temperature=0.0,
        )
        return r.choices[0].message.content.strip()

    sentiment, topics, summary = await asyncio.gather(
        get_sentiment(), get_topics(), get_summary()
    )
    return {"sentiment": sentiment, "topics": topics, "summary": summary}

# Pattern 2: Parallel with error handling (one failure doesn't kill others)
async def parallel_with_errors(texts: list[str]) -> list[dict]:
    """Process multiple texts in parallel; return errors for failed ones."""

    async def process_one(text: str) -> dict:
        try:
            r = await aclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Summarize: {text}"}],
                max_tokens=50,
            )
            return {"text": text[:30], "summary": r.choices[0].message.content, "error": None}
        except Exception as e:
            return {"text": text[:30], "summary": None, "error": str(e)}

    return await asyncio.gather(*[process_one(t) for t in texts])

# Pattern 3: Semaphore to limit concurrent calls
async def rate_limited_parallel(texts: list[str], max_concurrent: int = 5) -> list[str]:
    """Process texts in parallel but limit to max_concurrent at a time."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(text: str) -> str:
        async with semaphore:
            r = await aclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Summarize in 5 words: {text}"}],
                max_tokens=20,
            )
            return r.choices[0].message.content

    return await asyncio.gather(*[process_with_limit(t) for t in texts])

# Test
async def demo():
    result = await parallel_analysis("Machine learning is transforming how businesses operate. Companies are investing heavily in AI infrastructure.")
    print(result)

asyncio.run(demo())
```

---

## Background tasks for post-request work

```python
import os
import time
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from openai import AsyncOpenAI
import json

app = FastAPI()
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

AUDIT_LOG = []  # In production: write to a database or message queue

def log_to_audit(request_data: dict, response_data: dict, latency_ms: float) -> None:
    """Non-blocking audit logging — runs after the response is sent."""
    entry = {
        "timestamp": time.time(),
        "latency_ms": round(latency_ms, 1),
        **request_data,
        **response_data,
    }
    AUDIT_LOG.append(entry)
    print(f"[AUDIT] Logged: {json.dumps(entry)[:100]}")

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    start = time.perf_counter()

    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": request.message}],
    )
    reply = response.choices[0].message.content
    latency_ms = (time.perf_counter() - start) * 1000

    # Schedule background task — runs after response is returned to client
    background_tasks.add_task(
        log_to_audit,
        request_data={"user_id": request.user_id, "message": request.message[:100]},
        response_data={"tokens": response.usage.total_tokens, "reply": reply[:100]},
        latency_ms=latency_ms,
    )

    return {"response": reply}
```

---

## App lifecycle: startup and shutdown

```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from openai import AsyncOpenAI

aclient: AsyncOpenAI = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global aclient
    # Startup: initialize shared resources
    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("OpenAI client initialized")
    yield
    # Shutdown: cleanup
    await aclient.close()
    print("OpenAI client closed")

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(request: dict):
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": request["message"]}],
    )
    return {"response": response.choices[0].message.content}
```

> [!tip] Create the async client once and share it across requests
> Creating a new `AsyncOpenAI()` per request creates a new HTTP connection pool every time — expensive and unnecessary. Create it once at startup (in `lifespan` or as a module-level variable) and share it across all requests. The client is thread-safe and coroutine-safe.

---

[[02-streaming-responses]] | [[04-serverless]]
