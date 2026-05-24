# Deployment

Deployment questions test whether you've taken an LLM feature past the notebook stage. Interviewers want to see that you understand the operational realities: async, streaming, cold starts, and graceful failure.

---

## Q1: Why use `AsyncOpenAI` instead of the sync client in a FastAPI application?

??? "Show answer"
    FastAPI runs on an async event loop (uvicorn/asyncio). The synchronous `OpenAI` client uses `requests` under the hood — blocking the event loop during the HTTP call. While waiting for an LLM response (1–5 seconds), no other requests can be processed.

    `AsyncOpenAI` uses `httpx` with `await`, releasing the event loop to handle other requests concurrently during the wait.

    ```python
    import os
    from openai import AsyncOpenAI
    from fastapi import FastAPI

    app = FastAPI()
    # Instantiate once at module level — not per request
    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @app.post("/chat")
    async def chat(query: str):
        response = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}],
        )
        return {"answer": response.choices[0].message.content}
    ```

    Measured impact: with sync client and 10 concurrent requests at 2s LLM latency = 20s total processing time. With async client = ~2s total.

---

## Q2: How does SSE (Server-Sent Events) streaming work for LLM output?

??? "Show answer"
    SSE is a one-way HTTP streaming protocol. The server sends a stream of `data: {content}\n\n` lines; the client reads them as they arrive without polling.

    ```python
    import os, json
    from openai import AsyncOpenAI
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate_tokens(query: str):
        stream = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}],
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    @app.get("/stream")
    async def stream_response(query: str):
        return StreamingResponse(
            generate_tokens(query),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},  # critical for nginx
        )
    ```

    **`X-Accel-Buffering: no`** tells nginx to not buffer the response. Without it, nginx accumulates chunks and sends them in batches — defeating the purpose of streaming.

---

## Q3: What are the tradeoffs between serverless and container-based deployment for LLM APIs?

??? "Show answer"
    | | Serverless (Lambda, Modal) | Container (Fly.io, ECS, k8s) |
    |-|---------------------------|------------------------------|
    | **Cold starts** | 0.5–3s for pure Python; 5–15s with heavy deps | None (always warm) |
    | **Scaling** | Automatic, scales to zero | Manual or HPA configuration |
    | **Cost** | Pay per invocation; free at zero traffic | Pay per instance-hour; fixed baseline cost |
    | **Best for** | Bursty or low traffic; background jobs | Steady traffic; latency-critical |

    For LLM APIs that call external APIs (OpenAI, Anthropic), the LLM latency (1–5s) dominates cold start time (~500ms). Serverless is often fine.

    For self-hosted models (Ollama, vLLM), avoid serverless — model loading takes 10–30 seconds, making cold starts unacceptable.

    Fly.io `auto_stop_machines = true` with `min_machines_running = 0` gives serverless-like behaviour with a container runtime. Set `min_machines_running = 1` in production to eliminate cold starts.

---

## Q4: How do you implement a semantic cache for LLM responses?

??? "Show answer"
    Exact-match caching (SHA256 of the prompt) only hits on identical queries. A semantic cache hits on *similar* queries by embedding the query and finding near-matches in a vector store.

    ```python
    import hashlib, os
    import numpy as np
    from openai import AsyncOpenAI

    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    class SemanticCache:
        def __init__(self, threshold: float = 0.92):
            self.threshold = threshold
            self.entries: list[dict] = []  # {embedding, response}

        async def get(self, query: str) -> str | None:
            q_emb = await self._embed(query)
            for entry in self.entries:
                sim = np.dot(q_emb, entry["embedding"])  # assumes normalized vectors
                if sim >= self.threshold:
                    return entry["response"]
            return None

        async def set(self, query: str, response: str):
            q_emb = await self._embed(query)
            self.entries.append({"embedding": q_emb, "response": response})

        async def _embed(self, text: str) -> list[float]:
            result = await aclient.embeddings.create(input=text, model="text-embedding-3-small")
            emb = np.array(result.data[0].embedding)
            return (emb / np.linalg.norm(emb)).tolist()  # normalize

    cache = SemanticCache(threshold=0.92)

    async def cached_answer(query: str) -> str:
        if cached := await cache.get(query):
            return cached
        response = await aclient.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": query}]
        )
        answer = response.choices[0].message.content
        await cache.set(query, answer)
        return answer
    ```

    Threshold 0.90–0.95 is the typical range. Below 0.90, unrelated queries match. Above 0.95, you rarely hit the cache.

---

## Q5: How do you implement graceful shutdown in a FastAPI LLM service?

??? "Show answer"
    Graceful shutdown ensures in-flight requests complete before the process exits. This matters especially for streaming endpoints and background LLM jobs.

    ```python
    import asyncio, signal, os
    from contextlib import asynccontextmanager
    from openai import AsyncOpenAI
    from fastapi import FastAPI

    aclient: AsyncOpenAI = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global aclient
        # Startup
        aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        yield
        # Shutdown — aclient closes its httpx session automatically

    app = FastAPI(lifespan=lifespan)
    ```

    For long-running background tasks, track them and wait on shutdown:

    ```python
    background_tasks: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Wait for all background tasks to complete (with timeout)
        if background_tasks:
            await asyncio.wait(background_tasks, timeout=30.0)

    @app.post("/process")
    async def start_job(query: str):
        task = asyncio.create_task(long_running_job(query))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return {"job_id": id(task)}
    ```

---

*Previous: [Tracing and Monitoring](01-tracing-and-monitoring.md) | Next: [Cost Optimization](03-cost-optimization.md)*
