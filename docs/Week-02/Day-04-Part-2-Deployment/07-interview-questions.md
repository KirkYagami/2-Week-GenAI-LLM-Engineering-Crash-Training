# Interview Questions — Deployment

---

## Q1. When should you stream an LLM response, and how does SSE work?

??? "Show answer"
    Stream when perceived latency matters more than total latency — a response that starts appearing in 400ms feels faster than one that arrives complete after 4 seconds, even if the total time is identical. Stream for: chat interfaces, long-form generation, real-time summarization. Do not stream for: classification, structured extraction, batch pipelines where the full result is needed before any action is taken.

    Server-Sent Events (SSE) is the wire format: the server sends `Content-Type: text/event-stream` and pushes newline-delimited messages:

    ```
    data: {"token": "Hello"}\n\n
    data: {"token": " world"}\n\n
    data: {"token": "", "done": true}\n\n
    ```

    In FastAPI, `StreamingResponse` wraps an async generator that `yield`s these strings. The client reads lines, strips the `data: ` prefix, and parses the JSON. The connection stays open until the generator is exhausted.

---

## Q2. Why does FastAPI require `AsyncOpenAI` instead of the synchronous `OpenAI` client?

??? "Show answer"
    FastAPI runs on an async event loop (uvicorn + asyncio). A single event loop handles all concurrent requests by switching between coroutines at `await` points. If you call the synchronous `OpenAI` client inside an `async def` endpoint, it blocks the thread — no other requests can be served while waiting for the LLM response. Under load, this serializes all requests through one at a time.

    `AsyncOpenAI` issues HTTP requests through `httpx.AsyncClient`, which yields control at every `await`, letting the event loop serve other requests concurrently.

    The fix is one import change:

    ```python
    # Wrong — blocks the event loop
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(...)  # Blocks!

    # Correct — non-blocking
    from openai import AsyncOpenAI
    aclient = AsyncOpenAI()
    response = await aclient.chat.completions.create(...)  # Yields control
    ```

    Create the async client once at module level or in the `lifespan` handler — creating a new client per request wastes connection pool setup.

---

## Q3. What is the `X-Accel-Buffering: no` header and why is it required for streaming?

??? "Show answer"
    Nginx and many reverse proxies buffer upstream responses by default: they accumulate the full response before forwarding it to the client. This defeats streaming — the client receives the entire response at once despite the server streaming it token by token.

    `X-Accel-Buffering: no` tells nginx to pass each chunk through immediately. The equivalent nginx configuration directive is `proxy_buffering off`. Without this header, streaming endpoints appear to work in local testing (no proxy) but break in production (behind nginx or a load balancer).

    Always include both headers for SSE endpoints:

    ```python
    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",         # Prevent proxy caching
            "X-Accel-Buffering": "no",            # Disable nginx buffering
        },
    )
    ```

---

## Q4. What is the difference between exact-match caching and semantic caching? When should you use each?

??? "Show answer"
    **Exact-match caching** hashes the request parameters (messages, model, temperature) with SHA-256 and stores the response. A cache hit requires the input to be byte-for-byte identical. Advantages: zero false positives, constant-time lookup, no embedding cost. Disadvantage: "What is the capital of France?" and "Tell me France's capital" are different keys, so both call the LLM.

    **Semantic caching** embeds each query and checks cosine similarity against stored queries. If similarity exceeds a threshold (typically 0.90–0.95), the cached response is returned. Advantages: catches paraphrases and minor rewording. Disadvantages: requires an embedding call for every lookup (adds latency and cost), risks false positives at lower thresholds (returning wrong answers for unrelated similar-looking queries).

    Use exact-match for: fixed-template prompts (FAQ bots, document classification), high-throughput pipelines where identical inputs repeat often.

    Use semantic caching for: conversational interfaces where users rephrase common questions, knowledge bases where the query space is bounded.

    Never cache: personalized responses (include `user_id` in the cache key or don't cache), real-time data lookups (prices, news), or any response with non-zero temperature.

---

## Q5. How do LLM services experience cold starts in serverless environments, and what are the mitigation strategies?

??? "Show answer"
    A cold start is the delay between a new serverless instance spinning up and serving its first request. For LLM services, cold start sources vary by architecture:

    | Source | Typical Delay |
    |--------|--------------|
    | OpenAI API client init | < 100ms (negligible) |
    | Heavy library imports (torch, transformers) | 5–10s |
    | Loading a local model (7B, llama.cpp) | 30–90s |

    Mitigation strategies, in order of effectiveness:

    1. **Minimum instances > 0** — keep one instance always warm (`min_containers=1` in Modal). Costs money continuously but eliminates cold starts entirely.
    2. **Keep-warm pings** — a scheduled request every 5 minutes to prevent the instance from scaling to zero. Works on most platforms.
    3. **Lazy initialization** — initialize the model on the first real request, not at import time. Doesn't reduce cold start duration but moves it to a time the user is already waiting.
    4. **Lightweight dependencies** — using the OpenAI API rather than a local model avoids the model-loading cold start entirely. An `openai + fastapi` app cold-starts in ~1–2 seconds.

    For GPU models on Modal: the T4 warm cost (~$0.30/hr) is almost always worth it compared to a 60-second cold start on every new deployment or inactivity period.

---

## Q6. What is `BackgroundTasks` in FastAPI and when should you use it?

??? "Show answer"
    `BackgroundTasks` lets you schedule work to run after the HTTP response has been sent to the client. The response is delivered first, then the background function runs in the same process.

    ```python
    @app.post("/chat")
    async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
        response = await aclient.chat.completions.create(...)
        reply = response.choices[0].message.content

        # Runs after client receives the reply
        background_tasks.add_task(log_to_database, user_id=request.user_id, reply=reply)

        return {"response": reply}
    ```

    Use it for: audit logging, usage tracking, cache warming, sending webhooks, updating metrics — anything that the client doesn't need to wait for.

    Do not use it for: long-running tasks (> a few seconds), work that must complete reliably (background tasks are lost if the process crashes), or CPU-intensive work that would starve the event loop. For those cases, use a task queue (Celery, ARQ, or Modal's background function calling).

---

## Q7. How would you implement rate limiting on a FastAPI LLM endpoint, and what should the client receive when it hits the limit?

??? "Show answer"
    A sliding-window rate limiter tracks request timestamps per key (IP address or API key) and rejects requests that would exceed the configured rate:

    ```python
    import time
    from collections import defaultdict, deque
    from fastapi import HTTPException, Request

    class RateLimiter:
        def __init__(self, max_requests: int, window_seconds: int):
            self.max_requests = max_requests
            self.window = window_seconds
            self._windows: dict[str, deque] = defaultdict(deque)

        def check(self, key: str) -> None:
            now = time.time()
            timestamps = self._windows[key]
            while timestamps and now - timestamps[0] > self.window:
                timestamps.popleft()
            if len(timestamps) >= self.max_requests:
                retry_after = int(self.window - (now - timestamps[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(retry_after)},
                )
            timestamps.append(now)

    limiter = RateLimiter(max_requests=10, window_seconds=60)
    ```

    The client should receive HTTP 429 with a `Retry-After` header indicating how many seconds to wait before retrying. Clients that respect this header implement exponential backoff automatically.

    For production, replace the in-memory `deque` with Redis (using `ZADD`/`ZRANGEBYSCORE` on a sorted set) so the limiter works correctly across multiple server instances.

---

[[06-practice-exercises]]
