# OpenAI API

The OpenAI API is the most common API in LLM engineering interviews. Interviewers expect you to know it at the implementation level, not just "I called `client.chat.completions.create()`."

---

## Q1: Walk me through the key parameters of `chat.completions.create()`.

??? "Show answer"
    ```python
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",              # model version — pin this in production
        messages=[...],              # list of {role, content} dicts
        temperature=0.0,             # 0=deterministic, 1=creative
        max_tokens=1024,             # cap output length; avoids runaway generation
        top_p=1.0,                   # nucleus sampling threshold (use with temperature, not both)
        frequency_penalty=0.0,       # penalise repeated tokens (-2 to 2)
        presence_penalty=0.0,        # penalise tokens already used (-2 to 2)
        stop=None,                   # stop sequences — generation halts on match
        seed=42,                     # reproducibility (best-effort, not guaranteed)
        response_format={"type": "json_object"},  # enforce valid JSON output
        stream=False,                # True = receive tokens as they generate
    )
    ```

    In production: always set `max_tokens` to prevent unbounded output. Pin the model name — `gpt-4o` silently upgrades; `gpt-4o-2024-08-06` is immutable.

---

## Q2: Why use `AsyncOpenAI` instead of the sync client in FastAPI?

??? "Show answer"
    FastAPI runs on an async event loop. Calling the synchronous `OpenAI` client blocks the event loop during the network request — no other requests can be handled while waiting for the LLM response.

    `AsyncOpenAI` uses `httpx` under the hood and `await`s the network call, releasing the event loop to handle other requests concurrently.

    ```python
    import os
    from openai import AsyncOpenAI
    from fastapi import FastAPI

    app = FastAPI()
    # Create one shared client at module level — don't instantiate per request
    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @app.post("/chat")
    async def chat(query: str):
        response = await aclient.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}],
        )
        return {"answer": response.choices[0].message.content}
    ```

    With 10 concurrent requests and a 2-second LLM latency: sync client = 20 seconds total, async client = ~2 seconds total.

---

## Q3: How does streaming work and how do you implement it as SSE?

??? "Show answer"
    With `stream=True`, the API returns an iterator of `ChatCompletionChunk` objects. Each chunk contains a `delta.content` fragment (may be `None` for the first and last chunks).

    For a FastAPI SSE endpoint:

    ```python
    import os, json
    from openai import AsyncOpenAI
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def token_stream(query: str):
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
    async def stream_endpoint(query: str):
        return StreamingResponse(
            token_stream(query),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )
    ```

    `X-Accel-Buffering: no` tells nginx not to buffer the response — without it, tokens arrive in batches rather than one at a time.

---

## Q4: How do you handle rate limit errors in production?

??? "Show answer"
    OpenAI rate limits return `openai.RateLimitError` (HTTP 429). The standard response is exponential backoff with jitter.

    ```python
    import os, time, random
    from openai import OpenAI, RateLimitError

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def call_with_backoff(messages: list, max_retries: int = 5):
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
    ```

    In production: use the `tenacity` library for cleaner retry logic. Also implement a token bucket at your application layer so you never hit rate limits in the first place. Return `HTTP 429` with a `Retry-After` header to callers when your own limit is reached.

---

## Q5: What is the function calling (tools) cycle and how do you execute it correctly?

??? "Show answer"
    Four steps:

    1. Send the request with `tools` defined
    2. Check if `response.choices[0].message.tool_calls` is non-empty
    3. Execute each tool call locally and collect results
    4. Append the assistant message + tool results to the message history and call again

    ```python
    import json, os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_weather(city: str) -> str:
        return f"72°F, sunny in {city}"  # replace with real API call

    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }}]

    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)

    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        result = get_weather(**args)

        messages.append(response.choices[0].message)  # assistant turn with tool_calls
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

        final = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)
        print(final.choices[0].message.content)
    ```

    Common mistake: forgetting to append the assistant message before the tool result. Without it, the message history is invalid and the model errors.

---

*Previous: [Vector Databases](../02-RAG-and-Vector-Search/03-vector-databases.md) | Next: [Anthropic API](02-anthropic-api.md)*
