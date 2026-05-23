# Streaming Responses

Waiting 4 seconds for a complete LLM response feels slow. Streaming the same response token-by-token, starting in 400ms, feels fast — even though the total time is the same. Streaming changes perceived latency from "time to last token" to "time to first token," which is typically 10x shorter.

## Learning objectives

- Implement Server-Sent Events (SSE) streaming in FastAPI
- Handle streaming from both OpenAI and Anthropic APIs
- Stream LangChain chain outputs
- Test streaming endpoints with httpx
- Handle stream errors and cleanup gracefully

---

## SSE streaming with FastAPI

```python
# streaming_app.py
import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import asyncio

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."

async def stream_openai_response(message: str, system: str):
    """Generator that yields SSE-formatted chunks from OpenAI."""
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ],
            stream=True,
            max_tokens=500,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                # SSE format: "data: {json}\n\n"
                data = json.dumps({"token": delta.content, "done": False})
                yield f"data: {data}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    except Exception as e:
        error_data = json.dumps({"error": str(e), "done": True})
        yield f"data: {error_data}\n\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        stream_openai_response(request.message, request.system_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )

# Run: uvicorn streaming_app:app --reload
```

---

## Streaming with metadata events

```python
import os
import json
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def stream_with_metadata(message: str):
    """Stream tokens with metadata events at start and end."""
    start_time = time.perf_counter()

    # Send start event with request metadata
    yield f"data: {json.dumps({'event': 'start', 'model': 'gpt-4o-mini'})}\n\n"

    token_count = 0
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
            stream=True,
            max_tokens=300,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                token_count += 1
                yield f"data: {json.dumps({'event': 'token', 'token': token})}\n\n"

        # Send completion event with metrics
        latency_ms = (time.perf_counter() - start_time) * 1000
        yield f"data: {json.dumps({'event': 'done', 'tokens': token_count, 'latency_ms': round(latency_ms, 0)})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

@app.post("/chat/stream-meta")
async def chat_stream_meta(request: dict):
    return StreamingResponse(
        stream_with_metadata(request.get("message", "")),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

---

## Streaming from Anthropic

```python
import os
import json
import anthropic
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()
ac_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class AnthropicRequest(BaseModel):
    message: str
    system: str = "You are a helpful assistant."

async def stream_anthropic(message: str, system: str):
    with ac_client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": message}],
    ) as stream:
        for text in stream.text_stream:
            yield f"data: {json.dumps({'token': text, 'done': False})}\n\n"
    yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

@app.post("/anthropic/stream")
async def anthropic_stream(request: AnthropicRequest):
    return StreamingResponse(
        stream_anthropic(request.message, request.system),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
```

---

## LangChain streaming

```python
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json

app = FastAPI()
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, api_key=os.getenv("OPENAI_API_KEY"))

async def stream_langchain(message: str):
    async for chunk in llm.astream([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=message),
    ]):
        if chunk.content:
            yield f"data: {json.dumps({'token': chunk.content})}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/langchain/stream")
async def lc_stream(request: dict):
    return StreamingResponse(
        stream_langchain(request.get("message", "")),
        media_type="text/event-stream",
    )
```

---

## Testing streaming endpoints

```python
import httpx
import json

# Test streaming endpoint
def test_stream(url: str, payload: dict) -> None:
    """Consume a streaming SSE endpoint and print each event."""
    with httpx.Client(timeout=30.0) as client:
        with client.stream("POST", url, json=payload) as response:
            full_text = ""
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "token" in data and data["token"]:
                        print(data["token"], end="", flush=True)
                        full_text += data["token"]
                    if data.get("done"):
                        print(f"\n\nStream complete. Total: {len(full_text)} chars")
                        break

# Run against a local server
# test_stream("http://localhost:8000/chat/stream", {"message": "What is machine learning?"})
```

> [!warning] Disable nginx/proxy buffering for streaming to work
> By default, nginx and many reverse proxies buffer responses before sending them to the client. This defeats streaming. Set `X-Accel-Buffering: no` in your response headers and configure your proxy with `proxy_buffering off`. Without this, the client receives the complete response all at once even though you're streaming.

---

[[01-fastapi-wrappers]] | [[03-async-patterns]]
