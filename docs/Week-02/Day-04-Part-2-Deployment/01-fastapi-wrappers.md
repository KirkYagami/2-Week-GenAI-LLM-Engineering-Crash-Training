# FastAPI Wrappers

Wrapping your LLM pipeline in a FastAPI endpoint is the first step to making it accessible as a service. A production endpoint needs more than just calling the OpenAI API — it needs request validation, authentication, rate limiting, error handling, and structured responses that client code can depend on.

## Learning objectives

- Build a FastAPI app that wraps an LLM pipeline
- Define Pydantic request and response models
- Implement API key authentication with dependency injection
- Add rate limiting and request size validation
- Return proper HTTP status codes for LLM errors

---

## Minimal working FastAPI LLM endpoint

```python
# main.py
import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI, APIError, RateLimitError

app = FastAPI(title="LLM API", version="1.0.0")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    system_prompt: Optional[str] = Field(None, max_length=2000)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(500, ge=1, le=4000)

class ChatResponse(BaseModel):
    response: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in 60 seconds.")
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"Upstream API error: {e.message}")

    usage = response.usage
    cost = (usage.prompt_tokens / 1e6 * 0.15) + (usage.completion_tokens / 1e6 * 0.60)

    return ChatResponse(
        response=response.choices[0].message.content,
        model="gpt-4o-mini",
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost_usd=round(cost, 8),
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

# Run with: uvicorn main:app --reload
```

---

## API key authentication

```python
import os
import secrets
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

app = FastAPI()
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

VALID_API_KEYS = {
    os.getenv("API_KEY_1", "dev-key-001"): {"user": "user1", "tier": "free", "rpm": 10},
    os.getenv("API_KEY_2", "dev-key-002"): {"user": "user2", "tier": "pro", "rpm": 100},
}

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> dict:
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required. Pass X-API-Key header.")
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return VALID_API_KEYS[api_key]

from fastapi import Depends

@app.post("/chat")
async def chat(request: dict, user_info: dict = Depends(verify_api_key)):
    # user_info contains the authenticated user's metadata
    return {"user": user_info["user"], "tier": user_info["tier"], "message": "authenticated"}
```

---

## Rate limiting with a sliding window

```python
import time
from collections import defaultdict, deque
from fastapi import HTTPException, Depends

class RateLimiter:
    def __init__(self):
        self._windows: dict[str, deque] = defaultdict(deque)

    def check(self, key: str, limit: int, window_seconds: int = 60) -> None:
        now = time.time()
        window = self._windows[key]

        # Remove expired timestamps
        while window and window[0] < now - window_seconds:
            window.popleft()

        if len(window) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {limit} requests per {window_seconds}s",
                headers={"Retry-After": str(window_seconds)},
            )
        window.append(now)

rate_limiter = RateLimiter()

def check_rate_limit(user_info: dict = Depends(verify_api_key)) -> dict:
    rpm = user_info.get("rpm", 10)
    rate_limiter.check(user_info["user"], limit=rpm, window_seconds=60)
    return user_info

@app.post("/chat-limited")
async def chat_limited(request: dict, user_info: dict = Depends(check_rate_limit)):
    return {"message": "Request accepted", "user": user_info["user"]}
```

---

## Structured error responses

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    details: str = ""

app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
        ).model_dump(),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details=str(exc) if os.getenv("DEBUG") == "true" else "",
        ).model_dump(),
    )
```

> [!success] Return 429 for rate limits, 502 for upstream errors, 422 for validation
> FastAPI automatically returns 422 for Pydantic validation failures (e.g., message too long). Return 429 with a `Retry-After` header for rate limits. Return 502 (bad gateway) rather than 500 when the OpenAI API itself fails — it signals a dependency failure, not a bug in your code.

---

[[00-agenda]] | [[02-streaming-responses]]
