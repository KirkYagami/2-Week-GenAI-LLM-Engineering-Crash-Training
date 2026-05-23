# OpenAI Chat Completions

The Chat Completions API is the core interface for GPT-4o and o-series models. Everything else — vision, tool use, structured output — is built on top of it.

## Learning objectives

- Send synchronous and streaming chat completions
- Configure all production-relevant parameters
- Handle errors, retries, and rate limits correctly
- Run concurrent requests with `asyncio`

---

## Basic request anatomy

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100,
    temperature=0.0
)

print(response.choices[0].message.content)
# Output: Paris is the capital of France.

print(f"Input tokens:  {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
```

The response object follows a consistent schema across all calls — always access content via `response.choices[0].message.content`.

---

## Key parameters

| Parameter | Type | Default | What it does |
|-----------|------|---------|--------------|
| `model` | str | required | Model ID: `gpt-4o`, `gpt-4o-mini`, `o4-mini`, `o3` |
| `messages` | list | required | Conversation history as role/content dicts |
| `max_tokens` | int | model max | Hard cap on output length |
| `temperature` | float | 1.0 | Randomness (0 = deterministic, 2 = very random) |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `frequency_penalty` | float | 0.0 | Penalizes repeated tokens (-2 to 2) |
| `presence_penalty` | float | 0.0 | Penalizes any previously used tokens (-2 to 2) |
| `stop` | list[str] | None | Stop sequences — generation halts at first match |
| `seed` | int | None | Reproducibility seed (best-effort, not guaranteed) |
| `n` | int | 1 | Number of completions to generate |
| `logprobs` | bool | False | Return log-probabilities for each output token |
| `stream` | bool | False | Stream tokens as they are generated |

> [!tip] Temperature for production
> Set `temperature=0.0` for extraction, classification, and structured output — deterministic responses are easier to test. Use `temperature=0.7–1.0` for creative tasks. Never set above 1.0 in production.

---

## Multi-turn conversations

The API is stateless — you must pass the full conversation history on every call.

```python
def chat_session(client: OpenAI, system: str) -> None:
    messages = [{"role": "system", "content": system}]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        assistant_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_reply})
        print(f"Assistant: {assistant_reply}\n")

chat_session(client, "You are a concise technical advisor.")
```

> [!warning] Context window management
> Every turn grows the messages list. At ~100K tokens you hit the context limit and the API returns a 400 error. Implement a sliding window or summarization strategy for long sessions — see `03-context-windows.md` for patterns.

---

## Streaming responses

Streaming returns tokens as they are generated, reducing perceived latency significantly.

```python
def stream_response(prompt: str) -> str:
    full_response = ""

    with client.chat.completions.stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    print()  # newline after stream ends
    return full_response

result = stream_response("Explain gradient descent in 3 sentences.")
```

The `stream.text_stream` helper (from the OpenAI SDK >= 1.30) iterates text deltas directly. For lower-level control:

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Count to 5."}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

---

## Async requests

For server applications handling many concurrent users, use the async client to avoid blocking the event loop.

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_completion(prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:  # rate-limit: max 5 concurrent requests
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content

async def batch_completions(prompts: list[str]) -> list[str]:
    sem = asyncio.Semaphore(5)
    tasks = [get_completion(p, sem) for p in prompts]
    return await asyncio.gather(*tasks)

# Process 20 prompts concurrently (5 at a time)
prompts = [f"Summarize topic #{i} in one sentence." for i in range(20)]
results = asyncio.run(batch_completions(prompts))
print(f"Got {len(results)} responses")
```

> [!tip] Semaphore for rate limits
> OpenAI's default rate limit is 500 RPM (requests per minute) for tier-1 accounts. A semaphore of 5–10 concurrent requests is safe for most workloads. Increase only after checking your account's actual limits at platform.openai.com/account/limits.

---

## Error handling and retries

The OpenAI SDK raises typed exceptions you should handle explicitly.

```python
import time
from openai import RateLimitError, APITimeoutError, APIConnectionError

def robust_completion(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                timeout=30.0
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # exponential back-off: 1s, 2s, 4s
            print(f"Rate limit hit, waiting {wait}s...")
            time.sleep(wait)

        except APITimeoutError:
            if attempt == max_retries - 1:
                raise
            print(f"Timeout on attempt {attempt + 1}, retrying...")

        except APIConnectionError as e:
            raise RuntimeError(f"Cannot reach OpenAI API: {e}") from e

    raise RuntimeError("Max retries exceeded")
```

> [!warning] Never catch bare `Exception`
> Catching `Exception` hides bugs. Catch only the specific OpenAI exception types: `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `APIStatusError`. Let unexpected errors propagate so you can see them.

---

## o-series reasoning models

The `o4-mini` and `o3` models use a different parameter set — they think before responding and don't support `temperature`.

```python
response = client.chat.completions.create(
    model="o4-mini",
    messages=[
        {"role": "user", "content": "A snail climbs 3 feet up a 10-foot wall each day and slides back 2 feet each night. How many days to reach the top?"}
    ],
    reasoning_effort="high",  # "low", "medium", or "high"
    max_completion_tokens=2000  # includes reasoning tokens
)

print(response.choices[0].message.content)
# Output: 8 days (with step-by-step reasoning)
print(f"Reasoning tokens: {response.usage.completion_tokens_details.reasoning_tokens}")
```

> [!info] o-series vs GPT-4o
> o-series models are optimized for multi-step reasoning (math, coding, analysis). They're slower and cost more per output token because they generate "thinking" tokens before the visible response. Use GPT-4o for latency-sensitive tasks and o4-mini for problems that need deep reasoning.

---

## Model selection guide

| Use case | Recommended model | Why |
|----------|-------------------|-----|
| Chat, summarization, Q&A | `gpt-4o` | Best capability/cost ratio |
| High-volume, simple tasks | `gpt-4o-mini` | ~10× cheaper than gpt-4o |
| Complex reasoning, math, coding | `o4-mini` | Reasoning effort = better accuracy |
| Hardest reasoning tasks | `o3` | Highest capability, highest cost |
| Structured extraction | `gpt-4o` + structured output | Native JSON schema enforcement |

---

## What's next

You can send text — now learn to send images and documents.

[[00-agenda]] | [[02-vision-and-multimodal]]
