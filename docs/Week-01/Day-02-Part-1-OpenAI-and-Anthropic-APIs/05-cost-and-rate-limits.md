# Cost and Rate Limits

Ignoring cost and rate limits is the fastest way to ship a product that hemorrhages money or breaks under load. This section gives you the numbers and the patterns to handle both.

## Learning objectives

- Calculate token costs for real workloads
- Implement exponential backoff retry logic
- Use async concurrency within rate limits
- Apply prompt caching and model routing to reduce spend

---

## 2025 pricing reference

Prices as of May 2025 — always verify at platform.openai.com/pricing and anthropic.com/pricing.

### OpenAI

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Notes |
|-------|----------------------|------------------------|-------|
| gpt-4o | $2.50 | $10.00 | Cached input: $1.25 |
| gpt-4o-mini | $0.15 | $0.60 | Best value for simple tasks |
| o4-mini | $1.10 | $4.40 | + reasoning tokens at $1.10/1M |
| o3 | $10.00 | $40.00 | Highest capability |

### Anthropic

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Cache write | Cache read |
|-------|----------------------|------------------------|-------------|------------|
| claude-opus-4-7 | $15.00 | $75.00 | $18.75 | $1.50 |
| claude-sonnet-4-6 | $3.00 | $15.00 | $3.75 | $0.30 |
| claude-haiku-4-5 | $0.80 | $4.00 | $1.00 | $0.08 |

---

## Cost calculator

```python
import os
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class TokenCost:
    input_tokens: int
    output_tokens: int
    model: str

    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "o4-mini": {"input": 1.10, "output": 4.40},
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    }

    @property
    def cost_usd(self) -> float:
        p = self.PRICING.get(self.model, {"input": 0, "output": 0})
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1_000_000

    def __str__(self) -> str:
        return (
            f"Model: {self.model}\n"
            f"  Input:  {self.input_tokens:,} tokens\n"
            f"  Output: {self.output_tokens:,} tokens\n"
            f"  Cost:   ${self.cost_usd:.6f}"
        )

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain transformers in 100 words."}],
    max_tokens=150
)

cost = TokenCost(
    input_tokens=response.usage.prompt_tokens,
    output_tokens=response.usage.completion_tokens,
    model="gpt-4o"
)
print(cost)
```

---

## Monthly cost projection

```python
def project_monthly_cost(
    calls_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: str = "gpt-4o"
) -> dict:
    PRICING = {
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-haiku-4-5-20251001": (0.80, 4.00),
    }
    input_price, output_price = PRICING[model]

    monthly_calls = calls_per_day * 30
    monthly_input = monthly_calls * avg_input_tokens
    monthly_output = monthly_calls * avg_output_tokens
    monthly_cost = (monthly_input * input_price + monthly_output * output_price) / 1_000_000

    return {
        "model": model,
        "monthly_calls": monthly_calls,
        "monthly_input_tokens": monthly_input,
        "monthly_output_tokens": monthly_output,
        "monthly_cost_usd": monthly_cost
    }

# Example: 10K daily calls, 800 input tokens, 200 output tokens
for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]:
    result = project_monthly_cost(10_000, 800, 200, model)
    print(f"{model:35s}: ${result['monthly_cost_usd']:,.2f}/month")

# Output (approximate):
# gpt-4o                             : $660.00/month
# gpt-4o-mini                        : $39.00/month
# claude-sonnet-4-6                  : $810.00/month
# claude-haiku-4-5-20251001          : $216.00/month
```

> [!warning] Output tokens cost 4–5× more than input
> Most engineers focus on system prompt size (input tokens) but forget that a verbose model response is far more expensive. Set `max_tokens` to the minimum needed, and instruct the model to be concise when output length isn't critical.

---

## Rate limits

OpenAI and Anthropic both limit requests per minute (RPM) and tokens per minute (TPM). Default limits for new accounts:

| Provider | Tier 1 RPM | Tier 1 TPM |
|----------|-----------|-----------|
| OpenAI (gpt-4o) | 500 | 30,000 |
| OpenAI (gpt-4o-mini) | 500 | 200,000 |
| Anthropic (Sonnet) | 50 | 40,000 |

Limits increase automatically as you spend more. Check your actual limits:
- OpenAI: platform.openai.com/account/limits
- Anthropic: console.anthropic.com/settings/limits

---

## Retry with exponential backoff

```python
import time
import random
from openai import RateLimitError, APITimeoutError, APIConnectionError, APIStatusError

def with_retry(fn, max_retries: int = 5, base_delay: float = 1.0):
    """Decorator-style retry wrapper with exponential backoff + jitter."""
    for attempt in range(max_retries):
        try:
            return fn()

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            retry_after = float(e.response.headers.get("retry-after", base_delay * (2 ** attempt)))
            jitter = random.uniform(0, 0.5)
            wait = retry_after + jitter
            print(f"Rate limit — waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)

        except APITimeoutError:
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"Timeout — waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)

        except APIStatusError as e:
            if e.status_code in {500, 502, 503, 529}:  # server errors
                if attempt == max_retries - 1:
                    raise
                wait = base_delay * (2 ** attempt)
                print(f"Server error {e.status_code} — waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                raise  # 400, 401, 404 — don't retry

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

result = with_retry(lambda: client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=10
))
print(result.choices[0].message.content)
```

---

## Async batch processing within rate limits

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def process_one(text: str, sem: asyncio.Semaphore, model: str = "gpt-4o-mini") -> dict:
    async with sem:
        try:
            response = await async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Classify sentiment as positive, negative, or neutral. Respond with one word."},
                    {"role": "user", "content": text}
                ],
                max_tokens=5,
                temperature=0.0
            )
            return {
                "text": text,
                "sentiment": response.choices[0].message.content.strip().lower(),
                "tokens": response.usage.total_tokens
            }
        except Exception as e:
            return {"text": text, "error": str(e)}

async def batch_sentiment(texts: list[str], concurrency: int = 10) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [process_one(t, sem) for t in texts]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    total_tokens = sum(r.get("tokens", 0) for r in results)
    print(f"Processed {len(results)} texts, {total_tokens:,} total tokens")
    return results

# Process 100 reviews concurrently (10 at a time)
reviews = [f"Review text number {i}" for i in range(100)]
results = asyncio.run(batch_sentiment(reviews))
```

---

## Cost reduction strategies

### 1. Model routing — use cheap models for simple tasks

```python
def route_request(user_message: str) -> str:
    # Classify complexity with a cheap model first
    classification = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Is this a simple factual question or a complex reasoning task? Answer 'simple' or 'complex'.\n\nQuestion: {user_message}"
        }],
        max_tokens=5,
        temperature=0.0
    ).choices[0].message.content.strip().lower()

    model = "gpt-4o-mini" if classification == "simple" else "gpt-4o"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=500
    )
    print(f"Routed to: {model}")
    return response.choices[0].message.content
```

### 2. Response caching for repeated queries

```python
import hashlib
from functools import lru_cache

def cache_key(model: str, messages: list[dict]) -> str:
    content = f"{model}:{messages}"
    return hashlib.sha256(content.encode()).hexdigest()

_response_cache: dict[str, str] = {}

def cached_completion(model: str, messages: list[dict], max_tokens: int = 500) -> str:
    key = cache_key(model, messages)
    if key in _response_cache:
        print("Cache hit — no API call")
        return _response_cache[key]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0  # deterministic = safe to cache
    )
    result = response.choices[0].message.content
    _response_cache[key] = result
    return result
```

### 3. Batch API for non-urgent workloads

OpenAI's Batch API processes requests asynchronously at 50% discount with 24h turnaround — ideal for evaluation pipelines, bulk annotation, and offline analysis.

```python
import json

# Prepare batch file
requests = [
    {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Summarize: document {i}"}],
            "max_tokens": 100
        }
    }
    for i in range(50)
]

# Write JSONL batch file
with open("batch_requests.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

# Upload and submit
with open("batch_requests.jsonl", "rb") as f:
    batch_file = client.files.create(file=f, purpose="batch")

batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
print(f"Batch ID: {batch.id}, Status: {batch.status}")
# Poll client.batches.retrieve(batch.id) until status == "completed"
```

---

## Common mistakes

> [!warning] Forgetting `max_tokens` allows runaway costs
> Without `max_tokens`, the model can generate up to its context window limit. On a 128K model, an unconstrained generation could produce 100K+ tokens at $10/1M output tokens. Always set `max_tokens`.

> [!warning] Retrying 400 errors wastes money
> 400 errors (invalid request) will never succeed on retry. Only retry 429 (rate limit), 500/502/503 (server errors), and timeouts. Retrying 400s burns API quota and delays failure detection.

> [!warning] Token counting != word counting
> "My prompt is 500 words" does not mean 500 tokens. English text is roughly 0.75 tokens per word, but code, JSON, and non-English text can be 2–4 tokens per word. Always count tokens directly.

---

[[04-anthropic-messages-api]] | [[06-practice-exercises]]
