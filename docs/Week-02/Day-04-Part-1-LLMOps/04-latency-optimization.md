# Latency Optimization

A 4-second LLM response feels slow. A 400ms response feels fast. The techniques in this note target the four main sources of LLM latency: cold starts, time to first token, total generation time, and sequential API calls that could run in parallel.

## Learning objectives

- Measure latency at the component level (TTFT vs total)
- Implement prompt caching to reduce repeated prefix costs
- Use async/batch APIs to eliminate sequential bottlenecks
- Apply streaming to improve perceived latency
- Select models based on latency-quality tradeoffs

---

## Measuring latency: TTFT vs total

```python
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def measure_latency(model: str, messages: list[dict], n_runs: int = 3) -> dict:
    """Measure time-to-first-token and total latency via streaming."""
    ttfts = []
    totals = []

    for _ in range(n_runs):
        start = time.perf_counter()
        first_token_time = None
        tokens = 0

        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True, max_tokens=200,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens += 1

        total_time = time.perf_counter() - start
        ttft = (first_token_time - start) * 1000 if first_token_time else 0
        ttfts.append(ttft)
        totals.append(total_time * 1000)

    return {
        "model": model,
        "avg_ttft_ms": round(sum(ttfts) / n_runs, 0),
        "avg_total_ms": round(sum(totals) / n_runs, 0),
        "tokens_generated": tokens,
    }

messages = [{"role": "user", "content": "Explain how transformers work in 3 sentences."}]

for model in ["gpt-4o-mini", "gpt-4o"]:
    result = measure_latency(model, messages, n_runs=2)
    print(f"{result['model']:<20} TTFT: {result['avg_ttft_ms']:>5.0f}ms | Total: {result['avg_total_ms']:>6.0f}ms")
```

---

## Prompt caching

OpenAI's prompt caching automatically reduces cost for repeated prompt prefixes (>1024 tokens). You get it automatically — but you need to structure prompts to maximize cache hits.

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Long, stable system prompt — will be cached after the first call
LONG_SYSTEM = """You are an expert Python tutor with 15 years of teaching experience.
You follow the Socratic method: you ask guiding questions before providing answers.
You write clear, well-commented code examples.
You explain concepts with real-world analogies.
You provide common mistakes and how to avoid them.
""" * 50  # Make it long enough to trigger caching (>1024 tokens)

def ask_question(question: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": LONG_SYSTEM},  # Long static prefix
            {"role": "user", "content": question},        # Short dynamic part
        ],
    )
    usage = response.usage
    cached = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(cached, "cached_tokens", 0) if cached else 0

    return {
        "answer": response.choices[0].message.content[:100],
        "prompt_tokens": usage.prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_hit_pct": f"{cached_tokens / usage.prompt_tokens * 100:.0f}%" if usage.prompt_tokens else "0%",
    }

# First call — no cache
r1 = ask_question("What is a list comprehension?")
print(f"Call 1 — Cached: {r1['cached_tokens']}/{r1['prompt_tokens']} ({r1['cache_hit_pct']})")

# Second call — cache hit on the system prompt
r2 = ask_question("What is a generator?")
print(f"Call 2 — Cached: {r2['cached_tokens']}/{r2['prompt_tokens']} ({r2['cache_hit_pct']})")
```

**Cache design rules:**
1. Put the longest, most stable content at the beginning (system prompt)
2. Put the dynamic parts (user input, current date, session-specific data) at the end
3. Cache discounts apply at 50% on OpenAI (cached tokens cost 0.5× input price)

---

## Async for parallel LLM calls

```python
import os
import asyncio
import time
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def summarize_async(text: str, topic: str) -> str:
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Summarize this text about {topic} in one sentence."},
            {"role": "user", "content": text},
        ],
        max_tokens=100,
    )
    return response.choices[0].message.content

async def sequential_summarize(documents: list[dict]) -> list[str]:
    results = []
    for doc in documents:
        result = await summarize_async(doc["text"], doc["topic"])
        results.append(result)
    return results

async def parallel_summarize(documents: list[dict]) -> list[str]:
    tasks = [summarize_async(doc["text"], doc["topic"]) for doc in documents]
    return await asyncio.gather(*tasks)

DOCS = [
    {"text": "Machine learning is a subset of AI...", "topic": "ML"},
    {"text": "Neural networks are inspired by the brain...", "topic": "neural networks"},
    {"text": "Transformers use attention mechanisms...", "topic": "transformers"},
    {"text": "RAG combines retrieval with generation...", "topic": "RAG"},
]

async def benchmark():
    start = time.perf_counter()
    await sequential_summarize(DOCS)
    sequential_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    await parallel_summarize(DOCS)
    parallel_ms = (time.perf_counter() - start) * 1000

    print(f"Sequential: {sequential_ms:.0f}ms")
    print(f"Parallel:   {parallel_ms:.0f}ms")
    print(f"Speedup:    {sequential_ms / parallel_ms:.1f}x")

asyncio.run(benchmark())
```

---

## Batch API for offline workloads

For workloads that don't need real-time responses (nightly eval runs, bulk extraction), the OpenAI Batch API offers 50% cost reduction:

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prepare batch requests
requests = [
    {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Classify: 'Example text {i}'. Reply: positive/negative/neutral"}],
            "max_tokens": 10,
        }
    }
    for i in range(5)
]

# Write to JSONL
with open("batch_requests.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

# Upload and create batch
with open("batch_requests.jsonl", "rb") as f:
    batch_file = client.files.create(file=f, purpose="batch")

batch = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
print(f"Batch ID: {batch.id} | Status: {batch.status}")
# Check status with: client.batches.retrieve(batch.id)
# Results available when status = "completed"
```

> [!success] Parallel async calls are the single highest-impact latency optimization
> If your pipeline makes 5 sequential LLM calls that don't depend on each other, switching to `asyncio.gather()` cuts latency by 70–80% with zero change to output quality. Do this before anything else.

---

[[03-cost-tracking]] | [[05-observability]]
