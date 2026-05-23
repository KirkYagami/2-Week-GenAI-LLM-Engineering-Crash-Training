# LLMOps Cheat Sheet

Quick reference for tracing, cost tracking, and latency optimization.

---

## LangSmith tracing

```python
# Automatic tracing for LangChain
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "my-project"

# Manual tracing for non-LangChain code
from langsmith import traceable

@traceable(name="classify_text", tags=["classification"])
def classify(text: str) -> str:
    response = client.chat.completions.create(...)
    return response.choices[0].message.content
```

## LangSmith evaluation

```python
from langsmith import Client
from langsmith.evaluation import evaluate

ls_client = Client()

# Create dataset
dataset = ls_client.create_dataset("my-eval-dataset")
ls_client.create_examples(
    inputs=[{"question": "What is RAG?"}],
    outputs=[{"answer": "Retrieval-Augmented Generation..."}],
    dataset_id=dataset.id,
)

# Define evaluator
def correctness(run, example) -> dict:
    generated = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    score = 1.0 if expected.lower() in generated.lower() else 0.0
    return {"key": "correctness", "score": score}

# Run evaluation
results = evaluate(
    my_function,
    data="my-eval-dataset",
    evaluators=[correctness],
    experiment_prefix="v2",
)
```

## Cost tracking

```python
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},   # per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0},
}

def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0, "output": 0})
    return (prompt_tokens / 1e6 * p["input"]) + (completion_tokens / 1e6 * p["output"])

# After each API call:
usage = response.usage
cost = compute_cost("gpt-4o-mini", usage.prompt_tokens, usage.completion_tokens)
```

## Latency optimization

```python
# 1. Parallel independent calls
import asyncio
results = await asyncio.gather(call_a(), call_b(), call_c())

# 2. Prompt caching (OpenAI) — structure prompt so static parts come first
messages = [
    {"role": "system", "content": LONG_STATIC_CONTEXT},  # Cached after first call
    {"role": "user", "content": user_query},              # Changes each request
]

# 3. Streaming — reduce perceived latency
stream = client.chat.completions.create(model=..., stream=True)

# 4. OpenAI Batch API — 50% cost discount, 24h turnaround
batch = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

# 5. Token counting before API call
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")
n_tokens = len(enc.encode(text))
```

## Key metrics to track

```python
# Per-request metrics
{
    "latency_ms": ...,          # time.perf_counter() diff * 1000
    "ttft_ms": ...,             # Time to first token (streaming)
    "prompt_tokens": ...,       # usage.prompt_tokens
    "completion_tokens": ...,   # usage.completion_tokens
    "cost_usd": ...,            # compute_cost()
    "model": ...,               # Which model was used
    "cached": ...,              # Whether cache was hit
}

# System-level metrics
{
    "p50_latency_ms": ...,      # Median latency
    "p95_latency_ms": ...,      # 95th percentile
    "cache_hit_rate": ...,      # hits / (hits + misses)
    "error_rate": ...,          # 4xx/5xx / total requests
    "tokens_per_minute": ...,   # For rate limit planning
    "daily_cost_usd": ...,      # Sum of per-request costs
}
```

## Observability checklist

- [ ] Every LLM call is traced (LangSmith or custom)
- [ ] Cost tracked per model, per function, per user
- [ ] P95 latency measured (not just average)
- [ ] Cache hit rate monitored
- [ ] Error rate with error type breakdown
- [ ] Alert when daily cost exceeds budget threshold
