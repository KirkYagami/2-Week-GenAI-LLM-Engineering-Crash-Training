# When to Run Locally

The most common mistake with local LLMs is defaulting to one approach without examining the tradeoffs. This note gives you a decision framework and the tools to make it data-driven.

## Learning objectives

- Apply a structured decision framework to local vs API inference
- Calculate break-even cost and latency thresholds for your use case
- Identify hybrid architectures that get the best of both worlds
- Know which model families perform best at local inference sizes

---

## Decision framework

```
                    Does the data contain PII, PHI, or
                    classified information that must not
                    leave your infrastructure?
                    ┌─────────────────────────────────┐
              YES ──► Local inference required         │
              NO  ──► Continue to next check          │
                    └─────────────────────────────────┘

                    Do you need frontier-model quality?
                    (complex reasoning, nuanced writing,
                     state-of-the-art code generation)
                    ┌─────────────────────────────────┐
              YES ──► Use API (GPT-4o, Claude Sonnet) │
              NO  ──► Continue                        │
                    └─────────────────────────────────┘

                    Is your monthly API cost > $500?
                    ┌─────────────────────────────────┐
              YES ──► Evaluate local inference         │
              NO  ──► API is cheaper all-in           │
                    └─────────────────────────────────┘

                    Do you have ML engineers to maintain
                    local inference infrastructure?
                    ┌─────────────────────────────────┐
               NO ──► Stick with API                  │
              YES ──► Local inference viable           │
                    └─────────────────────────────────┘
```

---

## Cost break-even calculator

```python
from dataclasses import dataclass

@dataclass
class InferenceCost:
    name: str
    setup_cost_usd: float          # One-time hardware or deployment cost
    monthly_cost_usd: float        # Ongoing cost (amortized hardware or cloud)
    tokens_per_second: float       # Throughput
    quality_score: float           # Relative quality (1.0 = frontier)

API_OPTIONS = {
    "gpt-4o": InferenceCost(
        name="GPT-4o (API)",
        setup_cost_usd=0,
        monthly_cost_usd=0,  # Pay per token
        tokens_per_second=150,  # Network-limited
        quality_score=1.0
    ),
    "gpt-4o-mini": InferenceCost(
        name="GPT-4o-mini (API)",
        setup_cost_usd=0,
        monthly_cost_usd=0,
        tokens_per_second=200,
        quality_score=0.85
    ),
}

LOCAL_OPTIONS = {
    "llama3.1-8b-rtx4090": InferenceCost(
        name="Llama 3.1 8B Q4 (RTX 4090)",
        setup_cost_usd=1600,   # GPU cost
        monthly_cost_usd=30,   # Power + amortized hardware over 24 months
        tokens_per_second=100,
        quality_score=0.75
    ),
    "llama3.1-70b-2xa100": InferenceCost(
        name="Llama 3.1 70B Q4 (2× A100)",
        setup_cost_usd=20000,
        monthly_cost_usd=500,
        tokens_per_second=40,
        quality_score=0.95
    ),
}

def calculate_monthly_api_cost(
    daily_requests: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: str = "gpt-4o-mini"
) -> float:
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},      # per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    }

    prices = PRICING.get(model, PRICING["gpt-4o-mini"])
    monthly_input_tokens = daily_requests * avg_input_tokens * 30
    monthly_output_tokens = daily_requests * avg_output_tokens * 30

    input_cost = (monthly_input_tokens / 1_000_000) * prices["input"]
    output_cost = (monthly_output_tokens / 1_000_000) * prices["output"]

    return input_cost + output_cost

def break_even_months(
    local: InferenceCost,
    daily_requests: int,
    avg_input_tokens: int = 1000,
    avg_output_tokens: int = 500,
    api_model: str = "gpt-4o-mini"
) -> dict:
    api_monthly = calculate_monthly_api_cost(
        daily_requests, avg_input_tokens, avg_output_tokens, api_model
    )

    months_to_breakeven = None
    if api_monthly > local.monthly_cost_usd:
        months_to_breakeven = local.setup_cost_usd / (api_monthly - local.monthly_cost_usd)

    return {
        "api_monthly_usd": round(api_monthly, 2),
        "local_monthly_usd": round(local.monthly_cost_usd, 2),
        "local_setup_usd": local.setup_cost_usd,
        "break_even_months": round(months_to_breakeven, 1) if months_to_breakeven else "Never (local costs more)",
        "quality_tradeoff": f"Local is {(1 - local.quality_score) * 100:.0f}% lower quality"
    }

# Example: 5,000 requests/day with gpt-4o-mini vs local 8B model
rtx4090_local = LOCAL_OPTIONS["llama3.1-8b-rtx4090"]
result = break_even_months(
    local=rtx4090_local,
    daily_requests=5000,
    avg_input_tokens=800,
    avg_output_tokens=400,
    api_model="gpt-4o-mini"
)

print("Break-even analysis: 5,000 req/day")
for key, val in result.items():
    print(f"  {key}: {val}")
```

---

## Hybrid architectures

Local and API inference are not mutually exclusive. Hybrid architectures route requests based on complexity.

```python
import os
from openai import OpenAI
from enum import Enum

class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

# Two clients: one local (Ollama), one API
local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
api_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_complexity(query: str) -> ComplexityLevel:
    """Classify query complexity to route to appropriate model."""
    query_lower = query.lower()

    # Simple: factual lookups, short answers
    simple_patterns = ["what is", "define", "when did", "how many", "list the"]
    if any(p in query_lower for p in simple_patterns) and len(query) < 100:
        return ComplexityLevel.SIMPLE

    # Complex: multi-step reasoning, code generation, comparison
    complex_patterns = ["compare", "analyze", "explain why", "design", "implement",
                        "debug", "optimize", "write a program"]
    if any(p in query_lower for p in complex_patterns):
        return ComplexityLevel.COMPLEX

    return ComplexityLevel.MEDIUM

def smart_route(query: str, system_prompt: str = "") -> dict:
    """Route query to local or API model based on complexity."""
    complexity = classify_complexity(query)

    if complexity == ComplexityLevel.SIMPLE:
        client = local_client
        model = "llama3.1:8b"
        cost_estimate = 0.0
    elif complexity == ComplexityLevel.MEDIUM:
        client = api_client
        model = "gpt-4o-mini"
        cost_estimate = len(query.split()) * 0.0000015  # rough estimate
    else:
        client = api_client
        model = "gpt-4o"
        cost_estimate = len(query.split()) * 0.000015

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=500
    )

    return {
        "answer": response.choices[0].message.content,
        "model_used": model,
        "complexity": complexity.value,
        "estimated_cost_usd": cost_estimate
    }

# Test routing
queries = [
    "What is RAG?",
    "Explain the tradeoffs between HNSW and IVF-PQ index types.",
    "Write a Python class implementing a full RAG pipeline with ChromaDB and GPT-4o-mini.",
]

for query in queries:
    result = smart_route(query)
    print(f"\nQuery: {query[:60]}...")
    print(f"  Routed to: {result['model_used']} ({result['complexity']})")
    print(f"  Cost: ${result['estimated_cost_usd']:.4f}")
```

---

## Recommended local models by use case (2025)

| Use case | Recommended model | Size (Q4) | Why |
|---------|-----------------|-----------|-----|
| General chat | Llama 3.1 8B | 4.7 GB | Best 8B quality, excellent chat template |
| Coding | Qwen2.5-Coder 7B | 4.5 GB | Top coding benchmark scores at 7B |
| RAG Q&A | Mistral 7B v0.3 | 4.1 GB | Good instruction following, fast |
| Summarization | Phi-3 mini 3.8B | 2.3 GB | Fits on 4 GB VRAM, good quality |
| Embeddings | nomic-embed-text | 274 MB | Fast, 768-dim, good MTEB scores |
| Frontier quality | Llama 3.1 70B | 40 GB | Near GPT-4o quality, needs 2× A100 |

> [!success] The practical recommendation
> Start with the API. Build your product. When monthly API costs exceed $200–300 and you have operational bandwidth, evaluate local inference with a 30-day pilot. Run both in parallel on 10% of traffic, compare output quality, measure latency, and make the decision with data. Never migrate to local inference on a hunch.

---

[[04-quantization]] | [[06-practice-exercises]]
