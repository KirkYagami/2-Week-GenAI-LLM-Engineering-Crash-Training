# Why Run LLMs Locally

The default for most teams is API-first: call OpenAI or Anthropic and don't think about infrastructure. But there are real cases where local inference is the right answer — and knowing the difference saves both money and compliance headaches.

## Learning objectives

- Identify the use cases where local inference is justified
- Understand the tradeoffs: cost, latency, quality, privacy, and engineering complexity
- Recognize the hardware landscape for consumer and server-grade local inference

---

## The case for local inference

### 1. Data privacy and compliance

Some data must not leave your organization:

- **Medical records (HIPAA)** — patient data cannot be sent to third-party APIs without a BAA
- **Legal documents** — attorney-client privileged material
- **Financial records** — PII in contracts, tax documents, internal projections
- **Defense and government** — classified information, ITAR-controlled data

If your use case involves any of these, the answer is local inference (or a private cloud deployment with a BAA, which costs significantly more than local).

### 2. Cost at scale

API pricing is attractive for low volume. At high volume, the economics shift:

```python
# Cost comparison: API vs local inference for a document processing pipeline
def api_vs_local_cost(
    daily_documents: int,
    avg_tokens_per_doc: int = 2000,
    api_price_per_1m_tokens: float = 0.15,  # gpt-4o-mini input
) -> dict:

    monthly_tokens = daily_documents * avg_tokens_per_doc * 30

    api_monthly = (monthly_tokens / 1_000_000) * api_price_per_1m_tokens

    # Local inference: one-time GPU cost
    # A100 80GB can process ~500K tokens/hour (7B model, Q4)
    # Cloud A100: ~$3/hour; amortize over 24/7 operation
    gpu_hourly = 3.0
    hours_needed_per_month = (monthly_tokens / 500_000)  # 500K tokens/hour
    local_monthly = gpu_hourly * hours_needed_per_month

    return {
        "monthly_tokens": f"{monthly_tokens:,}",
        "api_monthly_usd": f"${api_monthly:.2f}",
        "local_monthly_usd": f"${local_monthly:.2f}",
        "crossover_daily_docs": int(3.0 * 500_000 / (avg_tokens_per_doc * api_price_per_1m_tokens / 1e6))
    }

result = api_vs_local_cost(daily_documents=10_000)
print(result)
# For 10K docs/day, 2K tokens each:
# monthly_tokens: 600,000,000
# api_monthly: $90.00
# local_monthly: ~$108 (cloud GPU)
# But self-owned GPU (one-time $10K) amortizes over 3+ years
```

### 3. Latency control

API calls involve network round-trips: 100–500ms minimum even for small requests. Local inference eliminates the network — with a GPU, time-to-first-token can be < 50ms.

This matters for:
- Real-time applications (voice assistants, interactive tools)
- High-frequency batch processing where latency compounds
- Edge deployments (IoT, mobile, offline)

### 4. No rate limits

API rate limits cap throughput. If you need 1000 concurrent inferences, you need either an enterprise tier or your own infrastructure.

---

## What you give up

```
┌──────────────────────────────────────────────────────┐
│  API (OpenAI, Anthropic)          │  Local inference  │
├──────────────────────────────────────────────────────┤
│  GPT-4o / Claude Sonnet quality   │  7B–13B quality   │
│  Zero setup                       │  Days of setup    │
│  Pay-as-you-go                    │  Hardware cost    │
│  Automatic updates                │  Manual updates   │
│  No hardware to manage            │  GPU maintenance  │
│  Vision, function calling, etc.   │  Varies by model  │
│  Elastic capacity                 │  Fixed capacity   │
└──────────────────────────────────────────────────────┘
```

> [!warning] The quality gap is real
> The best open-source models (Llama 3.1 70B, Qwen2.5 72B) are excellent — close to GPT-4o on many benchmarks. But they require significant GPU resources (35–40 GB VRAM for Q4). The 7B–13B models that run on consumer hardware are noticeably weaker on complex reasoning tasks. Don't choose local inference if your use case requires frontier-model quality.

---

## Hardware landscape (2025)

| Hardware | VRAM | Max model (Q4) | Tokens/sec | Cost |
|---------|------|----------------|-----------|------|
| Apple M3 Pro 18GB | 18 GB unified | 13B | 25–40 | $2,000 |
| Apple M3 Max 48GB | 48 GB unified | 34B | 30–50 | $4,000 |
| RTX 4090 | 24 GB | 13B (or 7B float16) | 80–120 | $1,600 |
| RTX 3090 | 24 GB | 13B (Q4) | 50–80 | $700–900 used |
| RTX 4060 Ti | 16 GB | 13B (Q4) | 40–60 | $400 |
| 2× A100 80GB | 160 GB | 70B (float16) | 60–80 | $20K+ or cloud |

> [!tip] Apple Silicon for local LLMs
> Apple M-series chips use unified memory shared between CPU and GPU. An M3 Max with 96 GB can run Llama 3.1 70B in Q4 quantization at 20–30 tokens/sec. This makes MacBook Pro and Mac Studio competitive with mid-range GPU workstations for local inference.

---

[[00-agenda]] | [[02-ollama]]
