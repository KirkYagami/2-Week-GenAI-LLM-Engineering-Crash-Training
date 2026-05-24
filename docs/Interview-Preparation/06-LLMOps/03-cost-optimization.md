# Cost Optimization

Cost questions come up in every interview at a company that pays LLM API bills. They want to know you've thought about cost at the design level, not just as an afterthought.

---

## Q1: How do you estimate the cost of an LLM API call before sending it?

??? "Show answer"
    Token-based pricing: cost = (input_tokens × price_per_input_token) + (output_tokens × price_per_output_token).

    Estimate before sending with `tiktoken`:

    ```python
    import tiktoken, os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},        # per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    }

    def estimate_cost(model: str, messages: list, max_output_tokens: int = 500) -> float:
        enc = tiktoken.encoding_for_model(model)
        input_tokens = sum(len(enc.encode(m["content"])) for m in messages) + 10  # +10 for message overhead
        prices = PRICING.get(model, PRICING["gpt-4o"])
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (max_output_tokens / 1_000_000) * prices["output"]
        return round(input_cost + output_cost, 6)

    cost = estimate_cost("gpt-4o", messages=[{"role": "user", "content": "Summarize this document: ..."}])
    print(f"Estimated cost: ${cost:.4f}")
    ```

    Track actual cost per request from `response.usage` and compare to estimates — large divergences signal prompt bloat or runaway generation.

---

## Q2: What is the most impactful way to reduce LLM API costs?

??? "Show answer"
    Ranked by impact:

    1. **Model downgrade** — switching from `gpt-4o` ($2.50/1M input) to `gpt-4o-mini` ($0.15/1M input) is a 16× cost reduction. For many tasks (classification, simple extraction, routing), `gpt-4o-mini` matches `gpt-4o` quality.

    2. **Caching** — exact-match cache for repeated queries (support bots, FAQ systems) can eliminate 30–60% of API calls with zero quality loss.

    3. **Prompt compression** — shorter prompts = fewer input tokens. Remove redundant instructions, trim examples, shorten context.

    4. **Limit `max_tokens`** — if your expected output is 100 tokens, set `max_tokens=200`. Prevents runaway generation and makes cost predictable.

    5. **Batch processing** — for offline jobs (classification, summarization), batch requests rather than running them in real-time. Batch API (OpenAI) gives 50% cost reduction.

    Start with model downgrade + caching — these two alone typically reduce costs by 60–80% for most applications.

---

## Q3: What is Anthropic's prompt caching and how much can it save?

??? "Show answer"
    Anthropic's **prompt caching** lets you cache a prefix of your prompt (system prompt + any fixed context) for 5 minutes. On cache hit, you pay only for reading from cache (~10% of normal input token price).

    Ideal for:
    - Large system prompts (instructions + examples)
    - RAG where the same document set is repeatedly included in context
    - Few-shot examples that don't change per request

    ```python
    import os
    from anthropic import Anthropic

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": LARGE_SYSTEM_PROMPT,  # e.g., 5000 tokens of instructions
                "cache_control": {"type": "ephemeral"},  # mark for caching
            }
        ],
        messages=[{"role": "user", "content": user_query}],
    )
    # First call: full input token cost
    # Subsequent calls within 5 min: ~90% reduction on cached prefix
    ```

    With a 5000-token system prompt and 100 requests/hour, caching saves approximately 90% of system prompt input costs = significant at scale.

---

## Q4: When does caching save money vs when does it not?

??? "Show answer"
    Caching saves money when:
    - Queries repeat — support bots, FAQ systems, search with common queries
    - Context is stable — the same system prompt or document set is used across many requests
    - Traffic volume is high — cache infrastructure has fixed cost; it only pays off with sufficient volume

    Caching does **not** save money when:
    - Every query is unique — creative writing, unique document analysis, personalized responses
    - Output must be fresh — news summaries, stock prices, live data
    - Cache hit rate is below ~20% — the overhead of embedding + lookup exceeds the savings
    - Cache TTL is too short — semantic cache entries expire before similar queries arrive

    Measure your cache hit rate in production. If it stays below 15–20%, reconsider the caching strategy or extend the TTL.

---

## Q5: How do you choose between `gpt-4o` and `gpt-4o-mini` for a task?

??? "Show answer"
    Decision framework:

    ```
    Is the task simple? (classification, routing, formatting, short extraction)
    → Use gpt-4o-mini ($0.15/1M input)

    Does the task require deep reasoning? (complex analysis, multi-step problems, nuanced writing)
    → Use gpt-4o ($2.50/1M input)

    Not sure?
    → Run both on 50 representative test cases, compare quality with LLM-as-judge
    → If quality difference < 5%, use gpt-4o-mini
    ```

    Tiered model strategy for RAG:
    - `gpt-4o-mini` — query analysis, intent classification, routing decisions
    - `gpt-4o` — final answer generation from retrieved context

    This splits cost while keeping quality where it matters: at the generation step.

    In practice, `gpt-4o-mini` handles ~80% of production workloads adequately. The key is testing, not assuming.

---

*Previous: [Deployment](02-deployment.md) | Next: [LLM System Design](../07-System-Design/01-llm-system-design.md)*
