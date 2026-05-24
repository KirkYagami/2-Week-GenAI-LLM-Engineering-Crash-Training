# LLM System Design

System design is the part of the interview that separates junior from senior candidates. These questions test whether you can reason about tradeoffs — latency vs cost, accuracy vs throughput, simplicity vs scalability.

---

## Q1: What components does every production LLM system need?

??? "Show answer"
    A production LLM system needs more than just an API call. The minimal production-grade architecture:

    ```
    Client
      ↓
    API Gateway / Auth Layer        ← rate limiting, authentication, request validation
      ↓
    Application Server (FastAPI)    ← business logic, prompt assembly, response formatting
      ↓
    Cache Layer                     ← exact-match or semantic cache to avoid redundant LLM calls
      ↓
    LLM Provider (OpenAI/Anthropic) ← the actual inference
      ↓
    Observability (LangSmith/logs)  ← traces, latency, cost, quality metrics
    ```

    Additional components for RAG systems:
    - Vector database (ChromaDB, Pinecone, Qdrant) for retrieval
    - Ingestion pipeline for document processing and embedding
    - Reranker for retrieval precision

    Additional components for agent systems:
    - Tool execution layer with sandboxing
    - State store / checkpointer (Redis, Postgres)
    - Queue for long-running jobs (Celery, RQ)

---

## Q2: How do you handle latency requirements for a real-time LLM feature?

??? "Show answer"
    LLM inference is slow (1–10 seconds). Strategies to make it acceptable:

    **Streaming** — don't wait for the full response. Send tokens as they generate. Users perceive streaming as fast even when total latency is the same.

    **Caching** — for predictable queries (FAQ, common search terms), cache responses. Cache hits return in < 10ms.

    **Model selection** — `gpt-4o-mini` at 500ms beats `gpt-4o` at 2s if quality is acceptable. Benchmark both before assuming you need the larger model.

    **Parallel retrieval** — for RAG, embed the query and retrieve from the vector DB concurrently with any other setup work.

    **Request timeout + fallback** — set a hard timeout (e.g., 8 seconds). On timeout, return a cached response, a simplified answer, or a graceful error — never leave the user waiting indefinitely.

    ```python
    import asyncio, os
    from openai import AsyncOpenAI

    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def answer_with_timeout(query: str, timeout: float = 8.0) -> str:
        try:
            response = await asyncio.wait_for(
                aclient.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": query}]),
                timeout=timeout,
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            return "I'm taking longer than expected. Please try again or contact support."
    ```

---

## Q3: How do you design for high availability with LLM APIs?

??? "Show answer"
    LLM APIs have rate limits, occasional outages, and variable latency. A highly available LLM service must handle all three.

    **Multi-provider fallback**: primary on OpenAI, fallback to Anthropic on 5xx errors or rate limits.

    ```python
    import os
    from openai import OpenAI, RateLimitError, APIError
    from anthropic import Anthropic

    oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ant = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def resilient_completion(prompt: str) -> str:
        try:
            response = oai.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except (RateLimitError, APIError):
            # Fallback to Anthropic
            response = ant.messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    ```

    **Circuit breaker**: track error rate; stop calling a provider if error rate exceeds threshold; resume after a cooldown period. This prevents cascading failures from a degraded provider.

    **Retry with exponential backoff**: for transient errors (429, 503), retry with increasing delays.

---

## Q4: How do you handle PII in an LLM pipeline?

??? "Show answer"
    PII (Personally Identifiable Information) entering an LLM API is a compliance and privacy risk — you're potentially sending customer data to a third party.

    Layered approach:

    **1. Detect before sending** — use `presidio-analyzer` to identify PII entities:

    ```python
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    def anonymize_pii(text: str) -> tuple[str, list]:
        results = analyzer.analyze(text=text, language="en")
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, results  # return mapping for de-anonymization

    clean_text, pii_map = anonymize_pii(user_input)
    # Send clean_text to LLM; restore PII in response if needed
    ```

    **2. Data residency** — for strict compliance (HIPAA, GDPR), use Anthropic or OpenAI's data processing agreements and zero-data-retention options, or run a self-hosted model.

    **3. Logging** — never log raw user input. Log anonymized or hashed versions.

    **4. User consent** — inform users their input is processed by a third-party LLM API. Provide opt-out for sensitive use cases.

---

## Q5: What's your approach to fallback and degradation for LLM services?

??? "Show answer"
    LLM features should degrade gracefully, not fail completely. Design a degradation ladder:

    | Level | Condition | Response |
    |-------|-----------|----------|
    | **Normal** | LLM responds < 3s | Full AI-generated answer |
    | **Slow** | LLM responds > 3s | Stream tokens (perceived as fast) |
    | **Cache hit** | Same/similar query seen recently | Cached response, no LLM call |
    | **LLM degraded** | > 5% error rate in last 5 min | Simplified prompt on smaller model |
    | **LLM down** | Provider outage | Rule-based or template responses |
    | **Full failure** | All providers down | Static FAQ page + human escalation |

    Implement with a feature flag so you can switch degradation modes without a deployment:

    ```python
    import os

    DEGRADATION_LEVEL = os.getenv("LLM_DEGRADATION_LEVEL", "normal")

    async def answer(query: str) -> str:
        if DEGRADATION_LEVEL == "fallback":
            return static_faq_lookup(query) or "Please contact support."
        if DEGRADATION_LEVEL == "lite":
            return await call_llm(query, model="gpt-4o-mini")
        return await call_llm(query, model="gpt-4o")
    ```

---

*Previous: [Cost Optimization](../06-LLMOps/03-cost-optimization.md) | Next: [RAG Architecture](02-rag-architecture.md)*
