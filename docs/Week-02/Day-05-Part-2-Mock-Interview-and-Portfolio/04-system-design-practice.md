# System Design Practice

System design questions test whether you can architect an LLM application at production scale — not just build a proof of concept. The interviewer wants to see that you reason about tradeoffs, not that you know a single "right" architecture.

## Learning objectives

- Structure a system design answer in 20 minutes
- Name specific components (models, databases, APIs) rather than generic boxes
- Articulate tradeoffs between design choices
- Identify the failure modes of your design

---

## The system design framework (20 minutes)

Use this structure every time:

| Phase | Time | What to do |
|-------|------|-----------|
| Clarify requirements | 0–3 min | Ask 3–5 scoping questions before drawing anything |
| High-level design | 3–8 min | Draw the main components and data flow |
| Deep dive | 8–16 min | Go deep on the 2–3 hardest components |
| Tradeoffs and failure modes | 16–20 min | State what breaks and how you'd fix it |

---

## Practice problem 1 — LLM-powered customer support system

> Design a customer support system that can answer user questions using a company's documentation, escalate to a human agent when needed, and improve over time.

### Clarifying questions to ask

- How many support tickets per day? (100 vs 100,000 changes the architecture significantly)
- What formats is the documentation in? (PDFs, Confluence, Zendesk articles?)
- What does "improve over time" mean — fine-tuning, better retrieval, or both?
- What's the latency requirement? Real-time chat (< 2s) or async (minutes)?
- What languages need to be supported?
- Is there an existing ticket system we're integrating with?

### Sample design

```
User message (web/mobile)
    ↓
API Gateway (FastAPI)
    ├── Rate limiter (sliding window per user)
    └── Auth (API key or session)
    ↓
Intent classifier (gpt-4o-mini, temperature=0)
    ├── is_escalation_needed → Route to human agent queue
    └── is_answerable → Continue to RAG
    ↓
Exact-match cache (Redis, TTL 1hr)
    ├── HIT → Return cached response (< 10ms)
    └── MISS ↓
Query embedding (text-embedding-3-small)
    ↓
Vector search (Pinecone, top-5 chunks)
    ↓
Reranker (cross-encoder, reorder top-5)
    ↓
Response generation (gpt-4o-mini, stream=True)
    ↓
Confidence check
    ├── High confidence → Stream to user + cache
    └── Low confidence → Prepend disclaimer + human escalation flag
    ↓
SSE stream to client
    ↓
Background tasks:
    - Log conversation to audit DB
    - Update ticket in CRM
    - Queue for human review if escalated
```

### Tradeoffs to discuss

**gpt-4o-mini vs gpt-4o for generation:** Mini is 10x cheaper. For factual Q&A grounded in retrieved context, mini quality is acceptable. Use gpt-4o only for edge cases or high-stakes escalation review.

**Redis vs in-memory cache:** Redis persists across restarts and works across multiple instances. In-memory cache is faster but lost on restart and doesn't work with horizontal scaling.

**Reranker or not:** A cross-encoder reranker adds 50–200ms but significantly improves retrieval precision. Include it if support quality is the primary metric; skip it if latency is the constraint.

**Failure mode:** The retrieval layer returns irrelevant chunks. Mitigation: confidence threshold on the retrieval score — if max similarity < 0.6, skip RAG and fall back to a canned "I don't know, contacting a human agent" response rather than generating a hallucinated answer.

---

## Practice problem 2 — LLM batch processing pipeline

> Design a system that processes 1 million customer reviews per day, extracting structured sentiment, topics, and urgency from each review, and storing results for downstream analytics.

### Clarifying questions to ask

- What's the latency requirement? Real-time or overnight batch?
- What's the output schema? Fixed fields or open-ended extraction?
- Does accuracy need to be verified? (Human review, ground truth labels?)
- What's the budget per review?

### Sample design

```
Reviews source (database / S3 / Kafka stream)
    ↓
Message queue (SQS or Kafka)
    ↓
Worker pool (async Python workers, 20 concurrent)
    │
    ├── Each worker:
    │   Dequeue review
    │   Hash review → check Redis cache (avoid reprocessing)
    │   Batch reviews into groups of 20 (OpenAI Batch API)
    │   Submit batch → poll for completion
    │   Parse structured output (Pydantic)
    │   Write to PostgreSQL results table
    │   Acknowledge message from queue
    │
    └── Dead letter queue for failed reviews (retry 3x, then flag)
    ↓
PostgreSQL results table
    ↓
Analytics layer (Redshift, BigQuery, or dbt)
```

### Cost calculation

At $0.075/1M input tokens (gpt-4o-mini Batch API, 50% discount):

- Average review: 100 tokens
- Extraction prompt overhead: 200 tokens
- Total input: 300 tokens/review
- 1M reviews × 300 tokens = 300M tokens/day
- Cost: 300M × $0.075/1M = **$22.50/day**

Compare to real-time API: 300M tokens × $0.15/1M = $45/day — Batch API halves the cost.

### Tradeoffs to discuss

**OpenAI Batch API:** 24-hour turnaround, 50% cost discount. Right for overnight processing. Wrong for real-time use cases.

**Parallelism strategy:** 20 concurrent async workers allows 20 × 20 = 400 reviews in flight per batch cycle. At 300ms average per review, that's ~1,333 reviews/second throughput — enough for 115M reviews/day, well above the 1M requirement.

**Failure handling:** A review that fails extraction 3 times goes to a dead letter queue for human review or fallback to rule-based extraction. Never silently drop data.

---

## Practice problem 3 — LLM coding assistant

> Design a coding assistant that can answer questions about a company's internal codebase, suggest completions, and explain code.

### Key design decisions to discuss

**Retrieval:** Codebase is a graph, not a list of documents. Code-specific chunking matters: split at function/class boundaries, not fixed tokens. Store file path, function name, and docstring as metadata.

**Embeddings:** General-purpose embeddings (text-embedding-3-small) work for most code retrieval. Code-specific models (like Voyage Code) improve precision for languages with unusual syntax.

**Context window management:** Including relevant code in the prompt can be expensive. Implement a token budget: retrieve chunks until you hit 4,000 tokens of context, then stop.

**Freshness:** The codebase changes constantly. Incremental indexing (only re-embed changed files on each commit) keeps the index fresh without re-indexing the entire codebase.

**Security:** The codebase is proprietary. The embedding model must be self-hosted or run via an enterprise API agreement with a data processing addendum. Never embed proprietary code via a public API without legal review.

---

> [!tip] Name specific technologies, not generic boxes
> "A vector database" is weaker than "Pinecone for its managed filtering and metadata query support." Specific choices signal that you've made real decisions, not theoretical ones. Have a reason for each choice.

> [!success] The goal of system design is to show your reasoning, not to get the "right" answer
> There is no single correct architecture for any of these problems. The interviewer wants to hear you reason about tradeoffs: why this database over that one, what breaks under high load, what you'd instrument to know the system is working. State tradeoffs explicitly and the interviewer will know you've thought about this before.

---

[[03-technical-interview-questions]] | [[05-mock-interview-script]]
