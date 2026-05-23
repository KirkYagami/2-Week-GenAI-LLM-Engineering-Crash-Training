# Interview Questions — LLMOps

---

**Q1: What is the difference between logging and tracing in the context of LLM applications?**

??? "Show answer"
    **Logging** records events as individual entries: "Call X happened at time T with result R." It's flat and queryable — good for counting errors, summing costs, finding slow calls.

    **Tracing** captures the causal chain of a request through your system. A trace shows parent-child relationships: the user request spawned a retrieval call, which spawned an embedding call, which spawned an LLM call. Tracing answers "why did this response happen?" — logging answers "how many times did this event occur?"

    For LLM applications, you need both. Logs tell you your p95 latency and daily cost. Traces tell you which specific prompt caused a hallucination and which tool call sequence led to a bad agent output.

    LangSmith provides traces (nested spans) with timing and token counts for each step. Your structured logs capture aggregates for dashboards and alerting.

---

**Q2: What fields should every LLM call log entry contain, and why?**

??? "Show answer"
    Minimum required fields:
    - **`trace_id`** — unique ID per call; used to correlate log entries with traces
    - **`session_id`** — groups calls from one user session; used for cost attribution and debugging
    - **`timestamp`** — ISO format; essential for time-series analysis
    - **`model`** — model version; critical because costs, capabilities, and latency differ per model
    - **`prompt_tokens` / `completion_tokens`** — enables exact cost computation and token budget tracking
    - **`latency_ms`** — for SLA monitoring and performance regression detection
    - **`finish_reason`** — "stop" vs "length" tells you if responses were truncated
    - **`function_name`** — which part of your code made the call; enables per-feature cost attribution
    - **`error`** — null on success; exception type on failure; enables error rate dashboards

    Optional but useful: `tags`, `user_id`, `input_preview`, `output_preview`.

---

**Q3: How does OpenAI's prompt caching work, and how should you structure prompts to benefit from it?**

??? "Show answer"
    OpenAI automatically caches prompt prefixes longer than 1024 tokens. Subsequent requests that share the same prefix get a 50% discount on cached input tokens (the prefix doesn't need to be re-processed by the model).

    **How to structure prompts for caching:**
    1. Put long, stable content at the beginning: system prompt, tool definitions, examples, RAG context that doesn't change per request
    2. Put dynamic content at the end: the user's specific question, current session context, per-request variables

    Bad structure: `f"{user_question}\n\nInstructions: {long_static_instructions}"` — the question changes every time, so the prefix never matches.

    Good structure: `f"{long_static_instructions}\n\nQuestion: {user_question}"` — the instructions are cached after the first call.

    Note: caching applies only after the first call and only when the prefix matches byte-for-byte. Changing even one character in the static portion breaks the cache.

---

**Q4: What is the difference between p50 and p95 latency, and why do you monitor both?**

??? "Show answer"
    **p50 (median):** The latency that 50% of requests complete faster than. Represents the typical user experience.

    **p95:** The latency that 95% of requests complete faster than. Represents the worst-case experience for most users (the slowest 5% are typically outliers).

    You need both because the mean hides bimodal distributions. If your p50 is 400ms and your p95 is 8,000ms, your average might be 800ms — which sounds fine but misses the fact that 1 in 20 users waits 8 seconds.

    For LLM applications, p95 is the SLA-relevant metric. Your service agreement should say "95% of requests complete in under X seconds." Monitor p95 in production; alert when it exceeds your threshold. Use p50 to track the typical experience and p99 to catch extreme outliers (sometimes caused by cold starts or model overload).

---

**Q5: How would you attribute LLM costs to specific features or users in a multi-feature application?**

??? "Show answer"
    Tag every LLM call with metadata at recording time:
    - `function_name`: the code function that made the call (e.g., `classify_ticket`, `generate_response`)
    - `user_id`: the authenticated user or tenant
    - `feature`: the product feature (e.g., `chat_assist`, `bulk_export`, `report_generation`)

    Then aggregate in your cost ledger by tag. A simple Python `defaultdict(float)` works for low-volume apps. For production, write tagged events to a data warehouse and query with SQL.

    This lets you answer:
    - "Which feature drives 60% of our LLM spend?" → optimize that feature first
    - "Which users exceed 1000 API calls/day?" → enforce rate limits or tier them
    - "Did the new bulk export feature cost more than projected?" → catch budget surprises before month-end

---

**Q6: What is the OpenAI Batch API, and when should you use it instead of synchronous calls?**

??? "Show answer"
    The Batch API accepts a JSONL file of requests, processes them asynchronously (up to 24 hours), and stores results in a file you download when processing completes. It offers a 50% cost discount on input and output tokens.

    **Use batch when:**
    - Results are not needed in real time (nightly evaluation runs, bulk data extraction, offline classification)
    - You have hundreds or thousands of similar requests to process (labeling, summarization, index building)
    - Cost is more important than latency for this specific workload

    **Don't use batch when:**
    - User is waiting for the response
    - Latency matters at all
    - Requests are time-sensitive (current data, user interactions)

    The 24-hour window is the key constraint. In practice, batches complete in minutes to hours for most sizes — the 24-hour window is just the maximum allowed processing time.

---

[[06-practice-exercises]]
