# Interview Questions — OpenAI and Anthropic APIs

These questions appear in ML engineering, backend, and AI product roles. Answers are calibrated for senior-level expectations.

---

## Q1: What is the difference between `max_tokens` and `max_completion_tokens` in OpenAI's API?

??? "Show answer"
    **`max_tokens`** (used by GPT-4o and older models) limits the number of output tokens the model generates in the visible response.

    **`max_completion_tokens`** (used by o-series reasoning models: o4-mini, o3) limits the *total* tokens including internal reasoning tokens. On reasoning models, the model generates a chain-of-thought before the visible response. These thinking tokens are billed but not shown — they count against `max_completion_tokens` but not against what you receive.

    **Why it matters:** If you set `max_completion_tokens=500` on o4-mini and the model uses 400 tokens reasoning, you'll receive at most 100 tokens of visible output. Budget accordingly — reasoning-heavy prompts may need 2,000–10,000 total tokens.

---

## Q2: Explain the Anthropic tool use message flow. How does it differ from OpenAI?

??? "Show answer"
    **OpenAI tool use:**
    1. Model responds with `role: "assistant"` + `tool_calls` array
    2. You execute tools
    3. You send results as `role: "tool"` messages with matching `tool_call_id`

    **Anthropic tool use:**
    1. Model responds with `role: "assistant"` + content blocks of `type: "tool_use"`
    2. You execute tools
    3. You send results as `role: "user"` messages containing `type: "tool_result"` content blocks

    The key structural difference: Anthropic uses a `user` turn for tool results, while OpenAI uses a `tool` role. Both include a matching ID to correlate calls with results.

    **Why Anthropic chose this design:** The model is trained to treat tool results as information coming back to it (like a user message), not as a separate message class. This makes the conversation structure cleaner when mixing tool results with follow-up user messages.

---

## Q3: When would you use the Batch API instead of standard completions?

??? "Show answer"
    The OpenAI Batch API is designed for asynchronous, non-time-sensitive workloads. It offers a 50% cost discount in exchange for up to 24-hour turnaround.

    **Use Batch API for:**
    - Evaluation pipelines (scoring hundreds of model outputs overnight)
    - Bulk document classification or extraction
    - Embedding generation for large corpora
    - A/B prompt comparison on a test set
    - Any offline enrichment pipeline that doesn't need real-time results

    **Stick with standard completions for:**
    - User-facing chat (requires sub-second response)
    - Agentic loops where each step depends on the previous result
    - Any task where failure needs immediate retry

    **Practical tip:** If you're spending more than $100/month on a batch workload that runs nightly, switching to Batch API cuts that to $50/month with zero code changes beyond the submission/polling logic.

---

## Q4: What is prompt caching and when does it actually save money?

??? "Show answer"
    Prompt caching stores a prefix of your input tokens on the provider's infrastructure. Subsequent requests that share that exact prefix pay a lower read rate instead of the full input rate.

    **Anthropic caching:**
    - Cache write: 1.25× normal input rate (one-time cost)
    - Cache read: 0.10× normal rate (90% discount)
    - Minimum cache size: 1,024 tokens for Sonnet
    - TTL: 5 minutes (refreshed on each read)
    - Use `cache_control: {type: "ephemeral"}` on system content blocks

    **OpenAI prefix caching:**
    - Automatic — no configuration needed
    - Input tokens > 1,024 in a prefix match: 50% discount
    - Works with Assistants API and Chat Completions

    **Break-even analysis:** For a 2,000-token system prompt on Claude Sonnet 4.6 at $3/1M:
    - No cache: $0.006 per call
    - Cache write: $0.0075 (first call, 1.25× rate)
    - Cache read: $0.0006 per call (0.1× rate)
    - Break-even: at the *second* call, you're already saving money

    **When caching doesn't help:**
    - System prompt is under 1,024 tokens
    - Each request has a unique prefix (e.g., per-user context in the system prompt)
    - Low-volume: 1–2 calls per session

---

## Q5: How do you handle API errors in a production LLM service?

??? "Show answer"
    Production error handling requires distinguishing error types by retryability:

    **Retry with exponential backoff:**
    - `429 Rate Limit` — always retry, honor `retry-after` header
    - `500/502/503/529 Server Error` — retry with backoff (provider-side issue)
    - `Timeout` — retry, may increase timeout on subsequent attempts

    **Never retry:**
    - `400 Bad Request` — invalid input; fix the code, not the retry
    - `401 Unauthorized` — invalid API key; alert immediately
    - `404 Not Found` — wrong model name or endpoint

    **Production pattern:**
    ```python
    for attempt in range(max_retries):
        try:
            return call_api()
        except RateLimitError as e:
            wait = float(e.response.headers.get("retry-after", 2 ** attempt))
            time.sleep(wait + random.uniform(0, 0.5))  # jitter
        except APIStatusError as e:
            if e.status_code in {500, 502, 503, 529}:
                time.sleep(2 ** attempt)
            else:
                raise  # don't retry 4xx
    ```

    Beyond retries: circuit breakers to stop hammering a degraded provider, fallback to a secondary provider (OpenAI ↔ Anthropic), and alerting when error rates exceed thresholds.

---

## Q6: Compare GPT-4o vision and Claude Sonnet 4.6 vision. When would you choose one over the other?

??? "Show answer"
    Both models handle images well, but they have different strengths:

    **GPT-4o vision advantages:**
    - Strict structured output (`response_format` with Pydantic) works seamlessly with images — ideal for extraction pipelines
    - `detail: "low"` option for cheap, fast classification
    - Better at following precise extraction schemas from charts and forms

    **Claude Sonnet 4.6 vision advantages:**
    - Native PDF support as a document type (up to 100 pages) — no preprocessing needed
    - 200K context window — fit more images in one request
    - Generally stronger at document understanding and long-form analysis

    **Decision framework:**
    - Document/PDF processing → Claude (native PDF API)
    - Structured extraction with a strict schema → GPT-4o (strict structured output + vision)
    - Product image classification at scale → GPT-4o (detail: "low" for cost control)
    - Chart analysis or research documents → Either; benchmark on your data

---

## Q7: What is the difference between `tool_choice: "auto"`, `"required"`, and a specific function?

??? "Show answer"
    **`"auto"` (default):** The model decides whether to call a tool or respond directly. Use this for conversational agents where most messages need a direct answer but some need tool access.

    **`"required"`:** The model must call at least one tool. Use this when you always want structured output but don't want to force a specific function (e.g., the model should decide between `search_web` or `query_database` based on the question).

    **Specific function `{"type": "function", "function": {"name": "extract_contact"}}`:** The model must call exactly this function. Use this for pure extraction tasks where you never want a text response — you always want the structured JSON from the tool call, regardless of input.

    **`"none"`:** The model cannot call any tools. Use this if you've included tool definitions for context (e.g., you want the model to know what tools exist) but want a text response for this particular turn.

---

## Q8: How does parallel tool calling work and what are its failure modes?

??? "Show answer"
    When `parallel_tool_calls=True` (the default), the model can return multiple `tool_calls` in a single response. You must execute all of them and return all results before making the next API call.

    **The loop:**
    1. Model returns `N` tool calls in one response
    2. You execute all `N` tools (ideally concurrently with `asyncio.gather`)
    3. You append the assistant message (with all tool_calls) + N tool result messages
    4. Model generates final response using all results

    **Failure modes:**

    **1. Missing results:** If you return fewer tool results than tool calls, the API returns a 400 error. You must always return exactly one `tool` message per `tool_call_id`.

    **2. Sequential execution blocking:** If you execute the `N` tools sequentially and each takes 2s, you block for `N × 2s`. Use `asyncio.gather` for concurrent execution.

    **3. Partial failure:** If one tool fails (e.g., API timeout), you still need to return a result for that tool_call_id — return an error JSON and let the model decide what to do.

    **4. Cascading tool calls:** The model may make another round of tool calls based on the results. You need a loop, not just one iteration.

---

## Q9: How would you design a cost monitoring system for a multi-model LLM application?

??? "Show answer"
    A production cost monitor needs three layers:

    **1. Per-call tracking:**
    Intercept every API response and extract `response.usage`. Store input tokens, output tokens, model, user_id, feature, and timestamp. A middleware/wrapper function works better than adding this to every call site.

    **2. Aggregation and alerting:**
    Roll up hourly and daily totals. Alert when:
    - Daily spend exceeds 150% of rolling average
    - A specific feature/user exceeds its budget
    - A single call exceeds N tokens (likely a bug — infinite loop, huge context)

    **3. Attribution:**
    Tag each call with the originating feature (e.g., `feature="rag_search"`, `feature="summary_pipeline"`). Without attribution, you can see total cost but not which feature is driving it.

    **Practical implementation:**
    ```python
    def tracked_completion(client, feature: str, **kwargs) -> Any:
        response = client.chat.completions.create(**kwargs)
        log_cost(
            model=kwargs["model"],
            feature=feature,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
        return response
    ```

    At scale, push usage events to a time-series database (InfluxDB, Prometheus) and build a Grafana dashboard per feature.

---

## Q10: What are the tradeoffs between using one provider vs. supporting multiple?

??? "Show answer"
    **Single provider advantages:**
    - Simpler codebase — one SDK, one auth pattern, one error handling path
    - Better use of provider-specific features (prompt caching, structured output, fine-tuning)
    - Easier to optimize cost (volume discounts, commitment tiers)

    **Multi-provider advantages:**
    - Failover: if OpenAI has an outage, route to Anthropic (and vice versa)
    - Task routing: use Claude for document analysis, GPT-4o for code generation
    - Cost arbitrage: compare prices as they change and shift workloads
    - Avoid vendor lock-in

    **The hidden cost of multi-provider:**
    APIs differ in response structure, error types, token counting, and feature availability. A wrapper that abstracts both providers perfectly either limits you to the lowest common denominator (lose provider-specific features) or has a leaky abstraction (you end up writing provider-specific code anyway).

    **Practical recommendation:** Start single-provider, build clean abstractions (a `call_llm()` function), and add a second provider only when you have a concrete need (failover or task routing). LangChain and LiteLLM provide provider-agnostic wrappers if you need this from day one.

---

[[06-practice-exercises]] | [[../Day-02-Part-2-Embeddings-and-Semantic-Search/00-agenda]]
