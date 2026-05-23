# Interview Questions — LangChain Fundamentals

---

## Q1: What is LCEL and how does it differ from earlier LangChain chain patterns?

??? "Show answer"
    LCEL (LangChain Expression Language) is the modern composition system in LangChain 0.3+. It uses Python's `|` (pipe) operator to chain `Runnable` objects: `chain = prompt | llm | parser`.

    **Key differences from older patterns (`LLMChain`, `SequentialChain`):**

    1. **Streaming by default:** Any LCEL chain automatically supports `.stream()` and `.astream()`. Old chains required special streaming-aware implementations.

    2. **Async and batch built-in:** `.ainvoke()`, `.abatch()`, `.astream()` work on every LCEL chain without extra code. Old chains needed manual async handling.

    3. **Composability:** LCEL components are all `Runnable` objects — they have a consistent interface and can be combined freely. Old chains were monolithic classes with fixed behaviors.

    4. **LangSmith tracing:** LCEL chains automatically emit trace events that LangSmith captures. Old chains needed manual instrumentation.

    5. **Transparency:** LCEL makes data flow explicit. You can inspect what's being passed between steps. Old chains often hid intermediate state.

    The old classes still exist for backward compatibility but should not be used in new code.

---

## Q2: What are the tradeoffs between different conversation memory strategies?

??? "Show answer"
    **Full history (InMemoryChatMessageHistory):**
    - Pro: Perfect recall — the model sees everything
    - Con: Grows unbounded; eventually exceeds context window; expensive
    - Use when: Short conversations (< 20 turns), context window is large (200K+)

    **Token windowing:**
    - Pro: Bounded token cost; simple to implement
    - Con: Loses old context abruptly; model can't reference early facts
    - Use when: Conversations are episodic; early context rarely matters

    **Summary buffer:**
    - Pro: Preserves semantic content of old messages; bounded context
    - Con: Summarization costs tokens and time; lossy — specific facts may be dropped
    - Use when: Long ongoing conversations where facts from early turns matter but verbatim recall doesn't

    **External memory (retrieval-based):**
    - Pro: Effectively unlimited history; retrieve only what's relevant
    - Con: Complex to implement; summarization or embedding of history required
    - Use when: Very long conversations, or when user profile/preference storage is needed

    In production: most applications use windowing (simple) or summary buffer (better quality). External memory is for advanced agent applications.

---

## Q3: How does RunnableParallel improve performance in LCEL?

??? "Show answer"
    `RunnableParallel` runs multiple chains concurrently in separate threads (or coroutines for async). This matters when you have independent steps that don't depend on each other's output.

    ```python
    # Sequential: takes time_a + time_b
    result_a = chain_a.invoke(input)
    result_b = chain_b.invoke(input)

    # Parallel: takes max(time_a, time_b)
    results = RunnableParallel(a=chain_a, b=chain_b).invoke(input)
    ```

    For LLM calls (which are I/O-bound and take 1–5 seconds each), parallel execution roughly halves the total time for two simultaneous calls.

    **When to use it:**
    - RAG: retrieve from multiple vector stores simultaneously
    - Evaluation: run faithfulness and relevancy scorers at the same time
    - Content generation: write outline and summary in parallel
    - Multi-model comparison: call GPT and Claude on the same input simultaneously

    **When not to use it:** When step B's input depends on step A's output — sequential is required.

---

## Q4: What is the difference between `invoke`, `stream`, `batch`, and `ainvoke` in LCEL?

??? "Show answer"
    All four are methods on every `Runnable` (and therefore every LCEL chain):

    **`invoke(input)`** — Synchronous single call. Blocks until complete. Returns the final output. Use for: request/response patterns, simple scripts, testing.

    **`stream(input)`** — Synchronous streaming. Returns a generator that yields chunks as they're produced. Blocks per chunk. Use for: streaming LLM output to a user interface, real-time display.

    **`batch([input1, input2, ...])`** — Runs multiple inputs concurrently. Returns a list of results. Internally uses a thread pool. Use for: processing many items at once (eval sets, batch embedding, parallel classification).

    **`ainvoke(input)`** — Async version of invoke. Must be `await`-ed. Use in: FastAPI endpoints, async web frameworks, Jupyter notebooks with `await`.

    **`astream(input)`** — Async streaming. Yields chunks asynchronously. Use for: async web servers that need to stream responses to clients.

    **`abatch([...])`** — Async version of batch. Use in async contexts where you're processing many items concurrently.

    For most production web applications: use `ainvoke` or `astream` in async request handlers to avoid blocking the event loop.

---

## Q5: When would you choose LangChain over raw API calls?

??? "Show answer"
    **Choose LangChain when:**

    - **Multi-provider portability:** You want to swap between OpenAI, Anthropic, Ollama, or Bedrock without rewriting your application logic. LangChain's common interface handles the translation.

    - **Complex pipelines:** Retrieval + generation + post-processing + memory — coordinating these steps manually is error-prone. LCEL makes the data flow explicit.

    - **Community integrations:** LangChain has 200+ pre-built integrations: vector stores, document loaders, output parsers, tools. Not having to write these from scratch saves days.

    - **Observability:** LangSmith gives you automatic tracing, debugging, and evaluation for free on any LCEL chain.

    **Choose raw API calls when:**

    - **Single, simple call:** If your application makes one API call and returns the result, LangChain adds complexity without value.

    - **Performance-critical path:** LangChain adds ~5–20ms of overhead per chain step from serialization and abstraction. For very high-throughput applications, this matters.

    - **Custom control needed:** If you need precise control over request headers, retry behavior, or response parsing that LangChain doesn't expose, raw API calls give you full control.

    - **Team familiarity:** If your team doesn't know LangChain and the application is simple, the learning curve isn't worth it.

    The practical heuristic: use LangChain for anything with retrieval, tool use, memory, or multi-step chains. Use raw API calls for simple completion endpoints.

---

[[05-practice-exercises]] | [[../Day-01-Part-2-Advanced-RAG/00-agenda]]
