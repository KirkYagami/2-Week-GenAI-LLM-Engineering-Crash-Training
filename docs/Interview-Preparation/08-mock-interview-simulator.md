# Mock Interview Simulator

Practice questions with progressive difficulty. Start with the warm-up, time yourself on each question, then check the answer. Cover the answer before you read it.

---

## Rules for self-practice

1. Set a 3-minute timer for each question.
2. Answer out loud, not in your head.
3. Only read the answer after your time is up.
4. Rate yourself: 1 = didn't know, 2 = partial, 3 = clear and complete.

Repeat 1-rated questions the next day.

---

## Round 1 — Warm-up (fundamental concepts)

**Q1: What is a token? How do you count tokens?**

??? "Show answer"
    A token is the smallest unit of text that a language model processes. It's roughly 4 characters or 3/4 of a word on average for English. "Hello world" is 2 tokens; "antidisestablishmentarianism" is 5–6 tokens.
    
    Count tokens with the `tiktoken` library:
    ```python
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    n_tokens = len(enc.encode("Hello world"))
    ```
    
    Why it matters: token count determines cost (billed per 1M tokens) and whether a prompt fits in the context window.

---

**Q2: What is the context window and why does it matter for RAG?**

??? "Show answer"
    The context window is the maximum number of tokens a model can process in one forward pass — both input and output combined. For gpt-4o and claude-sonnet-4-6, this is 128k–200k tokens.
    
    For RAG: the retrieved chunks plus the user's question plus any conversation history must fit within the context window. A 10-page document retrieved as 5 chunks of 800 tokens each = 4,000 tokens of context. The LLM then generates up to `max_tokens` in reply. Fail to account for this and you get a context overflow error at runtime.

---

**Q3: What is the difference between `stop` and `max_tokens`?**

??? "Show answer"
    `max_tokens` sets a hard limit: the model stops generating after this many tokens regardless of content. It's a safety cap.
    
    `stop` is a list of stop sequences: the model stops generating when it produces one of these strings. Common use: `stop=["Observation:"]` in ReAct agents to prevent the model from generating its own observations.
    
    They can be used together: `max_tokens=500, stop=["Final Answer:"]` means "stop when you see 'Final Answer:' or after 500 tokens, whichever comes first."

---

## Round 2 — Core topics

**Q4: Walk me through how you'd implement exact-match caching for an LLM API endpoint.**

??? "Show answer"
    Three components:
    
    1. **Cache key** — hash the full request deterministically:
    ```python
    import hashlib, json
    def cache_key(messages, model, temperature):
        payload = json.dumps({"messages": messages, "model": model, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
    ```
    
    2. **Cache store** — in-memory dict with TTL:
    ```python
    _cache = {}
    CACHE_TTL = 600  # 10 minutes
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < CACHE_TTL:
        return entry["response"]
    ```
    
    3. **Cache write** — only for deterministic requests:
    ```python
    if temperature == 0.0:  # Don't cache non-deterministic
        _cache[key] = {"response": result, "ts": time.time()}
    ```
    
    For production: replace the dict with Redis for persistence across restarts and multi-instance deployments.

---

**Q5: What is the difference between precision and recall in an extraction context?**

??? "Show answer"
    In structured extraction (e.g., extracting fields from an invoice):
    
    - **Precision** = of all fields the model claimed to extract, what fraction were correct? High precision means few false extractions.
    - **Recall** = of all fields that should have been extracted (per ground truth), what fraction did the model find? High recall means few missed fields.
    
    F1 = harmonic mean of precision and recall. Use F1 when both matter equally.
    
    Example: Ground truth has 5 fields. Model extracts 4 correctly and hallucinates 1 wrong field. Precision = 4/5 = 0.80. Recall = 4/5 = 0.80. F1 = 0.80.

---

**Q6: Explain what `prepare_model_for_kbit_training` does.**

??? "Show answer"
    After loading a model in 4-bit (QLoRA), the quantized weights can't hold gradients directly — the quantization format (`nf4`) doesn't support float gradient updates. `prepare_model_for_kbit_training()` does three things:
    
    1. Casts the model's layer norms and embedding layers back to float32 (these need full precision for stable training).
    2. Enables gradient checkpointing (saves memory by recomputing activations instead of storing them).
    3. Sets up the model so that only the LoRA adapter weights (in float32 or float16) receive gradient updates — the quantized base model weights remain frozen.
    
    Without this call, training will fail with dtype mismatch errors or produce NaN losses.

---

## Round 3 — System design and architecture

**Q7: You're building an LLM service that 100 users will share. One user submits a 200-page document for summarization — a 30-minute job. How do you prevent this from blocking other users?**

??? "Show answer"
    Don't process it synchronously in the HTTP request. Use an async task queue:
    
    1. The `/summarize` endpoint receives the upload, stores the file, creates a task ID, and queues the job. Returns `{"task_id": "abc123", "status": "queued"}` immediately.
    2. A background worker picks up the job, processes it (map-reduce over chunks), and writes the result to a database.
    3. The user polls `/status/abc123` until it shows `"status": "done"` with the result.
    
    For the queue: Celery + Redis for production, FastAPI's `BackgroundTasks` for simple cases (doesn't survive server restart). The key invariant: no synchronous HTTP handler should take more than 30 seconds — long-running work always goes to a queue.

---

**Q8: How would you detect when your RAG system is hallucinating?**

??? "Show answer"
    Three approaches at increasing cost:
    
    1. **Low confidence fallback** — if the maximum retrieval score < 0.6, don't generate. Return "I couldn't find relevant information" instead of risking hallucination.
    
    2. **Post-generation faithfulness check** — after generating, use an LLM judge to verify each claim: "Can this claim be found verbatim or by direct inference from the context? Answer yes/no for each claim." Flag responses where < 80% of claims are supported.
    
    3. **RAGAS faithfulness metric** — in an evaluation pipeline (not real-time), RAGAS decomposes the answer into atomic claims and checks each against the retrieved context.
    
    For real-time detection (approach 2), the check adds ~500ms latency. It's worth it for high-stakes domains (medical, legal, financial). For general Q&A, approach 1 + prompt constraints ("only answer if the context contains the information") is usually sufficient.

---

## Round 4 — Behavioral and experience

**Q9: Tell me about a time the LLM returned an output that broke your application. How did you fix it?**

*(No model answer — this should come from your own experience. If you don't have one, describe what you'd do hypothetically with a specific scenario: "If the model returned a JSON string with an extra trailing comma that broke `json.loads()`...")*

---

**Q10: If you had to pick one metric to monitor for a production LLM API, what would it be and why?**

??? "Show answer"
    P95 latency. 
    
    - P50 (median) hides tail behavior. A service with 1s P50 and 30s P95 feels broken to 1 in 20 users.
    - Error rate is important but doesn't tell you about slow requests that succeed.
    - Cost matters but it's a lagging indicator — you've already spent the money by the time you see it.
    
    P95 latency is leading: it catches slowdowns before they become outages, correlates with user experience, and helps diagnose whether the problem is in retrieval, LLM generation, or network.
    
    If forced to pick a second: daily cost, because it's the feedback signal that tells you whether the system is being used efficiently.

---

[[README]]
