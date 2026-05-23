# Technical Interview Questions

These are the questions you will actually be asked in LLM engineering interviews. Each question has a model answer — read it, then close the note and practice answering from memory. The goal is fluency, not recitation.

## Learning objectives

- Answer the 20 most common LLM engineering interview questions clearly and concisely
- Know the tradeoffs, not just the definitions
- Identify which questions require a concrete example from your own work

---

## RAG and retrieval

**Q: Explain how RAG works and when you would use it instead of fine-tuning.**

??? "Show answer"
    RAG retrieves relevant documents from an external knowledge base and includes them in the LLM's context window at inference time. The LLM generates a response grounded in the retrieved content rather than relying solely on training data.

    Use RAG when: your knowledge base changes frequently (a file system or database that updates daily), your data is too large to fit in a fine-tuning dataset, you need citations (RAG can cite the specific chunks it used), or you need to stay current without retraining.

    Use fine-tuning when: you need the model to adopt a specific style or format consistently, you have a closed domain with stable knowledge, or you want to improve performance on a narrow task with labeled data at lower inference cost.

    The key tradeoff: RAG is flexible and updatable but adds retrieval latency and depends on retrieval quality. Fine-tuning is fast at inference but expensive to update.

---

**Q: What is the difference between a bi-encoder and a cross-encoder? When do you use each?**

??? "Show answer"
    A bi-encoder embeds the query and each document independently, then compares embeddings with cosine similarity. This is fast because document embeddings can be pre-computed and indexed.

    A cross-encoder takes a (query, document) pair as input and outputs a relevance score. It's much more accurate because it can model query-document interactions directly, but it requires a forward pass per candidate document — making it too slow for retrieval over large corpora.

    The standard pattern: use a bi-encoder for fast first-stage retrieval (top-50 candidates), then a cross-encoder for reranking (reorder the top-50, take top-5). This gives you accuracy close to cross-encoder-only at cost close to bi-encoder-only.

---

**Q: What is HyDE and when does it help?**

??? "Show answer"
    HyDE (Hypothetical Document Embeddings) generates a hypothetical answer to the query using the LLM, then embeds that hypothetical answer instead of (or in addition to) the original query for retrieval.

    It helps when: the query is very short or poorly phrased ("transformer attention" won't embed well for retrieving a detailed technical paragraph), or when the query domain vocabulary differs from the document vocabulary.

    It hurts when: the LLM's hypothetical answer is hallucinated and confidently wrong — you end up retrieving content that supports a false premise. Don't use HyDE on factual queries where you can't verify the hypothetical.

---

## LLM APIs and fundamentals

**Q: What is temperature and how does it affect LLM outputs?**

??? "Show answer"
    Temperature scales the logits before the softmax at each token generation step. At temperature 0, the model always picks the highest-probability token (greedy decoding) — deterministic and consistent. At temperature 1, the model samples from the unmodified probability distribution. At temperature > 1, the distribution flattens and the model samples more randomly.

    Use temperature 0 for: classification, structured extraction, code generation, any task where you need reproducible outputs. Use temperature 0.3–0.7 for: chat and Q&A where some variation is acceptable. Use temperature 0.7–1.0 for: creative writing, brainstorming, generating diverse alternatives.

    Never use temperature > 0 when caching — the cache key must match the input exactly, and non-deterministic outputs mean cache hits would return stale responses.

---

**Q: What is the difference between `max_tokens` and the context window?**

??? "Show answer"
    The context window is the total number of tokens the model can process in a single forward pass — input (prompt + conversation history) plus output combined. For gpt-4o, this is 128k tokens.

    `max_tokens` is a parameter you set to limit the maximum length of the model's output. If `max_tokens=500`, the model will stop generating after 500 output tokens even if it hasn't finished.

    The relationship: `prompt_tokens + max_tokens ≤ context_window`. If your prompt is 2000 tokens and the context window is 4096, setting `max_tokens=3000` will cause an error because 2000 + 3000 > 4096.

---

**Q: How do you handle rate limits in a production LLM service?**

??? "Show answer"
    Rate limits come in two forms: requests per minute (RPM) and tokens per minute (TPM). Handling both requires:

    1. **Exponential backoff with jitter** — on a 429 response, wait `base_delay * 2^attempt + random_jitter` before retrying. The OpenAI SDK does this automatically; don't implement it yourself.

    2. **Client-side rate limiting** — use a semaphore (`asyncio.Semaphore(max_concurrent=5)`) to limit concurrent requests. This prevents bursting into rate limits.

    3. **Request queuing** — for batch workloads, a queue with a rate-limited worker is more reliable than parallel requests that hit limits.

    4. **Tier upgrades** — if you're consistently hitting limits in production, the correct fix is upgrading your API tier, not adding more retry logic.

    Return HTTP 429 with a `Retry-After` header to clients of your own API when you've hit downstream limits.

---

## Agents and LangGraph

**Q: What is the ReAct pattern and why does it work?**

??? "Show answer"
    ReAct interleaves reasoning and acting in a loop: the model generates a Thought (analysis), then an Action (tool call), then receives an Observation (tool result), then generates another Thought based on the observation, and so on until it has enough information to produce a Final Answer.

    It works because it forces the model to externalize its reasoning before acting, which reduces hallucination (the model is less likely to confabulate a tool result when it has to explicitly reason about what it expects). The observation step grounds subsequent reasoning in real data rather than imagination.

    The key implementation detail: pass `stop=["Observation:"]` to the API so the model stops generating when it outputs an Action, allowing you to execute the tool and inject the real observation before the next turn.

---

**Q: When would you use LangGraph instead of a simple chain?**

??? "Show answer"
    Use LangGraph when you need: state that persists across multiple LLM calls (conversation memory, accumulated research results), conditional routing (different nodes execute based on the output of a previous node), loops (a critic-rewrite loop that runs until quality is acceptable), or parallel execution of multiple nodes.

    A simple chain is appropriate for: fixed linear pipelines with no branching, single-turn operations, or when you don't need persistent state.

    The overhead of LangGraph (state schema, node functions, edge definitions) is worth it when the alternative is manually managing state dictionaries and if/else routing logic that becomes unreadable as the graph grows.

---

## Deployment and production

**Q: Why does using the synchronous OpenAI client in a FastAPI endpoint cause problems?**

??? "Show answer"
    FastAPI runs on an async event loop. A synchronous API call (from the `OpenAI` client) blocks the thread — while waiting for the LLM response, the event loop cannot serve other requests. This effectively serializes all concurrent requests through a single-file line, eliminating the concurrency benefit of async.

    The fix: use `AsyncOpenAI` and `await aclient.chat.completions.create(...)`. This yields control to the event loop at each network wait, allowing other requests to be served concurrently.

---

**Q: What is exact-match caching and when should you avoid it?**

??? "Show answer"
    Exact-match caching hashes the full request (messages, model, temperature) and returns the cached response if it's been seen before and hasn't expired. It's effectively O(1) lookup with zero LLM cost for repeated queries.

    Avoid it when: temperature > 0 (non-deterministic outputs would serve stale responses), the response includes personalized data (a cached response could be served to the wrong user — a data breach), or the response depends on real-time data (prices, weather, news).

    Always include `user_id` in the cache key for any endpoint that returns personalized content.

---

## Fine-tuning

**Q: What is LoRA and why is it more memory-efficient than full fine-tuning?**

??? "Show answer"
    LoRA (Low-Rank Adaptation) freezes the original model weights and adds trainable low-rank matrices to the attention layers. Instead of updating a weight matrix W (d × d), it learns two matrices A (d × r) and B (r × d) where r << d. At inference, the LoRA delta is W + AB — the same model, same architecture, no extra latency.

    Memory efficiency comes from r. For a 7B model with d=4096, a full attention layer weight matrix is 4096 × 4096 = 16.7M parameters. With r=16, LoRA adds 4096×16 + 16×4096 = 131k parameters — less than 1% of the original. Only these small matrices need gradients and optimizer states.

    QLoRA adds quantization: the frozen base model is loaded in 4-bit NF4 format (~3.5 bits/param effective), reducing memory by another 8x. A 7B model that requires 14GB in float16 fits in ~6GB with QLoRA.

---

**Q: When should you fine-tune instead of using few-shot prompting?**

??? "Show answer"
    Fine-tune when you have labeled examples, a narrow task, and at least one of these is true:

    - Few-shot examples make the prompt too long (high token cost at scale)
    - The task requires a specific output format or style that few-shot examples don't reliably produce
    - You need consistent behavior across thousands of diverse inputs (prompt variance is too high)
    - Latency or cost matters and a smaller fine-tuned model can replace a larger prompted model

    Stick with few-shot when: you have fewer than a few hundred examples, the task changes frequently (fine-tuning requires re-training), or the task is complex reasoning that benefits from the base model's broad knowledge.

---

## Evaluation

**Q: What is RAGAS and what does faithfulness measure?**

??? "Show answer"
    RAGAS is an evaluation framework for RAG systems. It measures four metrics without requiring human annotations: faithfulness, answer relevancy, context recall, and context precision.

    Faithfulness measures whether every claim in the generated answer can be attributed to the retrieved context — it catches hallucinations. It works by: (1) decomposing the answer into atomic claims using an LLM, (2) checking each claim against the context using an LLM, (3) computing the fraction of claims that are supported.

    A faithfulness score of 1.0 means every sentence in the answer is grounded in the retrieved context. A score of 0.6 means 40% of claims cannot be verified against what was retrieved — those are likely hallucinations.

---

[[02-portfolio-and-github]] | [[04-system-design-practice]]
