# Interview Questions — Advanced RAG

These questions test whether you understand not just what each technique does, but *when* to apply it and *why* it works.

---

**Q1: What is the fundamental difference between a bi-encoder and a cross-encoder, and why does it matter for RAG?**

??? "Show answer"
    A **bi-encoder** embeds the query and each document independently, then computes similarity (usually dot product or cosine) at retrieval time. Because embeddings are precomputed, bi-encoders can search millions of documents in milliseconds.

    A **cross-encoder** takes the query and a single candidate document as a joint input and outputs a relevance score. It sees the full interaction between query and document, so its scores are much more accurate — but it must process every candidate at inference time, making it O(n) in the number of candidates.

    **Why it matters for RAG:** You can't cross-encode against an entire corpus (too slow). The standard pattern is two-stage: bi-encoder retrieves a candidate pool of 50–100 documents quickly, then cross-encoder reranks that pool to find the top-5. You get near-cross-encoder precision at bi-encoder scale.

---

**Q2: Explain HyDE. When does it improve retrieval recall, and when can it hurt?**

??? "Show answer"
    **HyDE (Hypothetical Document Embeddings):** Instead of embedding the user's question and searching for similar documents, you ask the LLM to generate a hypothetical answer to the question, embed that hypothetical, then use the hypothetical's embedding as the search vector.

    The intuition: questions and answers occupy different regions of embedding space. Embedding a hypothetical answer brings the search vector closer to where real answers live.

    **When it helps:**
    - Short factual questions where the answer vocabulary differs from the question vocabulary ("What is the capital of X?" vs "The capital is Y")
    - Technical documentation where user phrasing and document phrasing diverge
    - Low-overlap corpora where query-document similarity is naturally sparse

    **When it hurts:**
    - Ambiguous questions (the LLM's hypothetical may pick one interpretation, biasing retrieval toward documents that support a wrong answer)
    - Highly specialized domains where the LLM will hallucinate domain-specific vocabulary, steering retrieval toward irrelevant documents
    - When bi-encoder recall is already high — HyDE adds latency and cost with no benefit

    **Mitigation:** Generate multiple hypotheticals (3–5) covering different interpretations and average their embeddings. This distributes the search vector across interpretations.

---

**Q3: Multi-query retrieval generates multiple query variants and merges results. What are two common merging strategies, and what are the tradeoffs?**

??? "Show answer"
    **Strategy 1 — Vote-based (reciprocal rank fusion or simple vote count):**
    Documents retrieved by more query variants rank higher. A document retrieved by 4 out of 5 variants is likely more broadly relevant than one retrieved by only 1.

    - *Pro:* Robust to individual query variant failures. A bad variant can't dominate the final ranking.
    - *Con:* Ignores embedding score magnitude — a document with score 0.99 on one variant beats a document with score 0.80 on five variants.

    **Strategy 2 — Max score:**
    For each document, take the highest score it received across all variants. Documents that are a very strong match for any single variant rank high.

    - *Pro:* Rewards strong relevance to at least one query angle.
    - *Con:* A single high-similarity variant can surface a document that's off-topic for the actual question.

    **Best practice:** Use vote-based as primary sort key, break ties with max score. This combines breadth (vote count) with strength (max score).

---

**Q4: What is contextual compression, and why is it especially valuable for large chunks?**

??? "Show answer"
    Contextual compression extracts only the portion of a retrieved chunk that's relevant to the user's question, discarding surrounding noise before passing context to the LLM.

    **Why it matters with large chunks:** If you chunk at 500–1000 words (to preserve context), each chunk typically covers multiple topics. For any given question, only 10–30% of the chunk is relevant. That irrelevant material:
    - Wastes context window tokens (increasing cost)
    - Can distract the LLM, leading it to generate answers based on adjacent but irrelevant content
    - Reduces faithfulness scores by making grounding checks harder

    **Two approaches:**

    | Approach | Quality | Latency | Cost |
    |----------|---------|---------|------|
    | LLM extraction | Highest — can paraphrase and understand nuance | +400–700ms per chunk | +~$0.002 per chunk |
    | Similarity filter | Good — removes low-similarity chunks entirely | +5ms | Negligible |

    **Decision rule:** Use LLM extraction when chunks are large (500+ words) and context quality is critical. Use similarity filtering as a fast pre-filter before reranking. For small focused chunks (under 200 words), skip compression — the overhead isn't justified.

---

**Q5: Describe parent-child (small-to-big) retrieval. What problem does it solve that standard chunking can't?**

??? "Show answer"
    **Problem with standard chunking:** If you use large chunks for retrieval, embedding similarity is diluted (the chunk covers too many topics). If you use small chunks, the retrieved context may be too narrow to answer the question — the relevant sentence is there but the surrounding explanation is missing.

    **Parent-child retrieval solves this:**
    - Index *child chunks* (50–100 words) for retrieval — small chunks match queries precisely
    - Store *parent documents* (the full original section or document) for generation
    - When a child chunk matches, return the parent document to the LLM

    This gives you the retrieval precision of small chunks and the generative context of large ones.

    **Implementation sketch:**
    1. Split each document into small child chunks
    2. Build a `child_to_parent` mapping
    3. At retrieval: embed query → find top child chunks → look up their parents → deduplicate → pass parents to LLM

    **Best use case:** Long documents with dense technical detail where a narrow chunk answers "is this relevant?" but the surrounding text is needed to actually answer the question (e.g., API reference docs, legal clauses, research papers).

---

**Q6: What is self-RAG, and how does it differ from standard RAG?**

??? "Show answer"
    **Standard RAG:** retrieve → generate → return. There's no check on whether the generated answer is actually supported by the retrieved context.

    **Self-RAG adds a verification loop:**
    1. Retrieve context
    2. Generate answer
    3. Verify: is this answer grounded in the retrieved context? (Ask the LLM or use NLI)
    4. If not grounded and retries remain: modify the query and retry from step 1
    5. Return the answer only when grounded (or after max retries)

    **Why it matters:** A retrieval failure (wrong documents retrieved) doesn't just cause hallucination — it causes *silent* hallucination. The model generates a confident-sounding answer based on inadequate context. Self-RAG catches this before the answer reaches the user.

    **Tradeoffs:**
    - *Pro:* Reduces hallucination from retrieval failures; gives you a grounded/not-grounded signal
    - *Con:* Adds 1–2 additional LLM calls per query; latency doubles in the worst case

    **When to use:** Production systems where hallucination is costly (medical, legal, financial). For high-throughput, low-stakes applications the latency cost is too high.

---

**Q7: A colleague says "we should add all advanced RAG techniques — multi-query, HyDE, reranking, compression, and self-RAG." What's wrong with this approach?**

??? "Show answer"
    The problem is cost and complexity without measured benefit.

    **Each technique adds:**
    - **Multi-query:** 1 extra LLM call, ~300ms, ~$0.001 per query
    - **HyDE:** 1–3 extra LLM calls, ~500ms, ~$0.002 per query
    - **Reranking:** CPU inference on 50–100 pairs, ~100–300ms
    - **Contextual compression:** 1 LLM call per retrieved chunk, ~500ms × 5 chunks = 2.5s, ~$0.01 per query
    - **Self-RAG:** 2 extra LLM calls, ~600ms–1s

    Combined, you've added 4–5 seconds and ~$0.015 per query compared to basic RAG.

    **The correct approach:**
    1. Measure your current failure mode using RAGAS: is it low context recall? low precision? low faithfulness?
    2. Apply only the technique that addresses your biggest measured failure
    3. Measure again before adding the next technique
    4. Stop when improvements plateau or cost exceeds acceptable thresholds

    Each technique solves a specific failure mode. Applying all of them blindly addresses problems you may not have, while adding compounding latency and cost.

---

[[06-practice-exercises]]
