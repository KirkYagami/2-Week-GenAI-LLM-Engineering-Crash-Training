# Interview Questions — RAG Basics

---

## Q1: Explain the RAG architecture. What problem does it solve that prompting alone cannot?

??? "Show answer"
    RAG (Retrieval-Augmented Generation) is a three-stage pipeline: index (chunk and embed documents into a vector store), retrieve (find the most relevant chunks for a query), and generate (use an LLM to produce an answer grounded in the retrieved context).

    **The problem it solves:** LLMs have fixed parametric knowledge frozen at training time. They cannot know about your internal documents, post-training events, or proprietary data. Without RAG, the model either hallucinates an answer or says "I don't know."

    With RAG, the relevant text is literally in the prompt — the model's task shifts from *remembering* to *extracting and synthesizing*. This is why RAG dramatically reduces hallucination for knowledge-intensive tasks: the model doesn't need to recall a fact, it just needs to read it from the context.

    **What prompting alone cannot solve:** You can't fit an entire knowledge base in the context window. Even with 1M-token context, querying 100K documents each time is economically prohibitive. RAG solves the retrieval problem before the context window problem.

---

## Q2: What are the most important hyperparameters in a RAG system and how do you tune them?

??? "Show answer"
    The five most impactful hyperparameters, in order of impact:

    **1. Chunk size (300–1500 tokens):** Too small = not enough context per chunk, retrieval is precise but generation lacks context. Too large = relevant information is diluted among irrelevant content. Benchmark at 256, 512, and 1024 tokens on your specific query distribution.

    **2. k (number of retrieved chunks, 3–10):** More chunks = more context = potentially better answers, but also more irrelevant content that confuses the model. Diminishing returns after k=5 for most use cases.

    **3. Chunk overlap (10–20% of chunk size):** Prevents important sentences from being cut across chunk boundaries. 64 tokens on a 512-token chunk is typical.

    **4. Minimum relevance score threshold (0.5–0.75):** Below this score, don't retrieve — return "no information found" instead. Prevents the model from hallucinating based on irrelevant context.

    **5. Embedding model:** The choice of embedding model affects retrieval quality more than any other single factor. Benchmark `text-embedding-3-small` vs. `BAAI/bge-large` on a sample of your actual queries before committing.

    **Tuning process:** Build an evaluation set of 50–200 (query, expected_source) pairs. Measure Recall@k. Tune one hyperparameter at a time.

---

## Q3: What is the difference between dense retrieval and sparse retrieval (BM25)?

??? "Show answer"
    **Dense retrieval (embedding-based):**
    - Represents documents and queries as dense continuous vectors (e.g., 1536 float32s)
    - Similarity = cosine similarity between vectors
    - Understands semantic meaning: "machine learning" matches "ML" matches "statistical modeling"
    - Fails at exact term matching: "GPT-4o-2024-11-20" may not match the exact version string

    **Sparse retrieval (BM25, TF-IDF):**
    - Represents documents as sparse vectors of term frequencies
    - Similarity = weighted term overlap between query and document
    - Excellent at exact term matching: product codes, names, IDs
    - Fails at synonyms and paraphrases

    **Hybrid retrieval (production standard):**
    Run both in parallel. Merge results with Reciprocal Rank Fusion (RRF):
    ```
    RRF score(d) = Σ 1 / (k + rank_i(d))
    ```
    Where the sum is over both retrieval systems. RRF outperforms either system alone on most BEIR benchmarks by 5–15%.

    Use hybrid when: users mix natural language queries with specific IDs or codes (e.g., "what are the symptoms of condition J45.20?").

---

## Q4: How do you handle a RAG system that returns wrong answers?

??? "Show answer"
    Wrong answers in RAG come from three root causes, each with different fixes:

    **Retrieval failure (wrong chunks retrieved):**
    - Symptom: the correct chunk exists but isn't in the top-k
    - Fix: improve chunking (smaller chunks, different strategy), switch embedding model, add reranking, use query expansion, lower the min_score threshold

    **Augmentation failure (right chunks, but poorly assembled into the prompt):**
    - Symptom: the correct answer is in the retrieved text but the model misses it
    - Fix: put the most relevant chunk first (lost-in-the-middle), reduce k, improve the system prompt, use XML tags to separate sources

    **Generation failure (right context, wrong answer):**
    - Symptom: retrieved context contains the answer, model ignores it
    - Fix: strengthen the "use only provided context" instruction, use a more capable model, add self-consistency (ask 3 times, take majority), use a reranking step

    **Debugging process:** Log retrieved chunks for every failed query. Check the retrieved text first — if the right answer is there and the model got it wrong, it's a generation problem. If the right answer is not in the retrieved text, it's a retrieval problem.

---

## Q5: What is MMR (Maximal Marginal Relevance) and when should you use it?

??? "Show answer"
    MMR is a retrieval algorithm that balances relevance and diversity. Standard top-k retrieval returns the k most similar chunks to the query — but if your corpus has many near-duplicate documents (10 versions of the same policy), you may retrieve 5 almost-identical chunks.

    MMR alternates between: (1) picking the most relevant remaining chunk, weighted by (2) how different it is from chunks already selected.

    ```
    MMR(d) = λ × relevance(d, query) - (1-λ) × max_similarity(d, selected)
    ```

    - λ = 1: pure relevance (same as top-k)
    - λ = 0: pure diversity (ignore relevance)
    - λ = 0.5: balanced (typical default)

    **When to use MMR:**
    - Document corpus has high redundancy (multiple versions of similar content)
    - Multi-aspect questions where you want the answer to cover different angles
    - FAQ retrieval where similar questions have identical answers

    **When NOT to use:**
    - Questions with a single definitive answer (exact fact lookup)
    - Small corpora where redundancy isn't an issue
    - When retrieval latency matters (MMR adds O(k²) comparisons)

---

## Q6: How do you evaluate a RAG system's answer quality (not just retrieval quality)?

??? "Show answer"
    Answer quality evaluation requires going beyond retrieval metrics to assess the final generated answer. Three dimensions matter:

    **1. Faithfulness:** Does the answer only use information from the retrieved context? A faithful answer never introduces facts not present in the context.
    - Measure: LLM-as-judge with a rubric (1–5 scale), or RAGAS faithfulness metric
    - Threshold: faithfulness > 0.8 before shipping to production

    **2. Answer relevance:** Does the answer actually respond to the question asked? High faithfulness + low relevance = the model cited the documents but didn't answer the question.
    - Measure: LLM-as-judge, or human labelers rating 1–5

    **3. Context precision / recall:** Of the retrieved chunks, how many were actually needed? (Precision.) Of all chunks that would have been helpful, how many were retrieved? (Recall.)

    **RAGAS framework (automated evaluation):**
    ```python
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall

    result = evaluate(
        dataset=eval_dataset,  # HuggingFace Dataset with question/answer/contexts/ground_truth
        metrics=[faithfulness, answer_relevancy, context_recall]
    )
    ```

    **Practical minimum:** Build a labeled test set of 50 (question, ground_truth_answer) pairs. Run the RAG pipeline on each, get automated scores, and do human review on the bottom 10% of scorers.

---

## Q7: What is the "lost in the middle" problem and how do you address it in RAG?

??? "Show answer"
    Research (Liu et al., 2023) showed that LLMs perform better on information at the beginning and end of long contexts compared to the middle. When you concatenate k retrieved chunks into a prompt, chunks in the middle receive less attention.

    **In RAG:** If you retrieve 6 chunks and the most relevant one happens to be at position 4 (middle), the model is more likely to miss it or synthesize it incorrectly.

    **Mitigations:**

    1. **Reduce k:** Use k=3–4 rather than k=10. Less context, less middle.

    2. **Sort by relevance with the best chunk first:** `chunks.sort(key=lambda x: x['score'], reverse=True)` — put the highest-scored chunk at position 0.

    3. **Reverse sort (best first and last):** Place highest-scored chunk first, second-highest last, then fill middle with remaining. This ensures the two strongest signals are at attention-friendly positions.

    4. **Reranking:** Use a cross-encoder to rerank the top-k and pick the top-3 most relevant — a smaller, higher-quality context.

    5. **Hierarchical prompting:** Split into two calls — first identify the most relevant chunk, then generate from just that chunk.

    The effect is less pronounced in Claude Sonnet 4.6 and GPT-4o compared to earlier models, but still measurable on very long contexts.

---

## Q8: Design a RAG system for a 10,000-page technical documentation corpus. What are your key architectural decisions?

??? "Show answer"
    Key decisions for a large-scale RAG system:

    **Chunking:** Use recursive character splitting with 512 tokens and 64-token overlap. For structured docs (API references), use heading-based chunking. Pre-compute all chunks offline.

    **Embedding:** `text-embedding-3-small` for cost efficiency, or `BAAI/bge-large` locally if privacy is required. Batch embed all 10,000 pages once; re-embed only on document updates. At ~150 tokens/chunk average and 2 chunks/page, that's ~30,000 chunks × $0.02/1M = $0.0006 — nearly free.

    **Index:** HNSW index (via Qdrant or Pinecone) for low-latency production serving. FAISS IndexHNSWFlat if self-hosted. Set `M=32, ef=200` for good accuracy.

    **Retrieval:** Hybrid search (dense + BM25) with RRF merging. Technical docs often contain product names, version numbers, and function names where exact matching matters.

    **Generation model:** Claude Sonnet 4.6 with a strict grounding prompt. Log all retrieved chunks for quality monitoring.

    **Update strategy:** Webhook → embedding queue → upsert to vector store. For Qdrant, native vector deletion handles document updates. For FAISS, weekly full rebuild.

    **Evaluation:** Instrument every query. Weekly automated RAGAS evaluation on a 100-query golden test set. Alert if faithfulness drops below 0.75.

---

[[06-practice-exercises]] | [[../Day-03-Part-2-Vector-Databases/00-agenda]]
