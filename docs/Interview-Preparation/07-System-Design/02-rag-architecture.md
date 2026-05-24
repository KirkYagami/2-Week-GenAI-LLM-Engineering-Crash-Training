# RAG Architecture

RAG system design is the most common system design question in LLM engineering interviews. Interviewers want a specific, defensible architecture — not just "embed documents and retrieve."

---

## Q1: Design a customer support bot backed by a 10,000-page documentation corpus.

??? "Show answer"
    **Clarifying questions to ask first**:
    - What's the expected query volume? (determines caching strategy)
    - Does the documentation change frequently? (determines update strategy)
    - What's the latency requirement? (determines streaming, model choice)
    - What's the acceptable hallucination rate? (determines faithfulness threshold)

    **Architecture**:

    ```
    Ingestion pipeline (offline):
    PDF/HTML docs → chunk (800 tokens, 150 overlap) → embed (text-embedding-3-small)
                 → store in ChromaDB with metadata (section, last_updated, doc_type)

    Query pipeline (online):
    User query
      → Query analysis (gpt-4o-mini): classify intent, extract keywords
      → Embed query + HyDE hypothetical doc
      → Retrieve top-20 from ChromaDB (dense + sparse hybrid)
      → Rerank top-5 with CrossEncoder
      → Assemble context + system prompt
      → Generate answer (gpt-4o) with streaming SSE
      → Faithfulness check: if score < 0.75, append disclaimer
    ```

    **Key design decisions**:
    - `gpt-4o-mini` for query analysis (cheap, fast) + `gpt-4o` for generation (quality where it matters)
    - Hybrid search (dense + BM25) because documentation has exact product names and codes
    - Streaming so users see output immediately despite 2–3s generation time
    - Metadata filter by `doc_type` lets you scope to "release notes" or "API reference" based on intent classification

    **What changes at scale** (1M+ daily queries):
    - Semantic cache for common questions (expected 30–40% hit rate on support queries)
    - Pinecone replaces ChromaDB (managed, multi-region)
    - Background ingestion pipeline (Celery + S3) for document updates

---

## Q2: How would you scale a RAG system to handle 10,000 queries per day?

??? "Show answer"
    10,000 queries/day = ~7 queries/minute average, ~50 queries/minute at peak. This is a modest scale — the focus is on cost and reliability, not raw throughput.

    **Cost at 10k queries/day without optimization**:
    - Average query: 500 input tokens (context + system prompt) + 200 output tokens
    - `gpt-4o`: 10,000 × (500/1M × $2.50 + 200/1M × $10.00) = $33/day

    **Cost with optimization**:
    - Semantic cache (30% hit rate): reduces LLM calls to 7,000 → saves $10/day
    - `gpt-4o-mini` for simple queries (50% of traffic): saves another $14/day
    - Total: ~$9/day vs $33/day without optimization

    **Reliability design**:
    - Deploy on Fly.io with 2 machines minimum (avoid cold starts)
    - ChromaDB on a persistent volume for the vector index
    - Rate limiting at the API gateway (sliding window per user)
    - Health check endpoint that queries ChromaDB and returns 503 if retrieval fails

---

## Q3: How do you handle multi-tenant RAG where different users have access to different documents?

??? "Show answer"
    Never mix documents from different tenants in a shared collection without isolation — retrieval can leak data across tenant boundaries.

    **Approach 1 — Metadata filtering** (simpler, lower cost):
    Store all documents in one collection with a `tenant_id` metadata field. Always filter by `tenant_id` at query time.

    ```python
    def tenant_rag(query: str, tenant_id: str) -> str:
        results = collection.query(
            query_embeddings=[embed(query)],
            n_results=5,
            where={"tenant_id": {"$eq": tenant_id}},  # enforced server-side
        )
        return generate_answer(query, results["documents"][0])
    ```

    Risk: a bug in the filter logic leaks cross-tenant data. Mitigate with integration tests that assert cross-tenant isolation.

    **Approach 2 — Separate collections per tenant** (stronger isolation, higher operational cost):
    Each tenant gets their own ChromaDB collection or Pinecone namespace. No metadata filtering needed — isolation is structural.

    ```python
    def get_tenant_collection(tenant_id: str):
        return chroma.get_or_create_collection(f"tenant_{tenant_id}")
    ```

    Use separate collections for tenants with strict data isolation requirements (healthcare, finance). Use metadata filtering for general SaaS products.

---

## Q4: How would you add evaluation to a production RAG pipeline?

??? "Show answer"
    Evaluation in production is different from offline evaluation — you need to measure quality continuously without ground truth labels for every query.

    **Offline evaluation** (run before deployment):

    ```python
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from datasets import Dataset

    test_set = Dataset.from_dict({
        "question": test_questions,
        "answer": [rag_answer(q) for q in test_questions],
        "contexts": [retrieve(q) for q in test_questions],
        "ground_truth": expected_answers,
    })

    result = evaluate(test_set, metrics=[faithfulness, answer_relevancy, context_recall])
    # Block deployment if faithfulness < 0.80
    ```

    **Online evaluation** (continuous monitoring):
    - **Implicit feedback**: track thumbs up/down, follow-up questions, escalations to human agents
    - **LLM-as-judge** on a 1% sample: run faithfulness scoring on sampled responses asynchronously
    - **Anomaly detection**: alert if average faithfulness drops below the baseline by 10%

    ```python
    import random

    async def rag_with_monitoring(query: str) -> str:
        answer = await rag_answer(query)

        if random.random() < 0.01:  # 1% sample
            asyncio.create_task(
                evaluate_and_log(query, answer)  # async, non-blocking
            )

        return answer
    ```

---

## Q5: How do you handle retrieval when the user's question requires combining information from multiple documents?

??? "Show answer"
    Single-document retrieval fails for multi-hop questions like "How does product X's billing policy compare to product Y's?"

    **Strategies**:

    **1. Multi-query retrieval**: decompose the question into sub-queries, retrieve for each, merge results.

    ```python
    import os
    from openai import AsyncOpenAI

    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def decompose_and_retrieve(question: str) -> list[str]:
        decomp = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Break this question into 2-3 simpler sub-questions, one per line:\n{question}"}],
        )
        sub_questions = decomp.choices[0].message.content.strip().split("\n")

        import asyncio
        all_chunks = await asyncio.gather(*[retrieve(q) for q in sub_questions])
        return deduplicate([chunk for chunks in all_chunks for chunk in chunks])
    ```

    **2. Iterative retrieval (agent-style)**: retrieve, generate a partial answer, identify gaps, retrieve again. Use this for complex research questions.

    **3. Increase `n_results`**: retrieve top-20 instead of top-5, rerank to select the most diverse and relevant chunks.

    Multi-hop questions are one of the hardest RAG failure modes. State this explicitly in an interview and describe which technique you'd apply based on the query type — interviewers reward knowing the limits of your design.

---

*Previous: [LLM System Design](01-llm-system-design.md) | Next: [Agent System Design](03-agent-system-design.md)*
