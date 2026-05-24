# Chunking and Retrieval

Chunking decisions determine the ceiling of your retrieval quality — no reranker or query expansion technique can recover information that was lost by poor chunking.

---

## Q1: What chunking strategy would you use for a 200-page technical PDF?

??? "Show answer"
    The right strategy depends on the document structure:

    **Recursive character splitting** is the default starting point. It respects natural boundaries (paragraphs, sentences) before splitting arbitrarily.

    ```python
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(document_text)
    ```

    For a structured technical PDF with headers, **hierarchical chunking** works better: embed the section header with every chunk so context isn't lost when a chunk is retrieved in isolation.

    ```python
    # Prepend section header to every chunk
    chunks_with_context = [f"Section: {section_title}\n\n{chunk}" for chunk in chunks]
    ```

    Key parameters:
    - `chunk_size=800` — balances context per chunk vs retrieval precision (too large = irrelevant sentences retrieved; too small = context fragmented)
    - `chunk_overlap=150` (~15-20%) — ensures sentences at chunk boundaries aren't lost

    Test different chunk sizes with Recall@3 on a sample of 20 questions to find the optimal value for your document set.

---

## Q2: What's the difference between dense and sparse retrieval? When does sparse win?

??? "Show answer"
    **Dense retrieval** (what most RAG systems use): embed query and documents into continuous vector space, find nearest neighbours by cosine similarity. Captures semantic meaning — "car" matches "automobile."

    **Sparse retrieval** (BM25, TF-IDF): represent documents as bags of words weighted by frequency and rarity. Exact token matching — "car" matches "car" only.

    Sparse retrieval wins when:
    - Queries contain specific identifiers: product codes, names, version numbers, error codes (`ERROR_CODE_7842`)
    - The user uses exact terminology from the document
    - The corpus is very small (< 10,000 docs) — BM25 is fast and requires no embedding

    **Hybrid search** combines both: retrieve top-k from dense and top-k from sparse, then merge with Reciprocal Rank Fusion.

    ```python
    def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
        scores = {}
        for rank, doc_id in enumerate(dense_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, doc_id in enumerate(sparse_results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)
    ```

---

## Q3: How does reranking improve retrieval quality and what's the cost?

??? "Show answer"
    Embedding-based retrieval (bi-encoders) computes query and document embeddings independently, then compares them. This is fast but imprecise because the embeddings don't interact.

    **Cross-encoder reranking** processes the query and each candidate chunk together, producing a relevance score with full attention across both. Much more accurate, but O(n) inference per query.

    The typical pattern: retrieve top-20 with the fast bi-encoder, rerank with a cross-encoder, return the top-3.

    ```python
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(query: str, candidates: list[str], top_k: int = 3) -> list[str]:
        pairs = [(query, doc) for doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
    ```

    Cost: ~50–100ms latency per reranking call for 20 candidates. Worth it for quality-sensitive applications; skip it for latency-critical ones.

---

## Q4: What is HyDE and when would you use it?

??? "Show answer"
    **HyDE (Hypothetical Document Embeddings)**: instead of embedding the user's query directly, ask the LLM to generate a hypothetical answer first, then embed that hypothetical answer and use it for retrieval.

    The insight: a question ("What is the return policy?") and an answer ("Returns are accepted within 30 days...") live in different regions of embedding space. The hypothetical answer is closer to the actual document than the raw question is.

    ```python
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def hyde_retrieve(query: str, collection) -> list[str]:
        # Step 1: Generate hypothetical answer
        hyp_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Write a short paragraph that would answer: {query}"}],
        )
        hypothetical_doc = hyp_response.choices[0].message.content

        # Step 2: Embed the hypothetical doc
        embedding = client.embeddings.create(
            input=hypothetical_doc, model="text-embedding-3-small"
        ).data[0].embedding

        # Step 3: Retrieve against real documents
        results = collection.query(query_embeddings=[embedding], n_results=5)
        return results["documents"][0]
    ```

    Use HyDE when queries are short or ambiguous. Avoid it for factual lookups with specific identifiers — the hypothetical answer may steer retrieval away from the right document.

---

## Q5: What is multi-query retrieval and what problem does it solve?

??? "Show answer"
    Multi-query retrieval generates multiple rephrased versions of the query, retrieves candidates for each, then unions the results. It improves recall when a single query phrasing misses relevant chunks.

    ```python
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def multi_query_retrieve(query: str, collection, n_queries: int = 3) -> list[str]:
        # Generate query variants
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Generate {n_queries} different phrasings of this question, one per line:\n{query}"
            }],
        )
        queries = [query] + response.choices[0].message.content.strip().split("\n")

        # Retrieve for each and deduplicate
        seen, all_docs = set(), []
        for q in queries:
            emb = client.embeddings.create(input=q, model="text-embedding-3-small").data[0].embedding
            results = collection.query(query_embeddings=[emb], n_results=5)
            for doc in results["documents"][0]:
                if doc not in seen:
                    seen.add(doc)
                    all_docs.append(doc)

        return all_docs[:10]  # return top candidates for reranking
    ```

    Multi-query adds 1 extra LLM call. Pair it with reranking to select the best chunks from the expanded candidate set.

---

*Previous: [RAG Fundamentals](01-rag-fundamentals.md) | Next: [Vector Databases](03-vector-databases.md)*
