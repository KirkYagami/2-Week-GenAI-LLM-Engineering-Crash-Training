# Retrieval and Augmentation

Retrieval is about finding the right chunks. Augmentation is about assembling them into a prompt the LLM can use effectively. Both are learnable — poor retrieval or poor augmentation kills answer quality even with perfect chunks.

## Learning objectives

- Implement query expansion, MMR, and threshold-based filtering
- Assemble retrieved context into effective prompts
- Design citation-aware prompts that attribute answers to sources
- Handle the case when no relevant context is found

---

## Retrieval strategies

### Basic top-k retrieval

```python
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding)

def top_k_retrieve(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    k: int = 5,
    min_score: float = 0.0
) -> list[tuple[dict, float]]:
    query_emb = embed(query)

    # Cosine similarity
    doc_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_emb)
    scores = (embeddings @ query_emb) / (doc_norms.squeeze() * query_norm + 1e-8)

    # Filter by minimum score threshold
    valid_mask = scores >= min_score
    valid_indices = np.where(valid_mask)[0]

    # Sort by score descending
    sorted_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]][:k]

    return [(chunks[i], float(scores[i])) for i in sorted_indices]
```

---

### Query expansion

The user's query is often incomplete or uses different vocabulary than the documents. Expand it with synonyms or a rephrase before searching.

```python
def expand_query(query: str) -> list[str]:
    """Generate multiple phrasings of a query to improve recall."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Generate 3 alternative phrasings of this search query that use different words but mean the same thing. Return only the alternatives, one per line, no numbering.

Query: {query}"""
        }],
        max_tokens=150,
        temperature=0.5
    )

    alternatives = response.choices[0].message.content.strip().split("\n")
    return [query] + [q.strip() for q in alternatives if q.strip()]

def multi_query_retrieve(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    k: int = 5
) -> list[tuple[dict, float]]:
    """Retrieve with multiple query phrasings, deduplicate, return top-k."""
    queries = expand_query(query)
    print(f"Expanded to {len(queries)} queries")

    all_results: dict[str, tuple[dict, float]] = {}

    for q in queries:
        results = top_k_retrieve(q, chunks, embeddings, k=k)
        for chunk, score in results:
            chunk_id = chunk["id"]
            if chunk_id not in all_results or all_results[chunk_id][1] < score:
                all_results[chunk_id] = (chunk, score)

    # Return top-k by max score across all queries
    return sorted(all_results.values(), key=lambda x: x[1], reverse=True)[:k]
```

---

### Maximal Marginal Relevance (MMR)

MMR balances relevance with diversity — it avoids returning 5 near-duplicate chunks that all say the same thing.

```python
def mmr_retrieve(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    k: int = 5,
    lambda_param: float = 0.5  # 0 = full diversity, 1 = full relevance
) -> list[tuple[dict, float]]:
    """Maximal Marginal Relevance: balance relevance vs. diversity."""
    query_emb = embed(query)

    # Normalize all embeddings
    doc_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    q_norm = np.linalg.norm(query_emb)
    norm_embs = embeddings / (doc_norms + 1e-8)
    norm_query = query_emb / (q_norm + 1e-8)

    # Relevance scores
    relevance = norm_embs @ norm_query

    selected = []
    remaining = list(range(len(chunks)))

    for _ in range(min(k, len(chunks))):
        if not selected:
            # Pick most relevant first
            best = remaining[np.argmax(relevance[remaining])]
        else:
            selected_embs = norm_embs[selected]
            mmr_scores = []
            for idx in remaining:
                rel = relevance[idx]
                # Max similarity to already-selected chunks
                max_sim = np.max(norm_embs[idx] @ selected_embs.T)
                mmr = lambda_param * rel - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)
            best = remaining[np.argmax(mmr_scores)]

        selected.append(best)
        remaining.remove(best)

    return [(chunks[i], float(relevance[i])) for i in selected]
```

> [!tip] When to use MMR
> Use MMR when documents in your corpus have high redundancy — e.g., a company has 50 slightly different versions of the same policy doc. MMR ensures the retrieved context covers different aspects of the question rather than repeating the same fact 5 times.

---

## Context assembly

Retrieved chunks must be assembled into a prompt that the LLM can parse unambiguously.

```python
def assemble_context(
    results: list[tuple[dict, float]],
    max_context_tokens: int = 3000,
    include_scores: bool = False
) -> tuple[str, list[str]]:
    """
    Assemble retrieved chunks into a context string.
    Returns (context_string, list_of_sources).
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    context_parts = []
    sources = []
    total_tokens = 0

    for chunk, score in results:
        chunk_text = chunk["text"]
        source = chunk.get("source", "unknown")
        chunk_tokens = len(enc.encode(chunk_text))

        if total_tokens + chunk_tokens > max_context_tokens:
            break

        score_annotation = f" [relevance: {score:.2f}]" if include_scores else ""
        context_parts.append(f"[Source: {source}{score_annotation}]\n{chunk_text}")
        sources.append(source)
        total_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts), list(dict.fromkeys(sources))  # deduplicated

# Example
sample_results = [
    ({"id": "hr-1", "text": "All employees receive 20 days of paid leave annually.", "source": "hr-policy.pdf"}, 0.92),
    ({"id": "hr-2", "text": "Leave requests must be submitted via the HR portal with 2 weeks notice.", "source": "hr-policy.pdf"}, 0.85),
    ({"id": "eng-1", "text": "Engineering teams can take comp time for weekend on-call shifts.", "source": "eng-policy.pdf"}, 0.71),
]

context, sources = assemble_context(sample_results, include_scores=True)
print(f"Context ({len(context)} chars) from sources: {sources}")
print(context[:300])
```

---

## Prompt design for RAG

The system prompt determines how faithfully the model sticks to the retrieved context.

```python
from anthropic import Anthropic

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided documents.

Rules:
1. Use ONLY information from the provided context to answer questions.
2. If the answer is not in the context, respond with: "I don't have information about that in the available documents."
3. Always cite your sources using the format [Source: filename].
4. If information from multiple sources is relevant, synthesize them clearly.
5. Do not add information from your general knowledge — only use the provided context.
6. If the question is ambiguous, answer the most likely interpretation and note the ambiguity."""

def rag_generate(
    question: str,
    context: str,
    sources: list[str],
    model: str = "claude-sonnet-4-6"
) -> dict:
    user_message = f"""<context>
{context}
</context>

<question>
{question}
</question>

Answer the question based only on the context above. Cite specific sources."""

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=500,
        system=RAG_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    return {
        "answer": response.content[0].text,
        "sources_retrieved": sources,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000
    }
```

---

## Handling no-context cases

Your retrieval may legitimately return nothing relevant. Handle this explicitly.

```python
def safe_rag_answer(
    question: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    k: int = 4,
    min_score: float = 0.55  # minimum relevance threshold
) -> dict:
    results = top_k_retrieve(question, chunks, embeddings, k=k, min_score=min_score)

    if not results:
        return {
            "answer": "I don't have relevant information in the knowledge base to answer this question.",
            "sources": [],
            "retrieved_count": 0,
            "fallback": True
        }

    context, sources = assemble_context(results)
    result = rag_generate(question, context, sources)
    result["retrieved_count"] = len(results)
    result["fallback"] = False
    return result
```

> [!warning] Don't skip the minimum score threshold
> Without a minimum score, your retrieval will always return k results — even when the best match has a score of 0.3 and is completely unrelated to the question. The model will then try to answer from irrelevant context and confabulate. Always set a minimum score and test it on out-of-domain queries.

---

## Self-RAG: verify before answering

A simple self-verification loop that checks if retrieved context actually supports the answer.

```python
def self_rag_answer(question: str, context: str) -> dict:
    # Step 1: Check if context is relevant before generating
    relevance_check = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Does this context contain information needed to answer the question?\n\nContext: {context[:500]}\n\nQuestion: {question}\n\nAnswer with YES or NO only."
        }],
        max_tokens=5,
        temperature=0.0
    ).choices[0].message.content.strip().upper()

    if "NO" in relevance_check:
        return {"answer": "No relevant information found.", "verified": False}

    # Step 2: Generate answer
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=400,
        temperature=0.0
    )
    answer = response.choices[0].message.content

    # Step 3: Verify answer is grounded
    grounding_check = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Is this answer supported by the context? Answer YES or NO only.\n\nContext: {context[:500]}\n\nAnswer: {answer[:300]}"
        }],
        max_tokens=5,
        temperature=0.0
    ).choices[0].message.content.strip().upper()

    return {
        "answer": answer,
        "verified": "YES" in grounding_check,
        "relevance_check": relevance_check,
        "grounding_check": grounding_check
    }
```

---

[[02-chunking-strategies]] | [[04-rag-pipeline-end-to-end]]
