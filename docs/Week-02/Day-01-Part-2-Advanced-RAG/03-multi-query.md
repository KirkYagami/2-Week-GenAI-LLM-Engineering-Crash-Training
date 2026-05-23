# Multi-Query Retrieval

A single query embedding may miss relevant documents that use different terminology or address a different aspect of the question. Multi-query retrieval generates multiple rephrased versions of the original query and retrieves candidates for each, then merges the results. It's one of the best recall improvements per unit of complexity.

## Learning objectives

- Generate query variations using an LLM
- Merge multi-query results with deduplication
- Combine multi-query with reranking for a two-stage improvement
- Implement step-back prompting for abstract knowledge retrieval

---

## Why single queries fail

```
User query: "Why does my RAG pipeline return irrelevant results?"

Alternative phrasings the user might have used:
  - "RAG retrieval quality issues"
  - "low context precision in RAG"
  - "vector search returning wrong documents"
  - "how to improve RAG retrieval accuracy"

If your corpus uses "retrieval quality" but the user said "irrelevant results",
embedding similarity may be low — even though the same document answers the question.
```

Multi-query generates several of these alternatives and retrieves for each.

---

## Multi-query retrieval implementation

```python
import os
import json
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QUERY_EXPANSION_PROMPT = """You are an expert at generating search queries.
Given a question, generate {n} different versions that rephrase or decompose it.
Each version should target a different aspect or use different terminology.

Original question: {question}

Return JSON: {{"queries": ["query1", "query2", ...]}}"""

def generate_query_variants(question: str, n: int = 4) -> list[str]:
    """Generate n query variants using GPT-4o-mini."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": QUERY_EXPANSION_PROMPT.format(question=question, n=n)
        }],
        temperature=0.7,
        max_tokens=300,
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    variants = result.get("queries", [])
    return [question] + variants  # Include the original

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([item.embedding for item in resp.data])

def multi_query_retrieve(
    question: str,
    corpus: list[str],
    n_variants: int = 4,
    k_per_query: int = 5
) -> list[tuple[str, float]]:
    """
    Retrieve using multiple query variants and merge results.
    Documents retrieved by more variants rank higher (vote-based scoring).
    """
    # Step 1: Generate query variants
    variants = generate_query_variants(question, n=n_variants)
    print(f"Query variants ({len(variants)}):")
    for v in variants:
        print(f"  → {v}")

    # Step 2: Retrieve candidates for each variant
    corp_embs = embed(corpus)
    corp_embs /= np.linalg.norm(corp_embs, axis=1, keepdims=True)

    # Track which documents were retrieved and their max scores
    doc_votes: dict[int, int] = {}
    doc_max_scores: dict[int, float] = {}

    for variant in variants:
        q_emb = embed([variant])[0]
        q_emb /= np.linalg.norm(q_emb)
        scores = corp_embs @ q_emb

        top_k_indices = np.argsort(scores)[::-1][:k_per_query]
        for idx in top_k_indices:
            doc_votes[idx] = doc_votes.get(idx, 0) + 1
            doc_max_scores[idx] = max(doc_max_scores.get(idx, 0), float(scores[idx]))

    # Step 3: Merge — sort by vote count, break ties with max score
    merged = sorted(
        doc_votes.keys(),
        key=lambda idx: (doc_votes[idx], doc_max_scores[idx]),
        reverse=True
    )

    return [(corpus[idx], doc_max_scores[idx]) for idx in merged[:k_per_query]]

# Test corpus
CORPUS = [
    "Context precision measures how many retrieved chunks are actually relevant to the question.",
    "Low context precision means irrelevant documents are being retrieved. Fix: better embedding model or reranking.",
    "Retrieval quality in RAG is measured by context recall and context precision metrics.",
    "RAGAS framework provides automated evaluation of RAG pipelines including faithfulness and relevancy.",
    "Embedding models affect retrieval quality. BAAI/bge-large-en performs best on MTEB retrieval benchmarks.",
    "Cross-encoder reranking improves context precision by reordering retrieved documents.",
    "Query expansion generates multiple versions of a user question to improve retrieval recall.",
]

question = "Why does my RAG pipeline return irrelevant results?"

print("=" * 60)
results = multi_query_retrieve(question, CORPUS, n_variants=3, k_per_query=4)
print("\nMerged results (votes-based ranking):")
for text, score in results:
    print(f"  [{score:.3f}] {text[:80]}...")
```

---

## Step-back prompting

Step-back prompting retrieves background knowledge before answering a specific question. It's a form of multi-query where the second query is more abstract.

```python
def stepback_retrieve(
    specific_question: str,
    corpus: list[str],
    k: int = 5
) -> tuple[list, list]:
    """
    Generate an abstract 'step-back' question and retrieve for both.
    Returns: (specific_results, abstract_results)
    """
    stepback_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Given this specific question, generate a more general/abstract question
that would retrieve background knowledge useful for answering the specific question.

Specific question: {specific_question}

General question (one sentence):"""
        }],
        temperature=0.3,
        max_tokens=80
    )
    abstract_question = stepback_response.choices[0].message.content.strip()
    print(f"Step-back question: {abstract_question}")

    # Retrieve for both specific and abstract questions
    corp_embs = embed(corpus)
    corp_embs /= np.linalg.norm(corp_embs, axis=1, keepdims=True)

    def retrieve(query: str) -> list[tuple[str, float]]:
        q_emb = embed([query])[0]
        q_emb /= np.linalg.norm(q_emb)
        scores = corp_embs @ q_emb
        top = np.argsort(scores)[::-1][:k]
        return [(corpus[i], float(scores[i])) for i in top]

    specific_results = retrieve(specific_question)
    abstract_results = retrieve(abstract_question)

    return specific_results, abstract_results

# Test
specific_q = "Why is my context precision 0.4 when testing with GPT-4o?"
specific_r, abstract_r = stepback_retrieve(specific_q, CORPUS, k=3)

print("\nSpecific question results:")
for text, score in specific_r[:2]:
    print(f"  [{score:.3f}] {text[:80]}...")

print("\nStep-back question results:")
for text, score in abstract_r[:2]:
    print(f"  [{score:.3f}] {text[:80]}...")
```

> [!tip] Combining multi-query with reranking
> Multi-query increases recall (more relevant documents in the candidate set). Reranking increases precision (most relevant documents rise to the top). They're complementary: run multi-query → collect 30–50 candidates → rerank → take top 5. This pipeline consistently outperforms single-query-no-reranking in experiments.

---

[[02-hyde]] | [[04-contextual-compression]]
