# Relevance Metrics

Faithfulness asks "did the model stick to the context?" Relevance asks "did the model actually answer the question?" These are orthogonal — a perfectly faithful answer can still be completely irrelevant.

## Learning objectives

- Measure answer relevancy using embedding-based and LLM-based approaches
- Evaluate context precision and context recall separately
- Implement query-context relevance scoring for retrieval evaluation
- Build a retrieval eval pipeline with Recall@k and MRR

---

## The relevance landscape

```
Question: "How do I reset my password?"

Answer: "Your password must be at least 8 characters."
  → Faithful (if context says this) but IRRELEVANT (doesn't answer the question)

Answer: "Click 'Forgot Password' on the login screen."
  → Faithful AND RELEVANT

Retrieved context: "Password policies require uppercase letters."
  → Retrieved but IMPRECISE (won't help answer the reset question)
```

Three distinct relevance signals:

| Metric | Question asked | Input needed |
|--------|---------------|--------------|
| **Answer relevancy** | Does the answer respond to the question? | question + answer |
| **Context precision** | Are the retrieved chunks relevant to the question? | question + retrieved chunks + ground truth |
| **Context recall** | Does the retrieved context cover the ground truth answer? | retrieved chunks + ground truth answer |

---

## Answer relevancy

RAGAS computes answer relevancy by generating back-questions from the answer and measuring how similar they are to the original question. You can implement the same idea directly.

```python
import os
import json
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in resp.data])

def generate_back_questions(answer: str, n: int = 3) -> list[str]:
    """Generate questions that the given answer would correctly respond to."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Given this answer, generate {n} different questions that this answer directly responds to.
Return JSON: {{"questions": ["q1", "q2", "q3"]}}

Answer: {answer}
JSON only:"""
        }],
        temperature=0.5,
        max_tokens=200,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)["questions"]

def answer_relevancy_score(question: str, answer: str, n_back: int = 3) -> float:
    """
    Score how well the answer addresses the question.
    Higher = answer is more directly relevant to the question.
    """
    back_questions = generate_back_questions(answer, n=n_back)

    # Embed original question and all back-questions
    all_texts = [question] + back_questions
    embeddings = embed(all_texts)

    q_emb = embeddings[0]
    back_embs = embeddings[1:]

    # Normalize
    q_emb = q_emb / np.linalg.norm(q_emb)
    back_embs = back_embs / np.linalg.norm(back_embs, axis=1, keepdims=True)

    # Average cosine similarity between original and back-questions
    similarities = back_embs @ q_emb
    return float(np.mean(similarities))

# Test
question = "How do I reset my password?"
answer_relevant = "Click 'Forgot Password' on the login screen, enter your email, and follow the reset link."
answer_irrelevant = "Passwords must be at least 8 characters and include a number."

score_rel = answer_relevancy_score(question, answer_relevant)
score_irr = answer_relevancy_score(question, answer_irrelevant)

print(f"Relevant answer:   {score_rel:.3f}")
print(f"Irrelevant answer: {score_irr:.3f}")
# Relevant answer:   0.921
# Irrelevant answer: 0.412
```

---

## Context precision

Context precision measures: of the chunks you retrieved, how many are actually useful for answering the question? Low precision means you're retrieving noise that wastes the context window.

```python
def context_precision_score(
    question: str,
    retrieved_chunks: list[str],
    ground_truth_answer: str
) -> float:
    """
    Score what fraction of retrieved chunks are relevant to answering the question.
    Uses LLM to judge each chunk's relevance.
    """
    if not retrieved_chunks:
        return 0.0

    relevant_count = 0
    for chunk in retrieved_chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Is this context chunk useful for answering the question?
Answer 'yes' or 'no' only.

Question: {question}
Ground truth answer: {ground_truth_answer}
Context chunk: {chunk}"""
            }],
            temperature=0.0,
            max_tokens=5
        )
        verdict = response.choices[0].message.content.strip().lower()
        if verdict.startswith("yes"):
            relevant_count += 1

    return relevant_count / len(retrieved_chunks)

# Test
question = "What is the refund policy?"
ground_truth = "Refunds are accepted within 14 days for unused items."
chunks = [
    "Refunds are processed within 14 days. Items must be unused and in original packaging.",
    "Shipping typically takes 3-5 business days within the continental US.",  # irrelevant
    "To request a refund, contact support@company.com with your order number."
]

precision = context_precision_score(question, chunks, ground_truth)
print(f"Context precision: {precision:.2f}")
# Context precision: 0.67  (2 out of 3 chunks are relevant)
```

---

## Context recall

Context recall measures: does the retrieved context contain the information needed to produce the ground truth answer? Low recall means your retriever missed key documents.

```python
def context_recall_score(
    retrieved_chunks: list[str],
    ground_truth_answer: str
) -> float:
    """
    Score what fraction of the ground truth answer is covered by the retrieved context.
    Decomposes the ground truth into claims, then checks each against context.
    """
    # Step 1: Decompose ground truth into atomic claims
    decomp_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Break this answer into atomic factual claims (one fact each).
Return JSON: {{"claims": ["claim1", "claim2"]}}

Answer: {ground_truth_answer}
JSON only:"""
        }],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"}
    )
    claims = json.loads(decomp_response.choices[0].message.content)["claims"]

    if not claims:
        return 0.0

    context_text = "\n\n".join(retrieved_chunks)

    # Step 2: Check each claim against the full retrieved context
    supported = 0
    for claim in claims:
        check_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Is this claim supported by the context? Answer 'yes' or 'no'.

Context: {context_text}
Claim: {claim}"""
            }],
            temperature=0.0,
            max_tokens=5
        )
        if check_response.choices[0].message.content.strip().lower().startswith("yes"):
            supported += 1

    return supported / len(claims)

# Test
ground_truth = "Refunds are accepted within 14 days. Items must be unused. Original receipt required."
chunks = [
    "Refund requests must be made within 14 days of purchase. Items must be in unused condition.",
    # Missing: "original receipt required" → recall should be < 1.0
]

recall = context_recall_score(chunks, ground_truth)
print(f"Context recall: {recall:.2f}")
# Context recall: 0.67 (2/3 claims covered; receipt requirement missing from chunks)
```

---

## Retrieval evaluation: Recall@k and MRR

For component-level retrieval evaluation, you need labeled relevance pairs — a question and which document IDs are relevant.

```python
from typing import Callable

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """What fraction of relevant documents appear in the top-k results?"""
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(top_k & relevant) / len(relevant)

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """What fraction of the top-k results are relevant?"""
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r in relevant) / k

def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1/rank of the first relevant result. 0 if none found."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def mean_reciprocal_rank(
    queries: list[tuple[list[str], list[str]]]
) -> float:
    """MRR over multiple queries. queries = [(retrieved_ids, relevant_ids), ...]"""
    rr_scores = [reciprocal_rank(ret, rel) for ret, rel in queries]
    return sum(rr_scores) / len(rr_scores)

def evaluate_retriever(
    retriever: Callable[[str], list[str]],
    eval_set: list[dict],
    k: int = 5
) -> dict:
    """
    eval_set: list of {"question": str, "relevant_ids": list[str]}
    retriever: function that takes a question and returns list of doc IDs
    """
    recall_scores = []
    precision_scores = []
    rr_scores = []

    for item in eval_set:
        retrieved = retriever(item["question"])
        relevant = item["relevant_ids"]

        recall_scores.append(recall_at_k(retrieved, relevant, k))
        precision_scores.append(precision_at_k(retrieved, relevant, k))
        rr_scores.append(reciprocal_rank(retrieved, relevant))

    return {
        f"recall@{k}": sum(recall_scores) / len(recall_scores),
        f"precision@{k}": sum(precision_scores) / len(precision_scores),
        "mrr": sum(rr_scores) / len(rr_scores),
        "n": len(eval_set)
    }

# Example eval set
eval_set = [
    {"question": "How do I reset my password?", "relevant_ids": ["faq-001", "faq-002"]},
    {"question": "What is your refund policy?", "relevant_ids": ["policy-005"]},
]

# Mock retriever for illustration
def mock_retriever(question: str) -> list[str]:
    if "password" in question.lower():
        return ["faq-001", "kb-012", "faq-002", "kb-050", "faq-999"]
    return ["policy-005", "kb-020", "faq-100", "kb-030", "faq-200"]

metrics = evaluate_retriever(mock_retriever, eval_set, k=5)
print(metrics)
# {'recall@5': 1.0, 'precision@5': 0.4, 'mrr': 1.0, 'n': 2}
```

---

## Query-context relevance (semantic similarity)

A fast, cheap proxy for relevance: embed both the query and the chunk, compute cosine similarity. Use this as a pre-filter before expensive LLM-based checks.

```python
def semantic_relevance_filter(
    query: str,
    chunks: list[str],
    threshold: float = 0.45
) -> list[tuple[str, float]]:
    """Filter chunks below similarity threshold. Returns (chunk, score) pairs."""
    all_texts = [query] + chunks
    embeddings = embed(all_texts)

    q_emb = embeddings[0] / np.linalg.norm(embeddings[0])
    chunk_embs = embeddings[1:]
    chunk_embs = chunk_embs / np.linalg.norm(chunk_embs, axis=1, keepdims=True)

    similarities = (chunk_embs @ q_emb).tolist()

    return [
        (chunk, score)
        for chunk, score in zip(chunks, similarities)
        if score >= threshold
    ]

# Use as a relevance pre-filter before sending to LLM
query = "How does HNSW improve search speed?"
chunks = [
    "HNSW builds a hierarchical graph where each node connects to its nearest neighbors.",
    "The company's revenue grew 40% year-over-year.",
    "Approximate nearest neighbor search trades recall for speed using graph traversal.",
]

relevant = semantic_relevance_filter(query, chunks, threshold=0.45)
print(f"Relevant chunks: {len(relevant)}/{len(chunks)}")
for chunk, score in relevant:
    print(f"  [{score:.3f}] {chunk[:70]}...")
```

> [!tip] Relevance threshold tuning
> Set your threshold using your eval set: find the similarity score that correctly excludes 90% of irrelevant chunks without dropping relevant ones. Typical values: 0.40–0.55 for `text-embedding-3-small`.

---

[[03-hallucination-and-faithfulness]] | [[05-human-evals]]
