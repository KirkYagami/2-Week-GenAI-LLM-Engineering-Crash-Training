# HyDE — Hypothetical Document Embeddings

Standard RAG embeds the user's question and searches for documents whose embeddings are similar. But questions and answers live in different parts of the embedding space — "How do I reset my password?" doesn't look like "Click Forgot Password on the login page." HyDE bridges this gap.

## Learning objectives

- Understand why query-document embedding mismatch hurts RAG recall
- Implement HyDE: generate a hypothetical answer, embed it, retrieve with it
- Measure recall improvement from HyDE on your corpus
- Know when HyDE helps and when it makes things worse

---

## The embedding gap problem

```
User query:  "How do I reset my password?"
Embedding:   [0.12, -0.34, 0.89, ...]  ← "question" region of embedding space

Ideal document: "Click 'Forgot Password' on the login screen, enter your email..."
Embedding:   [0.87, -0.11, 0.23, ...]  ← "answer" region of embedding space

Cosine similarity between these: 0.41 (low!)
```

Questions and answers have different linguistic structures. Bi-encoders trained on passage retrieval handle this better than generic sentence embeddings, but a gap often remains.

**HyDE solution:** Generate a hypothetical answer → embed the hypothetical answer → use that embedding to search. Now you're searching in the "answer" part of the embedding space.

---

## HyDE implementation

```python
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HYDE_PROMPT = """Write a short paragraph (2-4 sentences) that directly answers this question.
This is a hypothetical answer for retrieval purposes — it may not be accurate.
Focus on the structure and vocabulary of a real answer, not factual correctness.

Question: {question}

Hypothetical answer:"""

def generate_hypothetical_document(question: str, n: int = 1) -> list[str]:
    """Generate n hypothetical answers for the given question."""
    hypotheticals = []
    for _ in range(n):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
            temperature=0.7,  # Some variation helps cover different phrasings
            max_tokens=150
        )
        hypotheticals.append(response.choices[0].message.content.strip())
    return hypotheticals

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([item.embedding for item in resp.data])

def hyde_retrieve(
    question: str,
    corpus: list[str],
    k: int = 5,
    n_hypotheticals: int = 3
) -> list[tuple[str, float]]:
    """
    HyDE retrieval:
    1. Generate n hypothetical answers
    2. Embed each hypothetical
    3. Average the embeddings
    4. Use the averaged embedding to search the corpus
    """
    hypotheticals = generate_hypothetical_document(question, n=n_hypotheticals)
    print(f"Generated {len(hypotheticals)} hypothetical(s):")
    for h in hypotheticals:
        print(f"  → {h[:80]}...")

    # Embed hypotheticals and corpus
    hyp_embs = embed(hypotheticals)  # (n, dim)
    hyp_emb_mean = hyp_embs.mean(axis=0)  # Average across hypotheticals
    hyp_emb_mean /= np.linalg.norm(hyp_emb_mean)

    corp_embs = embed(corpus)
    corp_embs /= np.linalg.norm(corp_embs, axis=1, keepdims=True)

    # Retrieve using the hypothetical embedding
    scores = (corp_embs @ hyp_emb_mean).tolist()
    ranked = sorted(zip(corpus, scores), key=lambda x: -x[1])
    return ranked[:k]

def standard_retrieve(question: str, corpus: list[str], k: int = 5) -> list[tuple[str, float]]:
    """Standard embedding retrieval for comparison."""
    q_emb = embed([question])[0]
    q_emb /= np.linalg.norm(q_emb)
    c_embs = embed(corpus)
    c_embs /= np.linalg.norm(c_embs, axis=1, keepdims=True)
    scores = (c_embs @ q_emb).tolist()
    return sorted(zip(corpus, scores), key=lambda x: -x[1])[:k]

# Test corpus — technical documentation style
DOCS = [
    "To reset your account password: navigate to the login page, click 'Forgot Password', enter your registered email address, and follow the reset link sent to your inbox.",
    "Password requirements: minimum 8 characters, at least one uppercase letter, one number, and one special character.",
    "Your account will be locked after 5 failed login attempts. Contact support to unlock it.",
    "Two-factor authentication adds an extra layer of security. Enable it in Settings > Security.",
    "The refund policy allows returns within 30 days for unused items in original packaging.",
    "API rate limits: 1000 requests per minute for standard tier, 10000 for enterprise.",
]

question = "I forgot my password, what steps do I take?"

print("=== Standard Retrieval ===")
standard = standard_retrieve(question, DOCS)
for text, score in standard[:3]:
    print(f"  [{score:.3f}] {text[:80]}...")

print("\n=== HyDE Retrieval ===")
hyde = hyde_retrieve(question, DOCS, k=3, n_hypotheticals=2)
print("\nResults:")
for text, score in hyde:
    print(f"  [{score:.3f}] {text[:80]}...")
```

---

## When HyDE helps

```python
USE_HYDE = {
    "good_fit": [
        "Short factual questions (What, How, When)",
        "User queries phrased differently from document language",
        "Technical questions where answer vocabulary differs from question vocabulary",
        "Low-resource domains where query-document overlap is sparse",
    ],
    "poor_fit": [
        "Ambiguous questions (multiple valid answers → hypothetical may mislead)",
        "Highly technical/niche domains (hallucinated hypothetical may have wrong vocabulary)",
        "Questions with no clear answer format",
        "When retrieval recall is already high (HyDE adds cost without benefit)",
    ]
}
```

> [!warning] HyDE can retrieve wrong answers
> If the LLM's hypothetical answer is factually wrong, the embedding may retrieve documents that support the wrong answer rather than the right one. HyDE is best for improving recall on factual, well-structured question types — not for ambiguous or highly technical queries where the model's hypothetical will be unreliable.

---

## HyDE with multiple hypotheticals

Using multiple hypotheticals and averaging their embeddings improves robustness.

```python
def hyde_multi(question: str, corpus: list[str], k: int = 5) -> list[tuple[str, float]]:
    """
    Generate 3 hypotheticals covering different angles of the question.
    Average their embeddings for a more robust search vector.
    """
    prompts = [
        f"Answer this question briefly: {question}",
        f"Explain step by step: {question}",
        f"Give a technical answer to: {question}",
    ]

    hypotheticals = []
    for prompt in prompts:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=100
        )
        hypotheticals.append(resp.choices[0].message.content.strip())

    # Embed and average
    hyp_embs = embed(hypotheticals)
    avg_emb = hyp_embs.mean(axis=0)
    avg_emb /= np.linalg.norm(avg_emb)

    corp_embs = embed(corpus)
    corp_embs /= np.linalg.norm(corp_embs, axis=1, keepdims=True)

    scores = (corp_embs @ avg_emb).tolist()
    return sorted(zip(corpus, scores), key=lambda x: -x[1])[:k]
```

---

[[01-reranking]] | [[03-multi-query]]
