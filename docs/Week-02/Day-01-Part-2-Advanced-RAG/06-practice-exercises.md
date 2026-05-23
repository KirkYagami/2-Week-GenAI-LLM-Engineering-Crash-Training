# Practice Exercises — Advanced RAG

---

## Exercise 1 — Reranking comparison (Warm-up)

Compare retrieval quality with and without cross-encoder reranking on a small labeled eval set. Report Precision@3 for both.

```python
import os
import numpy as np
from openai import OpenAI
from sentence_transformers import CrossEncoder

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

CORPUS = [
    {"id": 0, "text": "To reset your password, click 'Forgot Password' on the login page and check your email."},
    {"id": 1, "text": "Password requirements: minimum 8 characters including uppercase and a number."},
    {"id": 2, "text": "Your account will be locked after 5 failed attempts. Contact support@company.com to unlock."},
    {"id": 3, "text": "Two-factor authentication is available in Settings > Security > Enable 2FA."},
    {"id": 4, "text": "For billing issues, visit Settings > Billing or email billing@company.com."},
    {"id": 5, "text": "If the reset email doesn't arrive within 5 minutes, check your spam folder or try again."},
    {"id": 6, "text": "You can change your password in Settings > Account > Change Password when logged in."},
    {"id": 7, "text": "Our support team is available 24/7 at support@company.com or 1-800-HELP."},
]

EVAL_SET = [
    {"query": "How do I reset my forgotten password?", "relevant_ids": [0, 5]},
    {"query": "My account is locked after wrong password attempts", "relevant_ids": [2]},
    {"query": "How do I change my current password?", "relevant_ids": [6]},
]

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embs = np.array([item.embedding for item in resp.data])
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)

# Pre-embed corpus
corpus_texts = [d["text"] for d in CORPUS]
corpus_ids = [d["id"] for d in CORPUS]
corp_embs = embed(corpus_texts)

def bi_encoder_rank(query: str, k: int = 5) -> list[int]:
    q_emb = embed([query])[0]
    scores = corp_embs @ q_emb
    return [corpus_ids[i] for i in np.argsort(scores)[::-1][:k]]

def reranked_rank(query: str, k_initial: int = 7, k_final: int = 3) -> list[int]:
    q_emb = embed([query])[0]
    scores = corp_embs @ q_emb
    top_initial = np.argsort(scores)[::-1][:k_initial]
    candidates = [(corpus_ids[i], corpus_texts[i]) for i in top_initial]
    pairs = [[query, text] for _, text in candidates]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, rerank_scores.tolist()), key=lambda x: -x[1])
    return [doc_id for (doc_id, _), _ in reranked[:k_final]]

def precision_at_k(ranked_ids: list[int], relevant_ids: list[int], k: int = 3) -> float:
    top_k = set(ranked_ids[:k])
    return len(top_k & set(relevant_ids)) / k

# Evaluate
print(f"{'Query':<45} {'BiEnc P@3':>10} {'Rerank P@3':>12}")
print("-" * 70)

bi_scores, rerank_scores = [], []
for item in EVAL_SET:
    bi_rank = bi_encoder_rank(item["query"], k=5)
    re_rank = reranked_rank(item["query"])
    bi_p = precision_at_k(bi_rank, item["relevant_ids"])
    re_p = precision_at_k(re_rank, item["relevant_ids"])
    bi_scores.append(bi_p)
    rerank_scores.append(re_p)
    print(f"{item['query'][:45]:<45} {bi_p:>10.2f} {re_p:>12.2f}")

print("-" * 70)
print(f"{'Average':<45} {sum(bi_scores)/len(bi_scores):>10.2f} {sum(rerank_scores)/len(rerank_scores):>12.2f}")
improvement = (sum(rerank_scores) - sum(bi_scores)) / len(bi_scores)
print(f"\nReranking improvement: {improvement:+.2f} Precision@3")
```

---

## Exercise 2 — Multi-technique RAG pipeline (Main)

Build an advanced RAG pipeline that combines multi-query + reranking + compression, and compare it against basic single-query retrieval using a RAGAS-inspired eval.

```python
import os
import json
import numpy as np
from openai import OpenAI
from sentence_transformers import CrossEncoder

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

KNOWLEDGE_BASE = [
    "Python was created by Guido van Rossum and first released in 1991. It emphasizes readability.",
    "Python 3.0, released in 2008, introduced backward-incompatible changes. Python 2 reached EOL in 2020.",
    "Python is widely used for data science, machine learning, web development, and automation.",
    "The GIL (Global Interpreter Lock) prevents true multithreading in CPython. Use multiprocessing for CPU parallelism.",
    "Type hints were introduced in Python 3.5 via PEP 484. They're optional but improve IDE support.",
    "Python package management uses pip. Virtual environments (venv) isolate project dependencies.",
    "List comprehensions in Python: [x**2 for x in range(10) if x % 2 == 0] creates a list of even squares.",
    "Python's async/await syntax (PEP 492) enables non-blocking I/O in asyncio-based frameworks.",
]

EVAL_QUESTIONS = [
    {
        "question": "When was Python first released and who created it?",
        "ground_truth": "Python was first released in 1991 by Guido van Rossum."
    },
    {
        "question": "How do you achieve parallelism in Python given the GIL?",
        "ground_truth": "Use multiprocessing for CPU parallelism since the GIL prevents true multithreading."
    },
    {
        "question": "What Python feature helps with non-blocking operations?",
        "ground_truth": "Python's async/await syntax enables non-blocking I/O via asyncio."
    },
]

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embs = np.array([item.embedding for item in resp.data])
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)

corp_embs = embed(KNOWLEDGE_BASE)

def basic_retrieve(question: str, k: int = 3) -> list[str]:
    q_emb = embed([question])[0]
    scores = corp_embs @ q_emb
    top = np.argsort(scores)[::-1][:k]
    return [KNOWLEDGE_BASE[i] for i in top]

def advanced_retrieve(question: str) -> list[str]:
    # Multi-query
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Generate 3 search queries for: {question}\nReturn JSON: {{\"queries\": [...]}}"}],
        temperature=0.5, max_tokens=150, response_format={"type": "json_object"}
    )
    variants = [question] + json.loads(resp.choices[0].message.content).get("queries", [])

    # Retrieve for each variant
    doc_scores: dict[int, float] = {}
    for v in variants:
        q_emb = embed([v])[0]
        scores = corp_embs @ q_emb
        for i, s in enumerate(scores):
            doc_scores[i] = max(doc_scores.get(i, 0), float(s))

    top_15 = sorted(doc_scores.keys(), key=lambda i: -doc_scores[i])[:15]
    candidates = [KNOWLEDGE_BASE[i] for i in top_15]

    # Rerank
    pairs = [[question, c] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    reranked = [c for c, _ in sorted(zip(candidates, rerank_scores.tolist()), key=lambda x: -x[1])[:5]]

    return reranked

def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n".join(f"- {c}" for c in context_chunks)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer precisely using only the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.0, max_tokens=150
    )
    return resp.choices[0].message.content

def judge_answer(question: str, answer: str, ground_truth: str) -> float:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""Score 0-1: how well does the answer match the ground truth?
Q: {question}
Ground truth: {ground_truth}
Answer: {answer}
Return JSON: {{"score": 0.0-1.0}}"""}],
        temperature=0.0, max_tokens=50, response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)["score"]

# Compare approaches
print(f"{'Question':<45} {'Basic':>6} {'Advanced':>9}")
print("-" * 65)

basic_scores, advanced_scores = [], []
for item in EVAL_QUESTIONS:
    q = item["question"]

    basic_ctx = basic_retrieve(q)
    basic_ans = generate_answer(q, basic_ctx)
    basic_score = judge_answer(q, basic_ans, item["ground_truth"])

    adv_ctx = advanced_retrieve(q)
    adv_ans = generate_answer(q, adv_ctx)
    adv_score = judge_answer(q, adv_ans, item["ground_truth"])

    basic_scores.append(basic_score)
    advanced_scores.append(adv_score)
    print(f"{q[:45]:<45} {basic_score:>6.2f} {adv_score:>9.2f}")

print("-" * 65)
print(f"{'Average':<45} {sum(basic_scores)/len(basic_scores):>6.2f} {sum(advanced_scores)/len(advanced_scores):>9.2f}")
```

---

[[05-advanced-rag-patterns]] | [[07-interview-questions]]
