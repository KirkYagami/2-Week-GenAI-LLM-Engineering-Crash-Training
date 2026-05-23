# Reranking

Bi-encoder (embedding) retrieval is fast but approximate. It embeds query and documents independently, which means it misses subtle relevance signals that require reading both together. Reranking fixes this: retrieve a large candidate set cheaply, then use a powerful cross-encoder to reorder by true relevance.

## Learning objectives

- Understand why bi-encoder retrieval needs reranking
- Implement cross-encoder reranking using `sentence-transformers`
- Use Cohere Rerank API for production reranking
- Measure the precision improvement from reranking

---

## Why reranking works

```
Bi-encoder retrieval (fast, approximate):
  Query → embed → cosine similarity → top 20 chunks
  [chunk 1: 0.82] ← ranked first, but not most relevant
  [chunk 7: 0.79] ← buried at rank 7, actually the answer
  [chunk 3: 0.78]

Cross-encoder reranking (slow, precise):
  Query + chunk 1 → read together → relevance score: 0.65  ← drops
  Query + chunk 7 → read together → relevance score: 0.94  ← rises to top
  Query + chunk 3 → read together → relevance score: 0.72
```

Cross-encoders are slower (O(candidates × sequence_length)) but read query and document jointly — catching semantic relevance that embedding cosine similarity misses.

---

## Cross-encoder reranking with sentence-transformers

```python
import os
import numpy as np
from sentence_transformers import CrossEncoder
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load cross-encoder model (~90 MB, fast on CPU)
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
    device="cpu"
)

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([item.embedding for item in resp.data])

def bi_encoder_retrieve(query: str, corpus: list[str], k: int = 20) -> list[tuple[str, float]]:
    """First-stage retrieval using embedding cosine similarity."""
    all_texts = [query] + corpus
    embs = embed(all_texts)
    q_emb = embs[0] / np.linalg.norm(embs[0])
    c_embs = embs[1:] / np.linalg.norm(embs[1:], axis=1, keepdims=True)
    scores = (c_embs @ q_emb).tolist()
    ranked = sorted(zip(corpus, scores), key=lambda x: -x[1])
    return ranked[:k]

def rerank(query: str, candidates: list[tuple[str, float]], top_k: int = 5) -> list[tuple[str, float]]:
    """Second-stage reranking using cross-encoder."""
    texts = [c[0] for c in candidates]
    pairs = [[query, text] for text in texts]

    scores = reranker.predict(pairs)  # Shape: (n_candidates,)

    reranked = sorted(zip(texts, scores.tolist()), key=lambda x: -x[1])
    return reranked[:top_k]

# Test
CORPUS = [
    "Python is a high-level programming language known for readability.",
    "The reset password flow requires clicking 'Forgot Password' on the login page.",
    "To reset your password: go to Settings > Security > Change Password.",
    "Password requirements: minimum 8 characters, one uppercase, one number.",
    "Our refund policy allows returns within 30 days of purchase.",
    "Contact support at support@company.com for account issues.",
    "If you forgot your password, click the 'Forgot Password' link and check your email.",
]

query = "How do I reset my password?"

# First-stage retrieval (top 5 candidates)
candidates = bi_encoder_retrieve(query, CORPUS, k=5)
print("Bi-encoder ranking:")
for i, (text, score) in enumerate(candidates, 1):
    print(f"  {i}. [{score:.3f}] {text[:70]}...")

# Second-stage reranking
reranked = rerank(query, candidates, top_k=3)
print("\nAfter reranking:")
for i, (text, score) in enumerate(reranked, 1):
    print(f"  {i}. [{score:.3f}] {text[:70]}...")
```

---

## Cohere Rerank API

For production use, Cohere's Rerank API is faster than running a local cross-encoder and handles longer documents.

```python
import os
import cohere

co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

def cohere_rerank(
    query: str,
    documents: list[str],
    top_k: int = 5,
    model: str = "rerank-english-v3.0"
) -> list[dict]:
    """Rerank documents using Cohere Rerank API."""
    results = co.rerank(
        query=query,
        documents=documents,
        top_n=top_k,
        model=model
    )

    return [
        {
            "text": documents[r.index],
            "score": r.relevance_score,
            "original_rank": r.index
        }
        for r in results.results
    ]

# documents = [...your retrieved chunks...]
# reranked = cohere_rerank(query, documents, top_k=3)
# for r in reranked:
#     print(f"[{r['score']:.3f}] {r['text'][:80]}...")
```

---

## Reranking in a full RAG pipeline

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

def retrieval_with_reranking(
    query: str,
    corpus: list[str],
    initial_k: int = 20,
    final_k: int = 5
) -> list[str]:
    """Two-stage retrieval: coarse embedding retrieval + cross-encoder reranking."""
    candidates = bi_encoder_retrieve(query, corpus, k=initial_k)
    reranked = rerank(query, candidates, top_k=final_k)
    return [text for text, _ in reranked]

def answer_with_reranking(query: str, corpus: list[str]) -> str:
    context_chunks = retrieval_with_reranking(query, corpus)
    context = "\n\n".join(f"- {chunk}" for chunk in context_chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the provided context. Be precise."),
        ("human", f"Context:\n{context}\n\nQuestion: {query}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})

answer = answer_with_reranking("How do I reset my password?", CORPUS)
print(f"Answer: {answer}")
```

> [!tip] Reranking sweet spot
> Retrieve 20–50 candidates with your bi-encoder, then rerank to the top 3–5. The initial large k ensures high recall (the right document is somewhere in the candidate set); reranking ensures high precision (the most relevant chunks are at the top). The total latency adds ~50–200ms for the reranking step.

---

## Measuring reranking improvement

```python
def evaluate_retrieval_precision(
    query: str,
    relevant_doc_ids: list[int],
    ranked_chunks: list[str],
    k: int = 5
) -> float:
    """Precision@k: what fraction of the top k are in the relevant set."""
    top_k = ranked_chunks[:k]
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for i, chunk in enumerate(top_k) if i in relevant_set)
    return hits / k

# Compare before and after reranking on your eval set
# If reranking improves Precision@3 by > 10%, keep it in the pipeline
```

---

[[00-agenda]] | [[02-hyde]]
