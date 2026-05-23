# Advanced RAG Patterns

Each technique covered in this section fixes a specific RAG failure mode. This note shows how to combine them into a coherent advanced pipeline, and introduces two additional patterns: parent document retrieval and self-RAG verification.

## Learning objectives

- Combine reranking, multi-query, and compression into one pipeline
- Implement parent document retrieval (small-to-big chunking)
- Apply self-RAG verification to catch retrieval failures
- Choose which techniques to apply based on your specific failure mode

---

## Pattern 1: Full advanced retrieval pipeline

```python
import os
import json
import numpy as np
from openai import OpenAI
from sentence_transformers import CrossEncoder

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embs = np.array([item.embedding for item in resp.data])
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)

def generate_query_variants(question: str, n: int = 3) -> list[str]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Generate {n} search query variations for: {question}\nReturn JSON: {{\"queries\": [...]}}"}],
        temperature=0.7, max_tokens=200, response_format={"type": "json_object"}
    )
    return [question] + json.loads(resp.choices[0].message.content).get("queries", [])

class AdvancedRAG:
    def __init__(self, corpus: list[str], top_k: int = 5):
        self.corpus = corpus
        self.top_k = top_k
        print("Indexing corpus...")
        self.corp_embs = embed(corpus)

    def multi_query_retrieve(self, question: str, n_variants: int = 3, k_per_query: int = 15) -> list[str]:
        variants = generate_query_variants(question, n=n_variants)
        doc_votes: dict[int, int] = {}

        for variant in variants:
            q_emb = embed([variant])[0]
            scores = self.corp_embs @ q_emb
            top_k = np.argsort(scores)[::-1][:k_per_query]
            for idx in top_k:
                doc_votes[idx] = doc_votes.get(idx, 0) + 1

        # Sort by vote count
        ranked_indices = sorted(doc_votes.keys(), key=lambda i: doc_votes[i], reverse=True)
        return [self.corpus[i] for i in ranked_indices[:k_per_query]]

    def rerank(self, question: str, candidates: list[str]) -> list[str]:
        pairs = [[question, c] for c in candidates]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores.tolist()), key=lambda x: -x[1])
        return [text for text, _ in reranked[:self.top_k]]

    def compress(self, question: str, chunks: list[str]) -> list[str]:
        compressed = []
        for chunk in chunks:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Extract only text relevant to: '{question}'\nDocument: {chunk}\nExtract (or 'NOT RELEVANT'):"}],
                temperature=0.0, max_tokens=200
            )
            result = resp.choices[0].message.content.strip()
            if result != "NOT RELEVANT" and len(result) > 15:
                compressed.append(result)
        return compressed

    def answer(self, question: str, use_compression: bool = True) -> dict:
        # Stage 1: Multi-query retrieval
        candidates = self.multi_query_retrieve(question)

        # Stage 2: Reranking
        reranked = self.rerank(question, candidates)

        # Stage 3: Contextual compression (optional)
        context_chunks = self.compress(question, reranked) if use_compression else reranked

        if not context_chunks:
            return {"answer": "I could not find relevant information.", "context": []}

        context = "\n\n".join(context_chunks)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer using only the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.0, max_tokens=300
        )

        return {
            "answer": resp.choices[0].message.content,
            "context": context_chunks,
            "stages": {"candidates": len(candidates), "after_rerank": len(reranked), "after_compress": len(context_chunks)}
        }
```

---

## Pattern 2: Parent document retrieval

Retrieve small chunks (for precise matching) but pass the parent document (for full context) to the LLM.

```python
from dataclasses import dataclass, field

@dataclass
class ParentDocument:
    id: str
    full_text: str
    chunks: list[str] = field(default_factory=list)

def create_parent_child_index(documents: list[dict]) -> tuple[dict, list[str], list[str]]:
    """
    Create an index where:
    - Child chunks are used for retrieval (small, precise)
    - Parent documents are used for generation (full context)
    """
    parents: dict[str, ParentDocument] = {}
    child_chunks: list[str] = []
    child_to_parent: list[str] = []

    for doc in documents:
        parent_id = doc["id"]
        full_text = doc["text"]

        # Split into small child chunks
        words = full_text.split()
        chunk_size = 50  # words per child chunk
        chunks = [
            " ".join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        parent = ParentDocument(id=parent_id, full_text=full_text, chunks=chunks)
        parents[parent_id] = parent

        for chunk in chunks:
            child_chunks.append(chunk)
            child_to_parent.append(parent_id)

    return parents, child_chunks, child_to_parent

def parent_retrieve(
    question: str,
    parents: dict,
    child_chunks: list[str],
    child_to_parent: list[str],
    k: int = 3
) -> list[str]:
    """Retrieve using child chunks, return parent documents."""
    child_embs = embed(child_chunks)
    q_emb = embed([question])[0]
    scores = child_embs @ q_emb

    top_k_children = np.argsort(scores)[::-1][:k*3]  # Over-retrieve children

    # Deduplicate parent documents
    seen_parents = set()
    parent_contexts = []
    for child_idx in top_k_children:
        parent_id = child_to_parent[child_idx]
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_contexts.append(parents[parent_id].full_text)
        if len(parent_contexts) >= k:
            break

    return parent_contexts

# Test
DOCUMENTS = [
    {
        "id": "policy_001",
        "text": """Our return and refund policy is designed to ensure customer satisfaction.
All products can be returned within 30 days of purchase.
Items must be unused and in original packaging.
To initiate a return, contact our support team at support@company.com.
Refunds are processed within 5-7 business days.
Digital products and services are non-refundable once activated."""
    },
    {
        "id": "shipping_001",
        "text": """Standard shipping takes 5-7 business days within the continental US.
Express shipping (2-3 days) is available for an additional $15.
Overnight shipping is available for orders placed before 2pm EST.
International shipping is available to 45 countries.
Free shipping on orders over $50."""
    }
]

parents, children, child_to_parent_map = create_parent_child_index(DOCUMENTS)
print(f"Created {len(parents)} parent docs and {len(children)} child chunks")

results = parent_retrieve(
    "What is the refund policy?",
    parents, children, child_to_parent_map, k=2
)
print(f"\nRetrieved {len(results)} parent documents:")
for r in results:
    print(f"  → {r[:100]}...")
```

---

## Pattern 3: Self-RAG verification

After generating an answer, verify it's grounded in the retrieved context. If not, decide whether to retry with different retrieval.

```python
def self_rag_answer(
    question: str,
    corpus: list[str],
    max_retries: int = 2
) -> dict:
    """
    Self-RAG: generate → verify → retry if needed.
    """
    for attempt in range(max_retries + 1):
        # Retrieve
        q_emb = embed([question])[0]
        c_embs = embed(corpus)
        c_embs /= np.linalg.norm(c_embs, axis=1, keepdims=True)
        scores = c_embs @ q_emb
        top_5 = np.argsort(scores)[::-1][:5]
        context = "\n".join(corpus[i] for i in top_5)

        # Generate
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer using only the provided context. If context is insufficient, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.0, max_tokens=200
        )
        answer = resp.choices[0].message.content

        # Verify — is the answer grounded?
        verify_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"""Is this answer fully supported by the context? Answer 'yes' or 'no'.
Context: {context}
Answer: {answer}"""}],
            temperature=0.0, max_tokens=5
        )
        is_grounded = "yes" in verify_resp.choices[0].message.content.lower()

        if is_grounded or attempt == max_retries:
            return {
                "answer": answer,
                "grounded": is_grounded,
                "attempts": attempt + 1
            }

        # If not grounded, modify the query and retry
        question = f"{question} (please base your answer strictly on available information)"

    return {"answer": "Unable to find a grounded answer.", "grounded": False, "attempts": max_retries + 1}
```

---

## Choosing your advanced RAG techniques

| Failure mode (from RAGAS) | Technique to apply | Expected improvement |
|--------------------------|-------------------|---------------------|
| Low context recall | Multi-query retrieval | +10–20% recall |
| Low context precision | Reranking | +15–25% precision |
| Low faithfulness | Contextual compression + grounding prompt | +10–15% faithfulness |
| Low answer relevancy | Step-back prompting | +5–10% relevancy |
| Query-document mismatch | HyDE | +10–20% recall |
| Large chunks with noise | Parent-child retrieval | Reduced noise |

Start with the technique that addresses your biggest measured failure. Don't apply all of them at once — each adds latency and cost, and their combined effect needs measurement.

---

[[04-contextual-compression]] | [[06-practice-exercises]]
