# What is RAG

Language models have a fundamental limitation: their knowledge is frozen at training time. Ask GPT-4o about your company's internal policies, last week's earnings report, or a document you wrote yesterday — it doesn't know. RAG fixes this by injecting relevant external knowledge into the prompt at query time.

## Learning objectives

- Describe the three-stage RAG pipeline (index, retrieve, generate)
- Explain why RAG reduces hallucination compared to prompting alone
- Identify the failure modes of naive RAG
- Know when RAG is the right architectural choice

---

## The three stages

```
OFFLINE (build once, update as needed):
┌─────────────────────────────────────────────────┐
│ 1. INDEX                                        │
│    Documents → Chunks → Embeddings → Vector DB  │
└─────────────────────────────────────────────────┘

ONLINE (every query):
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 2. RETRIEVE  │ →  │ 3. AUGMENT   │ →  │ 4. GENERATE  │
│ Query → Emb  │    │ Inject chunks│    │ LLM answers  │
│ → Top-k hits │    │ into prompt  │    │ from context │
└──────────────┘    └──────────────┘    └──────────────┘
```

```python
import os
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === STAGE 1: INDEX (offline) ===
documents = [
    {"id": "policy-1", "text": "Employees are entitled to 20 days of paid leave per year. Leave requests must be submitted 2 weeks in advance."},
    {"id": "policy-2", "text": "Remote work is permitted up to 3 days per week. Core hours are 10am-3pm in the employee's local time."},
    {"id": "policy-3", "text": "The annual performance review cycle runs from October to December. Ratings are: Exceptional, Meets Expectations, Needs Improvement."},
    {"id": "benefits-1", "text": "Health insurance covers the employee and up to 3 dependents. Dental and vision are included at no additional cost."},
]

def embed(text: str) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

doc_embeddings = np.array([embed(d["text"]) for d in documents])

# === STAGE 2: RETRIEVE (per query) ===
def retrieve(query: str, k: int = 2) -> list[dict]:
    query_emb = np.array(embed(query))
    # Cosine similarity
    norms = np.linalg.norm(doc_embeddings, axis=1)
    query_norm = np.linalg.norm(query_emb)
    scores = (doc_embeddings @ query_emb) / (norms * query_norm + 1e-8)
    top_k = np.argsort(scores)[::-1][:k]
    return [documents[i] for i in top_k]

# === STAGE 3: AUGMENT + GENERATE ===
def rag_answer(question: str) -> str:
    retrieved = retrieve(question, k=2)
    context = "\n\n".join(f"[{d['id']}] {d['text']}" for d in retrieved)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Answer questions using only the provided context. If the answer isn't in the context, say so."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=200,
        temperature=0.0
    )
    return response.choices[0].message.content

# Test
print(rag_answer("How many days of vacation do I get?"))
print()
print(rag_answer("Can I work from home every day?"))
print()
print(rag_answer("What is the company's stock price?"))  # Out of scope
```

---

## Why RAG reduces hallucination

Without RAG, the model answers from parametric memory — what it learned during training. It will confabulate plausible-sounding but wrong answers when it doesn't know something.

With RAG, the relevant text is literally in the prompt. The model's job shifts from *recall* to *extraction and synthesis*. It's much easier to say "the context says X" than to remember X correctly.

```
Without RAG:
User: "What is our refund policy?"
Model: *searches parametric memory, finds nothing specific*
Model: "Your refund policy allows returns within 30 days..." ← fabricated

With RAG:
User: "What is our refund policy?"
System: *retrieves policy-doc chunk from vector DB*
Context: "Refunds are available within 14 days for unused items only."
Model: "According to policy, refunds are available within 14 days for unused items only."
```

---

## RAG failure modes

> [!warning] Retrieval failure
> If the relevant chunk isn't retrieved — due to poor chunking, wrong embedding model, or inadequate index size — the model won't have the right context and may hallucinate or say "I don't know" when it shouldn't. Retrieval quality is the single biggest lever in RAG systems.

> [!warning] Context doesn't fit in the window
> Retrieving 20 chunks of 500 tokens each = 10,000 tokens before the question and system prompt. On gpt-4o (128K context), that's fine. On gpt-4o-mini (8K effective context for many use cases), you'll hit limits quickly. Size your retrieval k to your model's context budget.

> [!warning] "Lost in the middle" degradation
> Models attend more strongly to information at the beginning and end of long contexts. If the single most relevant chunk is buried in position 8 of 10, answer quality drops. Keep k small (3–5) and sort retrieved chunks by relevance score descending.

> [!warning] Faithfulness vs. grounding tension
> Even with perfect retrieval, a model may blend retrieved content with parametric knowledge. Always instruct the model explicitly: "Use only the provided context. If the answer is not in the context, say so."

---

## RAG vs. prompting alone

| Approach | Knowledge source | Update mechanism | Hallucination risk |
|----------|-----------------|------------------|--------------------|
| Prompt alone | Model weights | Retrain | High for specific facts |
| Prompt + RAG | External documents | Update vector index | Lower — grounded in retrieved text |
| Fine-tuning | Updated model weights | Retrain | Medium — knowledge baked in but can drift |

---

## When to use RAG

> [!success] Use RAG when
> - Your knowledge changes frequently (product docs, policies, news)
> - You have proprietary data the model was never trained on
> - You need citations or source attribution
> - You want to control exactly what information the model can access
> - You need to update the knowledge base without retraining

> [!warning] RAG is not a silver bullet
> RAG requires: a vector database, an embedding model, chunk quality, retrieval evaluation, and careful prompt design. For simple tasks with stable knowledge (e.g., "classify this sentiment"), RAG adds complexity with no benefit.

---

[[00-agenda]] | [[02-chunking-strategies]]
