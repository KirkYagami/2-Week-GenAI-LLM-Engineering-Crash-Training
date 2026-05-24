# RAG Fundamentals

RAG is the most-asked-about topic in LLM engineering interviews. Every interviewer will probe it from at least two angles: do you understand the architecture, and have you actually built one and measured it?

---

## Q1: Explain the RAG architecture in under two minutes.

??? "Show answer"
    RAG combines a retrieval system with a generative model. The pipeline has four stages:

    1. **Ingestion** — split documents into chunks, embed each chunk, store embeddings in a vector database
    2. **Retrieval** — embed the user's query, find the top-k most similar chunks by cosine similarity
    3. **Augmentation** — insert the retrieved chunks into the prompt as context
    4. **Generation** — the LLM answers using the provided context, not just its training data

    ```python
    import os
    from openai import OpenAI
    import chromadb

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_collection("docs")

    def rag_answer(query: str) -> str:
        # Retrieve
        query_embedding = client.embeddings.create(
            input=query, model="text-embedding-3-small"
        ).data[0].embedding

        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        context = "\n\n".join(results["documents"][0])

        # Generate
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer using only the provided context. If the answer isn't in the context, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return response.choices[0].message.content
    ```

---

## Q2: How does RAG reduce hallucination?

??? "Show answer"
    RAG reduces hallucination by grounding the model in retrieved source documents at inference time, rather than relying on parametric memory (weights trained on static data).

    The key mechanism: retrieved context is inserted directly into the prompt's context window. If the model is instructed to "answer only from the provided context," it must either use the retrieved text or admit it doesn't know. This is much harder for the model to deviate from than trying to recall facts from training.

    RAG does **not** eliminate hallucination — it shifts the failure modes:
    - If retrieval fails (wrong chunks returned), the model will still hallucinate with high confidence
    - If the context is ambiguous, the model may still interpolate incorrectly
    - The model can still hallucinate citations (claim a chunk says something it doesn't)

    This is why evaluation — specifically faithfulness scoring — matters as much as retrieval quality.

---

## Q3: What's the difference between RAG and fine-tuning? When do you use each?

??? "Show answer"
    | | RAG | Fine-tuning |
    |-|-----|-------------|
    | **What it changes** | What the model sees at inference time | The model's weights |
    | **Data freshness** | Update the vector index without retraining | Requires retraining for new data |
    | **Use when** | Facts change frequently, need citations, data is large | Style/format adaptation, task-specific behaviour, latency-critical |
    | **Weakness** | Retrieval quality is a ceiling | Expensive, slow to iterate, can cause forgetting |

    **Use RAG when**: your data changes (product docs, policy, prices), you need citations, your corpus is too large to fit in context.

    **Use fine-tuning when**: you need the model to write in a very specific style or format, output a domain-specific schema, or handle task patterns not in its training distribution.

    **Use both when**: you want the model to know your domain deeply (fine-tune) and access up-to-date specific facts (RAG). This is the approach most production systems eventually reach.

---

## Q4: What happens when retrieval returns nothing relevant? How do you handle it?

??? "Show answer"
    When retrieval fails, the model receives irrelevant or empty context and will either:
    - Hallucinate an answer from training data (common)
    - Generate a generic non-answer ("I don't have information about that")
    - Confuse itself with unrelated context

    Handling strategies:

    1. **Relevance threshold** — filter retrieved chunks by similarity score. If no chunk exceeds 0.75 cosine similarity, treat retrieval as failed.

    ```python
    results = collection.query(query_embeddings=[query_embedding], n_results=5, include=["distances"])
    threshold = 0.25  # ChromaDB returns L2 distance; lower = more similar
    good_chunks = [doc for doc, dist in zip(results["documents"][0], results["distances"][0]) if dist < threshold]

    if not good_chunks:
        return "I couldn't find relevant information in the knowledge base to answer this question."
    ```

    2. **Explicit fallback message** — instruct the model in the system prompt: "If the context doesn't contain the answer, respond with 'I don't have information about this.'"

    3. **Confidence metadata** — return the top similarity score alongside the answer so downstream systems can decide whether to surface it.

---

## Q5: How do you evaluate a RAG pipeline?

??? "Show answer"
    Evaluate at two levels:

    **Retrieval quality**:
    - **Recall@k** — what fraction of relevant chunks are in the top-k results
    - **MRR (Mean Reciprocal Rank)** — how high the first relevant chunk ranks

    **Generation quality** (RAGAS framework):
    - **Faithfulness** — does the answer contain only claims supported by the retrieved context? (0–1, higher is better)
    - **Answer relevancy** — does the answer address the question? (0–1)
    - **Context recall** — does the retrieved context contain the information needed? (requires ground truth)
    - **Context precision** — how much of the retrieved context is actually relevant?

    ```python
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from datasets import Dataset

    data = {
        "question": ["What is the return policy?"],
        "answer": ["Returns are accepted within 30 days."],
        "contexts": [["Our return policy allows returns within 30 days of purchase."]],
        "ground_truth": ["Returns are accepted within 30 days of purchase."],
    }

    result = evaluate(Dataset.from_dict(data), metrics=[faithfulness, answer_relevancy, context_recall])
    print(result)
    ```

    A minimum viable eval set is 20–50 question/answer pairs with ground truth. Less than that and variance dominates your numbers.

---

*Previous: [Structured Output](../01-Prompt-Engineering/03-structured-output.md) | Next: [Chunking and Retrieval](02-chunking-and-retrieval.md)*
