# RAG Cheat Sheet

Quick reference for Retrieval-Augmented Generation patterns.

---

## Minimal RAG pipeline

```python
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_fn = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("docs", embedding_function=embedding_fn)

def rag(question: str, n_results: int = 5) -> str:
    results = collection.query(query_texts=[question], n_results=n_results)
    chunks = results["documents"][0]
    context = "\n\n".join(chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer using only this context:\n{context}"},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content
```

## Chunking strategies

```python
# Fixed size with overlap (simple, good baseline)
def chunk_fixed(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

# Semantic: split on paragraph boundaries
def chunk_semantic(text: str, max_chars: int = 1000) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks
```

## Retrieval techniques

```python
# Basic cosine similarity
results = collection.query(query_texts=[question], n_results=5)

# Hybrid: combine semantic + BM25 keyword search
# (requires additional library: rank_bm25)
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([doc.split() for doc in all_docs])
bm25_scores = bm25.get_scores(question.split())

# Multi-query: generate variations and merge results
queries = [question, paraphrase1, paraphrase2]
all_results = [collection.query(query_texts=[q], n_results=3) for q in queries]
unique_docs = {id: doc for result in all_results
               for id, doc in zip(result["ids"][0], result["documents"][0])}

# HyDE: embed hypothetical answer instead of question
hypothetical = llm.invoke(f"Answer in 2 sentences: {question}")
results = collection.query(query_texts=[hypothetical], n_results=5)
```

## Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, docs: list[str], top_k: int = 3) -> list[str]:
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:top_k]]
```

## Prompt with citations

```python
context = "\n\n---\n\n".join(
    f"[Source {i+1}: {source_id}]\n{chunk}"
    for i, (source_id, chunk) in enumerate(zip(ids, chunks))
)
system = "Answer using only the provided context. Cite [Source N] for each claim."
```

## When RAG fails checklist

| Failure mode | Symptom | Fix |
|-------------|---------|-----|
| Retrieval miss | Answer is "I don't know" for known info | Improve chunking, increase k |
| Wrong chunks retrieved | Answer uses wrong context | Add reranker, use hybrid search |
| Hallucination | Answer contradicts context | Lower temperature, add faithfulness check |
| Context overflow | Truncated prompt errors | Reduce k or chunk size |
| Stale index | Old information returned | Re-ingest on document update |
