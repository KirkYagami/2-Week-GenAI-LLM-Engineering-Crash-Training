# Building a Semantic Search Pipeline

A semantic search pipeline has four stages: chunk, embed, index, retrieve. Add a generation step and you have the core of a RAG system. This note builds a working end-to-end pipeline.

## Learning objectives

- Implement document chunking with overlap
- Build and persist a complete vector index
- Retrieve relevant chunks and generate a grounded answer
- Add a reranking step to improve precision

---

## Stage 1 — Chunking

LLMs and embedding models have token limits. Long documents must be split into chunks that fit within these limits while preserving semantic coherence.

```python
import re
from dataclasses import dataclass, field

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,     # tokens (approximate: 1 token ≈ 4 chars)
    overlap: int = 50           # overlap between consecutive chunks
) -> list[Chunk]:
    """Split text into overlapping chunks by approximate token count."""
    words = text.split()
    chars_per_chunk = chunk_size * 4
    overlap_chars = overlap * 4

    chunks = []
    start = 0

    while start < len(text):
        end = start + chars_per_chunk

        # Find a natural boundary (sentence end) near the target end
        if end < len(text):
            # Look for sentence boundary within 200 chars
            boundary_search = text[max(end - 200, start):min(end + 200, len(text))]
            sentence_end = re.search(r'[.!?]\s+', boundary_search)
            if sentence_end:
                end = max(end - 200, start) + sentence_end.end()

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk = Chunk(
                id=f"{source}-{len(chunks)}",
                text=chunk_text,
                source=source,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)

        # Move forward with overlap
        start = max(start + chars_per_chunk - overlap_chars, start + 1)
        if start >= len(text):
            break

    return chunks

# Example
sample_text = """
Artificial intelligence is transforming every industry. Machine learning, a subset of AI,
enables computers to learn from data without explicit programming.

Deep learning, a type of machine learning, uses neural networks with many layers.
These networks can recognize patterns in images, text, and audio with remarkable accuracy.

Large language models like GPT-4o and Claude represent the frontier of AI capability.
They're trained on vast amounts of text and can understand and generate human-like language.
These models power applications from customer service chatbots to code assistants.
""".strip()

chunks = chunk_text(sample_text, source="ai-overview.txt", chunk_size=100)
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}: {len(chunk.text)} chars — {chunk.text[:60]}...")
```

---

## Stage 2 — Embed and Index

```python
import os
import json
import numpy as np
import faiss
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SemanticIndex:
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.model = embedding_model
        self.chunks: list[Chunk] = []
        self.index: faiss.Index | None = None
        self.dimension: int | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = client.embeddings.create(model=self.model, input=texts)
        embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def add_documents(self, documents: dict[str, str]) -> None:
        """Add documents as {source: text} pairs."""
        all_chunks = []
        for source, text in documents.items():
            all_chunks.extend(chunk_text(text, source=source))

        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        # Embed in batches of 100
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embed(texts)
            all_embeddings.append(embeddings)
            print(f"  Embedded batch {i // batch_size + 1}/{(len(all_chunks) + batch_size - 1) // batch_size}")

        embeddings_array = np.vstack(all_embeddings)

        if self.dimension is None:
            self.dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(embeddings_array)
        self.chunks.extend(all_chunks)
        print(f"Index contains {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_emb = self._embed([query])
        scores, indices = self.index.search(query_emb, k)

        return [
            (self.chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def save(self, path: str) -> None:
        faiss.write_index(self.index, f"{path}.faiss")
        chunks_data = [
            {"id": c.id, "text": c.text, "source": c.source,
             "chunk_index": c.chunk_index, "metadata": c.metadata}
            for c in self.chunks
        ]
        with open(f"{path}.json", "w") as f:
            json.dump({"model": self.model, "chunks": chunks_data}, f)
        print(f"Saved index to {path}.faiss and {path}.json")

    @classmethod
    def load(cls, path: str) -> "SemanticIndex":
        store = cls()
        store.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.json") as f:
            data = json.load(f)
        store.model = data["model"]
        store.chunks = [Chunk(**c) for c in data["chunks"]]
        store.dimension = store.index.d
        print(f"Loaded index: {store.index.ntotal} vectors")
        return store
```

---

## Stage 3 — Retrieve and Generate

```python
from anthropic import Anthropic

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def answer_with_rag(
    query: str,
    index: SemanticIndex,
    k: int = 4,
    model: str = "claude-sonnet-4-6"
) -> dict:
    # 1. Retrieve relevant chunks
    results = index.search(query, k=k)

    if not results:
        return {"answer": "No relevant information found.", "sources": [], "chunks_used": 0}

    # 2. Format context
    context_parts = []
    sources = set()
    for chunk, score in results:
        context_parts.append(f"[Source: {chunk.source}, Score: {score:.3f}]\n{chunk.text}")
        sources.add(chunk.source)

    context = "\n\n---\n\n".join(context_parts)

    # 3. Generate grounded answer
    system_prompt = """You are a helpful assistant that answers questions based on provided context.
Rules:
- Only use information from the provided context.
- If the context doesn't contain the answer, say "I don't have enough information to answer this."
- Always cite which source(s) your answer comes from.
- Be concise and accurate."""

    user_message = f"""Context:
{context}

Question: {query}"""

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    return {
        "answer": response.content[0].text,
        "sources": list(sources),
        "chunks_used": len(results),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }

# Full pipeline demonstration
DOCUMENTS = {
    "python-intro.txt": """
Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.
It emphasizes code readability and simplicity. Python is widely used in data science,
web development, automation, and artificial intelligence.

Python uses dynamic typing and garbage collection. Its standard library is extensive,
covering everything from file I/O to web services. The package ecosystem (PyPI) has
over 400,000 packages.

Key features: simple syntax, extensive libraries, strong community, cross-platform,
supports multiple programming paradigms (OOP, functional, procedural).
    """,
    "transformers.txt": """
Transformers are a neural network architecture introduced in 2017 by Vaswani et al.
in the paper "Attention Is All You Need." They use self-attention to process entire
sequences simultaneously, unlike earlier RNN-based models.

The key components are: multi-head self-attention, feed-forward networks, positional
encoding, and layer normalization. Transformers form the basis of all modern large
language models including GPT-4o, Claude, Llama 3, and Gemini.

The attention mechanism allows each token to attend to all other tokens in the sequence,
capturing long-range dependencies that RNNs struggled with.
    """
}

index = SemanticIndex()
index.add_documents(DOCUMENTS)

# Answer questions
questions = [
    "What is Python and who created it?",
    "How does self-attention work in transformers?",
    "What year was Python created?"
]

for question in questions:
    print(f"\nQ: {question}")
    result = answer_with_rag(question, index)
    print(f"A: {result['answer']}")
    print(f"   Sources: {result['sources']}, Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
```

---

## Stage 4 — Reranking

Retrieval gets you the top-k candidates. Reranking re-scores them with a more powerful model to improve precision.

```python
from sentence_transformers import CrossEncoder

# Cross-encoders take (query, passage) pairs and return relevance scores
# They're slower than bi-encoders but much more accurate
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(
    query: str,
    results: list[tuple[Chunk, float]],
    top_k: int = 3
) -> list[tuple[Chunk, float]]:
    """Rerank retrieval results using a cross-encoder."""
    if not results:
        return []

    pairs = [(query, chunk.text) for chunk, _ in results]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip([chunk for chunk, _ in results], rerank_scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    return reranked[:top_k]

# Use reranking in the pipeline
def answer_with_reranking(query: str, index: SemanticIndex, retrieve_k: int = 10, return_k: int = 3) -> dict:
    # Retrieve more candidates than needed
    candidates = index.search(query, k=retrieve_k)

    # Rerank to find the most relevant
    reranked = rerank_results(query, candidates, top_k=return_k)

    # Generate with reranked context
    context = "\n\n---\n\n".join(
        f"[Rerank score: {score:.3f}]\n{chunk.text}"
        for chunk, score in reranked
    )

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system="Answer based only on the provided context. Cite your sources.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}]
    )

    return {
        "answer": response.content[0].text,
        "candidates_retrieved": len(candidates),
        "chunks_after_rerank": len(reranked)
    }
```

> [!tip] When to add reranking
> Reranking adds 50–200ms and the cost of running a cross-encoder model. Add it when: your embedding model retrieves semantically similar but topically irrelevant results, you're operating in a domain with specialized vocabulary, or precision matters more than recall.

---

## Performance benchmarks

```python
import time

def benchmark_pipeline(index: SemanticIndex, queries: list[str]) -> dict:
    retrieval_times = []
    generation_times = []

    for query in queries:
        # Retrieval
        t0 = time.perf_counter()
        results = index.search(query, k=5)
        retrieval_times.append((time.perf_counter() - t0) * 1000)

        # Generation
        if results:
            context = "\n\n".join(chunk.text for chunk, _ in results[:3])
            t0 = time.perf_counter()
            anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",  # use Haiku for speed benchmarking
                max_tokens=200,
                messages=[{"role": "user", "content": f"Context:\n{context}\n\nQ: {query}\nA:"}]
            )
            generation_times.append((time.perf_counter() - t0) * 1000)

    return {
        "avg_retrieval_ms": np.mean(retrieval_times),
        "avg_generation_ms": np.mean(generation_times),
        "total_avg_ms": np.mean(retrieval_times) + np.mean(generation_times),
        "num_queries": len(queries)
    }

# benchmark_pipeline(index, questions)
```

---

## Common mistakes

> [!warning] Chunk size is a retrieval hyperparameter, not a config value
> The optimal chunk size depends on your query type. Short, factual queries ("What year was X founded?") need small chunks with precise information. Complex questions need larger chunks with more context. Benchmark retrieval quality at 200, 500, and 1000 token chunk sizes on your actual queries.

> [!warning] Don't embed the same text twice
> If you rebuild the index on every app start, you pay embedding costs repeatedly. Persist the index with `save()` and load it on startup. Only re-embed when documents change.

> [!warning] Context window limits still apply to generation
> Even if you retrieve 10 chunks of 500 tokens each, that's 5,000 tokens in your prompt before the question and system prompt. Add the model's output and you can hit context limits fast. Count tokens before sending.

---

[[04-vector-search]] | [[06-practice-exercises]]
