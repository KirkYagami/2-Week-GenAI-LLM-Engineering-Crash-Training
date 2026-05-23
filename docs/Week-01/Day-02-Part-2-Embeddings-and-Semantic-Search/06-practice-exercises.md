# Practice Exercises — Embeddings and Semantic Search

Three levels: warm-up explores a single concept, main builds a mini-system, stretch handles failure cases and evaluation.

---

## Exercise 1 — Semantic similarity explorer (Warm-up)

Embed a set of sentences and build a visual similarity report to understand what the embedding space captures.

```python
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SENTENCES = [
    # Programming
    "Python is a programming language",
    "Java is used for enterprise software development",
    "JavaScript runs in the browser",
    # AI/ML
    "Neural networks learn from labeled data",
    "Machine learning models improve with more training data",
    "Deep learning uses multiple hidden layers",
    # Unrelated
    "The Eiffel Tower is in Paris",
    "Sushi originated in Japan",
    "The Amazon River is the largest river by discharge",
]

def get_embeddings(texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data])

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return normalized @ normalized.T

print("Embedding sentences...")
embeddings = get_embeddings(SENTENCES)
sim_matrix = cosine_similarity_matrix(embeddings)

# Print similarity matrix as a heat map (ASCII)
print("\nSimilarity Matrix (truncated sentence labels):")
labels = [s[:25].ljust(25) for s in SENTENCES]
header = " " * 27 + "  ".join(f"{i:3d}" for i in range(len(SENTENCES)))
print(header)
for i, (label, row) in enumerate(zip(labels, sim_matrix)):
    scores = "  ".join(f"{score:3.0f}" for score in (row * 100))
    print(f"{i:2d} {label}  {scores}")

# Find most and least similar pairs
np.fill_diagonal(sim_matrix, -1)
max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
np.fill_diagonal(sim_matrix, 2)
min_idx = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
np.fill_diagonal(sim_matrix, np.diagonal(np.ones((9, 9))))

print(f"\nMost similar: '{SENTENCES[max_idx[0]]}'\n             '{SENTENCES[max_idx[1]]}'")
print(f"\nMost different: '{SENTENCES[min_idx[0]]}'\n               '{SENTENCES[min_idx[1]]}'")
```

**Questions to answer after running:**
1. Do the programming sentences cluster together? ML sentences?
2. What's the similarity score between "Python is a programming language" and "Machine learning models improve with more training data"?
3. Pick two paraphrases of the same idea and check their score — does it exceed 0.9?

---

## Exercise 2 — Personal knowledge base (Main)

Build a searchable index of your own notes or a set of Wikipedia excerpts.

```python
import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dataclasses import dataclass, field

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paste in your own notes or use these sample excerpts
KNOWLEDGE_BASE = {
    "rag-overview": """
Retrieval-Augmented Generation (RAG) is an AI architecture that combines a retrieval
system with a language model. Instead of relying solely on the model's parametric
knowledge, RAG fetches relevant documents from an external database and includes
them in the prompt.

RAG was introduced by Meta AI researchers in 2020. The key advantage is that the
knowledge base can be updated without retraining the model. RAG is particularly
useful for tasks requiring up-to-date information or proprietary data.
    """,
    "vector-databases": """
Vector databases are specialized storage systems designed to efficiently store and
query high-dimensional embedding vectors. Unlike traditional databases that use
B-tree indexes for exact match queries, vector databases use approximate nearest
neighbor (ANN) algorithms like HNSW and IVF.

Popular vector databases include Pinecone, Qdrant, Weaviate, Milvus, and ChromaDB.
ChromaDB is popular for local development due to its simple Python API.
Each database has different tradeoffs in terms of performance, scalability, and features.
    """,
    "fine-tuning": """
Fine-tuning adapts a pretrained language model to a specific task by continuing
training on a smaller, task-specific dataset. This updates the model's weights to
specialize its knowledge and behavior.

Modern fine-tuning techniques include LoRA (Low-Rank Adaptation) which adds small
trainable matrices to frozen base model weights, enabling efficient fine-tuning
with a fraction of the compute. QLoRA extends this with 4-bit quantization.

Fine-tuning is most valuable when: you need consistent output format, you have
domain-specific vocabulary, or you're doing the same task at very high volume.
    """,
    "prompt-engineering": """
Prompt engineering is the practice of designing inputs to language models to elicit
desired outputs. Key techniques include zero-shot (no examples), few-shot (2-10 examples),
chain-of-thought (step-by-step reasoning), and structured output (JSON schemas).

System prompts set the model's persona and behavioral rules. XML tags improve
instruction following. Output anchoring (pre-filling the assistant turn) forces
specific response formats.

The most common mistake is judging prompt quality on 5 examples. Always evaluate
on a labeled test set of 50+ representative inputs before shipping to production.
    """
}

@dataclass
class Chunk:
    id: str
    text: str
    source: str

def simple_chunk(text: str, source: str, size: int = 400) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.strip().split('\n\n') if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < size * 4:
            current = (current + " " + para).strip()
        else:
            if current:
                chunks.append(Chunk(id=f"{source}-{len(chunks)}", text=current, source=source))
            current = para
    if current:
        chunks.append(Chunk(id=f"{source}-{len(chunks)}", text=current, source=source))
    return chunks

def build_index(knowledge_base: dict[str, str]) -> tuple[faiss.Index, list[Chunk]]:
    all_chunks = []
    for source, text in knowledge_base.items():
        all_chunks.extend(simple_chunk(text, source))

    print(f"Created {len(all_chunks)} chunks")
    texts = [c.text for c in all_chunks]

    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, all_chunks

def search(query: str, index: faiss.Index, chunks: list[Chunk], k: int = 3) -> list[tuple[Chunk, float]]:
    response = client.embeddings.create(model="text-embedding-3-small", input=[query])
    query_emb = np.array([response.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, k)
    return [(chunks[idx], float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]

# Build and query
index, chunks = build_index(KNOWLEDGE_BASE)

test_queries = [
    "When should I use RAG vs fine-tuning?",
    "What is HNSW and which databases use it?",
    "How do I evaluate my prompt quality?",
    "What is LoRA?",
]

for query in test_queries:
    results = search(query, index, chunks, k=2)
    print(f"\nQ: {query}")
    for chunk, score in results:
        print(f"  [{score:.3f}] {chunk.source}: {chunk.text[:80]}...")
```

**Extension:** Add your own notes to `KNOWLEDGE_BASE` and see how well the search handles your domain.

---

## Exercise 3 — Chunk size ablation (Stretch)

Systematically test how chunk size affects retrieval quality.

```python
import os
import numpy as np
import faiss
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LONG_DOCUMENT = """
The history of artificial intelligence spans over seven decades. The field was formally
founded at the Dartmouth Conference in 1956, where John McCarthy coined the term
"artificial intelligence." Early AI research focused on symbolic reasoning and expert systems.

The first AI winter occurred in the 1970s when funding dried up after researchers
overpromised and underdelivered on early AI systems. Progress was slow due to
limited computing power and the difficulty of encoding human knowledge into rules.

The expert systems era of the 1980s saw renewed interest. Programs like DENDRAL
and MYCIN could diagnose diseases and analyze chemical compounds. However, these
systems were brittle — they failed outside their narrow domain.

The second AI winter came in the late 1980s and early 1990s. Machine learning
began to emerge as an alternative to hand-coded rules. Neural networks, though
invented in the 1950s, became practical as computing improved.

The deep learning revolution began in earnest in 2012 when AlexNet won the ImageNet
competition by a large margin. Convolutional neural networks proved that deep learning
could outperform hand-crafted features on visual tasks.

The transformer architecture, introduced in 2017, enabled the large language model era.
BERT (2018) and GPT-2 (2019) demonstrated that language models pretrained on large
corpora could be fine-tuned for many tasks. GPT-3 (2020) with 175 billion parameters
showed remarkable few-shot capabilities.

The current era (2022–present) is defined by instruction-tuned models and RLHF.
ChatGPT launched in November 2022 and reached 100 million users in two months.
GPT-4o, Claude, Llama 3, and Gemini represent the state of the art in 2025.
"""

TEST_QUERIES = [
    ("Who coined the term artificial intelligence?", "John McCarthy"),
    ("What happened in 2012 that accelerated deep learning?", "AlexNet"),
    ("When was ChatGPT launched?", "November 2022"),
    ("What caused the first AI winter?", "limited computing / overpromised"),
]

def evaluate_retrieval(chunk_size: int, expected_answers: list[tuple[str, str]]) -> dict:
    # Build index with this chunk size
    words = LONG_DOCUMENT.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunks.append(chunk_text)

    # Embed all chunks
    response = client.embeddings.create(model="text-embedding-3-small", input=chunks)
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Evaluate each query
    hits = 0
    for query, expected_keyword in expected_answers:
        q_resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
        q_emb = np.array([q_resp.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(q_emb)

        scores, indices = index.search(q_emb, 2)
        retrieved_text = " ".join(chunks[idx] for idx in indices[0] if idx != -1)

        if expected_keyword.lower() in retrieved_text.lower():
            hits += 1

    recall = hits / len(expected_answers)
    return {
        "chunk_size_words": chunk_size,
        "num_chunks": len(chunks),
        "avg_chunk_chars": len(LONG_DOCUMENT) // len(chunks),
        "recall_at_2": recall,
        "hits": hits,
        "total_queries": len(expected_answers)
    }

print("Chunk size ablation study:")
print("-" * 60)
for chunk_size in [25, 50, 100, 200]:
    result = evaluate_retrieval(chunk_size, TEST_QUERIES)
    print(f"Chunk size {chunk_size:3d} words ({result['num_chunks']:2d} chunks): "
          f"Recall@2 = {result['recall_at_2']:.0%} ({result['hits']}/{result['total_queries']} hits)")
```

**Expected insight:** Very small chunks (25 words) may split answers across chunks. Very large chunks (200 words) may dilute relevance. The sweet spot for this document is likely 50–100 words.

---

## Exercise 4 — Out-of-domain query detection (Stretch)

Build a system that detects when a user query is out of scope for your knowledge base.

```python
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embed a set of in-domain representative queries
IN_DOMAIN_QUERIES = [
    "How does RAG work?",
    "What is vector search?",
    "Explain transformer attention",
    "How do I fine-tune a language model?",
    "What is cosine similarity?",
    "How does prompt caching work?",
]

def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return np.array(response.data[0].embedding)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# Embed reference queries
print("Embedding reference queries...")
ref_embeddings = np.array([get_embedding(q) for q in IN_DOMAIN_QUERIES])

def is_in_domain(query: str, threshold: float = 0.55) -> tuple[bool, float]:
    """Returns (is_in_domain, max_similarity_to_reference)."""
    query_emb = get_embedding(query)
    similarities = [cosine_sim(query_emb, ref) for ref in ref_embeddings]
    max_sim = max(similarities)
    return max_sim >= threshold, max_sim

# Test with in-domain and out-of-domain queries
test_queries = [
    # Expected in-domain
    "What embedding model should I use for semantic search?",
    "How do I implement HNSW indexing?",
    # Expected out-of-domain
    "What is the best recipe for chocolate cake?",
    "Who won the 2024 World Cup?",
    "How do I change a car tire?",
    # Edge cases
    "What is Python?",  # could be either
    "How does attention work?",  # in-domain
]

print("\nOut-of-domain detection (threshold=0.55):")
for query in test_queries:
    in_domain, max_sim = is_in_domain(query)
    status = "IN DOMAIN  " if in_domain else "OUT OF DOMAIN"
    print(f"  [{status}] (sim={max_sim:.3f}) {query}")
```

**Extension:** Adjust the threshold and observe how it affects false positives (incorrectly rejecting in-domain queries) vs. false negatives (accepting irrelevant queries). What threshold minimizes both error types?

---

[[05-semantic-search-pipeline]] | [[07-interview-questions]]
