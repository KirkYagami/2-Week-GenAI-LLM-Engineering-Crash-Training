# Practice Exercises — RAG Basics

---

## Exercise 1 — Chunking strategy comparison (Warm-up)

Compare how different chunking strategies affect the size and coherence of chunks from the same document.

```python
import os
import tiktoken
import re

enc = tiktoken.get_encoding("cl100k_base")

SAMPLE_DOCUMENT = """
Introduction to Transformers

Transformers are a neural network architecture introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. at Google. The architecture relies entirely on attention mechanisms, dispensing with recurrence and convolutions entirely.

The Self-Attention Mechanism

Self-attention allows each position in the encoder to attend to all positions in the previous layer of the encoder. The mechanism works by computing queries, keys, and values from the input embeddings. The attention score between two positions is computed as the scaled dot product of their query and key vectors.

Positional Encoding

Since the transformer contains no recurrence and no convolution, it uses positional encodings to inject information about the relative or absolute position of the tokens in the sequence. The positional encodings have the same dimension as the embeddings so they can be summed.

Applications and Impact

Transformers have revolutionized natural language processing. BERT (2018) used bidirectional transformers for pre-training. GPT series models use decoder-only transformers for language generation. Vision Transformers (ViT) adapted the architecture for computer vision tasks.
""".strip()

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def fixed_chunk(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = enc.decode(tokens[i:i+size]).strip()
        if chunk:
            chunks.append(chunk)
        i += size - overlap
    return chunks

def paragraph_chunk(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def heading_chunk(text: str) -> list[str]:
    sections = re.split(r"\n(?=[A-Z][^\n]+\n)", text)
    return [s.strip() for s in sections if s.strip()]

# Compare
strategies = {
    "fixed-100-20": fixed_chunk(SAMPLE_DOCUMENT, 100, 20),
    "paragraph": paragraph_chunk(SAMPLE_DOCUMENT),
    "heading": heading_chunk(SAMPLE_DOCUMENT)
}

for strategy, chunks in strategies.items():
    token_counts = [count_tokens(c) for c in chunks]
    print(f"\n{strategy}:")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Avg tokens: {sum(token_counts)/len(token_counts):.1f}")
    print(f"  Min/Max: {min(token_counts)}/{max(token_counts)}")
    for i, chunk in enumerate(chunks):
        print(f"  [{i}] {chunk[:60]}... ({count_tokens(chunk)} tokens)")
```

**Questions:**
1. Which strategy produces the most semantically coherent chunks?
2. Which would you choose for this document? Why?
3. Try increasing the fixed chunk size to 200. How does it affect coherence?

---

## Exercise 2 — Build a Wikipedia RAG (Main)

Build a RAG system over a set of Wikipedia excerpts and measure retrieval quality.

```python
import os
import numpy as np
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

WIKI_EXCERPTS = {
    "python": """
Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.

Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0. Python 2.0 was released in 2000 and introduced new features such as list comprehensions, cycle-detecting garbage collection, reference counting, and Unicode support.

Python consistently ranks as one of the most popular programming languages, and has gained widespread use in the machine learning community.
    """,
    "machine_learning": """
Machine learning (ML) is a field of inquiry devoted to understanding and building methods that "learn" — that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence.

Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, agriculture, and computer vision.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
    """,
    "neural_networks": """
A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the human brain. It is a type of machine learning process, called deep learning, that uses interconnected nodes or neurons in a layered structure that resembles the human brain.

Neural networks rely on training data to learn and improve their accuracy over time. Once they are fine-tuned for accuracy, they are powerful tools in computer science and artificial intelligence, allowing us to classify and cluster data at a high velocity. Tasks in speech recognition or image recognition can take minutes versus hours when compared to the manual identification by human experts.

The first practical application of neural networks came in the 1980s, when AT&T Bell Labs used them to read handwritten digits on checks.
    """,
    "eiffel_tower": """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889.

Originally criticized by some of France's leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognisable structures in the world. The tower received 5,889,000 visitors in 2022. The Eiffel Tower is the most visited monument with an entrance fee in the world.

The tower is 330 metres (1,083 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris.
    """
}

def setup_rag(docs: dict[str, str], collection_name: str = "wiki_rag") -> chromadb.Collection:
    chroma = chromadb.Client()
    oai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    try:
        chroma.delete_collection(collection_name)
    except Exception:
        pass
    collection = chroma.create_collection(collection_name, embedding_function=oai_ef)

    ids, documents, metadatas = [], [], []
    for i, (source, text) in enumerate(docs.items()):
        # Split by paragraph
        paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
        for j, para in enumerate(paragraphs):
            ids.append(f"{source}-{j}")
            documents.append(para)
            metadatas.append({"source": source})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(ids)} chunks from {len(docs)} documents")
    return collection

def ask_rag(question: str, collection: chromadb.Collection, k: int = 3) -> dict:
    results = collection.query(query_texts=[question], n_results=k,
                               include=["documents", "metadatas", "distances"])

    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(f"[{src}]: {doc}" for src, doc in zip(sources, chunks))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer using only the provided context. Cite sources. If the answer isn't in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=300,
        temperature=0.0
    )
    return {
        "answer": response.choices[0].message.content,
        "sources_retrieved": sources,
        "chunks_retrieved": len(chunks)
    }

# Build the RAG system
collection = setup_rag(WIKI_EXCERPTS)

# Test with in-domain and out-of-domain questions
questions = [
    "Who created Python and when was it first released?",
    "What are the three types of machine learning?",
    "How tall is the Eiffel Tower?",
    "What is the capital of Japan?",  # out of domain
    "How do neural networks learn?",
]

for q in questions:
    result = ask_rag(q, collection)
    print(f"\nQ: {q}")
    print(f"A: {result['answer'][:200]}...")
    print(f"   Sources: {result['sources_retrieved']}")
```

**Extension:** Add 3 more Wikipedia excerpts on topics of your choice. Test whether queries about those topics are correctly retrieved.

---

## Exercise 3 — Retrieval quality evaluation (Stretch)

Build a small labeled evaluation set and measure Recall@k for your RAG pipeline.

```python
import os
import numpy as np
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Labeled test set: (question, expected_source_that_should_be_retrieved)
EVAL_SET = [
    ("Who invented Python?", "python"),
    ("What is reinforcement learning?", "machine_learning"),
    ("How do neurons in neural networks work?", "neural_networks"),
    ("When was the Eiffel Tower built?", "eiffel_tower"),
    ("What programming paradigms does Python support?", "python"),
    ("What are training data used for in ML?", "machine_learning"),
    ("What was the first practical use of neural networks?", "neural_networks"),
    ("How many people visit the Eiffel Tower?", "eiffel_tower"),
]

def evaluate_retrieval(collection: chromadb.Collection, eval_set: list[tuple], k: int = 3) -> dict:
    hits = 0
    results_detail = []

    for question, expected_source in eval_set:
        results = collection.query(
            query_texts=[question],
            n_results=k,
            include=["metadatas"]
        )
        retrieved_sources = [m["source"] for m in results["metadatas"][0]]
        hit = expected_source in retrieved_sources
        hits += hit
        results_detail.append({
            "question": question[:50],
            "expected": expected_source,
            "retrieved": retrieved_sources,
            "hit": hit
        })

    recall_at_k = hits / len(eval_set)

    return {
        "recall_at_k": recall_at_k,
        "k": k,
        "hits": hits,
        "total": len(eval_set),
        "details": results_detail
    }

# Evaluate at different k values
for k in [1, 2, 3]:
    result = evaluate_retrieval(collection, EVAL_SET, k=k)
    print(f"Recall@{k}: {result['recall_at_k']:.0%} ({result['hits']}/{result['total']})")

# Show failures
print("\nFailures:")
result = evaluate_retrieval(collection, EVAL_SET, k=3)
for detail in result["details"]:
    if not detail["hit"]:
        print(f"  MISS: '{detail['question']}...'")
        print(f"    Expected: {detail['expected']}, Got: {detail['retrieved']}")
```

**Expected:** Recall@3 should be > 85% for this simple corpus. If it's lower, check whether the failing questions are ambiguous or if the chunks are too large.

**Stretch challenge:** Modify the chunking strategy (paragraph vs. fixed-size) and re-run the evaluation. Which strategy achieves higher Recall@3?

---

[[05-rag-vs-fine-tuning]] | [[07-interview-questions]]
