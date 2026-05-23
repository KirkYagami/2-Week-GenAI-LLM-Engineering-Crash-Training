# Embedding Models

Dozens of embedding models exist. The right one depends on your language, domain, deployment constraints, and budget. This note covers the models you'll actually encounter in 2025 production systems.

## Learning objectives

- Compare OpenAI, Cohere, and open-source embedding models
- Use Sentence Transformers for local embedding generation
- Read MTEB benchmark scores to evaluate models for your task
- Batch embed large document collections efficiently

---

## OpenAI embedding models

```python
import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Current production models (2025)
OPENAI_MODELS = {
    "text-embedding-3-small": {"dims": 1536, "price_per_1M": 0.02},
    "text-embedding-3-large": {"dims": 3072, "price_per_1M": 0.13},
    # text-embedding-ada-002 is legacy — don't use for new projects
}

def embed_openai(texts: list[str], model: str = "text-embedding-3-small", dimensions: int | None = None) -> np.ndarray:
    kwargs = {"model": model, "input": texts}
    if dimensions:
        kwargs["dimensions"] = dimensions

    response = client.embeddings.create(**kwargs)
    return np.array([item.embedding for item in response.data])

# Single embedding
single = embed_openai(["What is the capital of France?"])
print(f"Shape: {single.shape}")  # (1, 1536)

# Batch embedding
texts = [f"Document number {i}" for i in range(10)]
batch = embed_openai(texts)
print(f"Batch shape: {batch.shape}")  # (10, 1536)

# Cost estimate
def estimate_embedding_cost(num_texts: int, avg_tokens_per_text: int, model: str = "text-embedding-3-small") -> float:
    price = OPENAI_MODELS[model]["price_per_1M"]
    total_tokens = num_texts * avg_tokens_per_text
    return total_tokens * price / 1_000_000

cost = estimate_embedding_cost(100_000, 150)
print(f"Cost to embed 100K docs (150 tokens avg): ${cost:.2f}")
# Output: Cost to embed 100K docs (150 tokens avg): $0.30
```

> [!tip] `text-embedding-3-small` is the default choice
> At $0.02/1M tokens, embedding a 100K-document corpus costs ~$0.30. Use `text-embedding-3-large` only when benchmarks show a meaningful quality improvement for your specific task.

---

## Sentence Transformers (local / open-source)

Sentence Transformers run locally — no API call, no cost per token, no data leaving your infrastructure.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Downloads model on first use, cached afterward
model = SentenceTransformer("all-MiniLM-L6-v2")  # 80MB, 384-dim

texts = [
    "The quick brown fox jumps over the lazy dog",
    "A fast auburn fox leaps above a sleepy canine",
    "The stock market crashed yesterday",
    "Python is a programming language"
]

# Encode returns numpy array by default
embeddings = model.encode(texts, show_progress_bar=False)
print(f"Shape: {embeddings.shape}")  # (4, 384)

# Compute similarity matrix
from sentence_transformers.util import cos_sim
similarity_matrix = cos_sim(embeddings, embeddings)
print(f"Fox sentences similarity: {similarity_matrix[0][1].item():.3f}")  # ~0.85
print(f"Fox vs. market similarity: {similarity_matrix[0][2].item():.3f}")  # ~0.25
```

### Top open-source models by use case

```python
from sentence_transformers import SentenceTransformer

MODEL_CATALOG = {
    # General purpose
    "all-MiniLM-L6-v2":        {"dims": 384, "speed": "fast", "quality": "good"},
    "all-mpnet-base-v2":       {"dims": 768, "speed": "medium", "quality": "better"},

    # Asymmetric retrieval (query vs. passage are different)
    "multi-qa-MiniLM-L6-cos-v1": {"dims": 384, "speed": "fast", "quality": "good for search"},

    # Multilingual
    "paraphrase-multilingual-MiniLM-L12-v2": {"dims": 384, "speed": "medium", "quality": "50+ languages"},

    # Code (very useful for RAG over codebases)
    "flax-sentence-embeddings/st-codesearch-distilroberta-base": {"dims": 768, "speed": "medium", "quality": "code search"},
}

def compare_models(query: str, candidates: list[str], model_names: list[str]) -> None:
    for model_name in model_names:
        model = SentenceTransformer(model_name)
        query_emb = model.encode([query])
        cand_embs = model.encode(candidates)

        from sentence_transformers.util import cos_sim
        scores = cos_sim(query_emb, cand_embs)[0]

        print(f"\n{model_name}:")
        for text, score in sorted(zip(candidates, scores.tolist()), key=lambda x: -x[1]):
            print(f"  {score:.3f}  {text}")

# compare_models(
#     query="machine learning model deployment",
#     candidates=["MLOps best practices", "Baking bread at home", "Docker for ML models", "PyTorch training loops"],
#     model_names=["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
# )
```

---

## MTEB — evaluating embedding models

The Massive Text Embedding Benchmark (MTEB) evaluates models across 8 task types: retrieval, classification, clustering, reranking, STS, summarization, bitext mining, and pair classification.

Key retrieval scores on BEIR benchmark (higher = better):

| Model | MTEB Retrieval Score | Dims | Cost |
|-------|---------------------|------|------|
| `text-embedding-3-large` | 54.9 | 3072 | $0.13/1M |
| `text-embedding-3-small` | 44.0 | 1536 | $0.02/1M |
| `BAAI/bge-large-en-v1.5` | 54.3 | 1024 | Free (local) |
| `all-mpnet-base-v2` | 43.8 | 768 | Free (local) |
| `all-MiniLM-L6-v2` | 41.0 | 384 | Free (local) |

> [!info] MTEB is a starting point, not the answer
> MTEB scores are averages across many datasets. Your domain may differ significantly. If you're building a legal document retrieval system, benchmark on legal text. If you're building a code search tool, evaluate on code. Always validate on a sample of your actual data before committing to a model.

---

## Batch embedding large corpora

Embedding 100K+ documents efficiently requires batching and error handling.

```python
import time
from typing import Generator

def batch_embed_documents(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 500,
    max_retries: int = 3
) -> np.ndarray:
    """Embed a large list of texts with batching, progress tracking, and retry."""
    all_embeddings = []
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(model=model, input=batch)
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                total_tokens += response.usage.total_tokens
                print(f"Batch {batch_num}/{total_batches}: {len(batch)} texts, {response.usage.total_tokens} tokens")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Error on batch {batch_num}, attempt {attempt + 1}: {e}")
                time.sleep(2 ** attempt)

    cost = total_tokens * 0.02 / 1_000_000  # text-embedding-3-small price
    print(f"\nDone: {len(all_embeddings)} embeddings, {total_tokens:,} tokens, ${cost:.4f}")
    return np.array(all_embeddings)

# Example usage
sample_docs = [f"Document {i}: " + "content " * 20 for i in range(50)]
embeddings = batch_embed_documents(sample_docs, batch_size=20)
print(f"Final embedding array shape: {embeddings.shape}")
```

---

## Local inference with GPU acceleration

For production systems handling high embedding volume where you want zero API cost:

```python
from sentence_transformers import SentenceTransformer
import torch

# Detect available device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

def embed_local(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,  # normalize for cosine similarity
        convert_to_numpy=True
    )

# BGE models work best with a query prefix
def embed_query(query: str) -> np.ndarray:
    return embed_local([f"Represent this sentence for searching relevant passages: {query}"])

def embed_passages(passages: list[str]) -> np.ndarray:
    return embed_local(passages)  # no prefix for passages
```

> [!tip] BGE instruction prefix
> BAAI/bge models are trained with task-specific instruction prefixes for queries. Only the query needs the prefix — document embeddings are fine without it. Skipping the prefix on queries degrades retrieval quality by 2–5% on BEIR benchmarks.

---

## Choosing a model: decision guide

```
Is data privacy critical? (medical, legal, financial)
├── Yes → Local model (bge-large or mpnet-base)
└── No  → Continue...

Do you need multilingual support?
├── Yes → paraphrase-multilingual-MiniLM-L12-v2 (local) or text-embedding-3-large
└── No  → Continue...

What's your latency budget for embedding at query time?
├── < 50ms → all-MiniLM-L6-v2 (local) or text-embedding-3-small (API)
└── > 50ms → mpnet-base-v2 (local) or text-embedding-3-large (API)

Do you have a GPU available?
├── Yes → bge-large-en-v1.5 (matches text-embedding-3-large, zero cost)
└── No  → text-embedding-3-small (cheap API, good quality)
```

---

[[01-what-are-embeddings]] | [[03-cosine-similarity]]
