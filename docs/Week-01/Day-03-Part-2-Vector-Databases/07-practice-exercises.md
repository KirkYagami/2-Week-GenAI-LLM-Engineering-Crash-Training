# Practice Exercises — Vector Databases

---

## Exercise 1 — ChromaDB multi-collection system (Warm-up)

Build a system with two separate ChromaDB collections — one for technical docs, one for HR docs — and route queries to the correct collection.

```python
import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma = chromadb.Client()

oai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Create two collections
tech_col = chroma.get_or_create_collection("tech_docs", embedding_function=oai_ef,
                                            metadata={"hnsw:space": "cosine"})
hr_col = chroma.get_or_create_collection("hr_docs", embedding_function=oai_ef,
                                          metadata={"hnsw:space": "cosine"})

# Populate tech collection
tech_col.add(
    ids=["t1", "t2", "t3"],
    documents=[
        "FastAPI is a Python web framework for building REST APIs with automatic OpenAPI docs",
        "Docker containerizes applications for consistent deployment across environments",
        "PostgreSQL is a relational database with excellent JSON support and full-text search"
    ],
    metadatas=[{"topic": "web"}, {"topic": "devops"}, {"topic": "database"}]
)

# Populate HR collection
hr_col.add(
    ids=["h1", "h2", "h3"],
    documents=[
        "Employees receive 20 days of annual paid leave. Requests need 2 weeks notice.",
        "Health insurance covers the employee and up to 3 dependents. Dental included.",
        "Performance reviews occur in Q4. Ratings range from Exceptional to Needs Improvement."
    ],
    metadatas=[{"topic": "leave"}, {"topic": "benefits"}, {"topic": "performance"}]
)

def classify_query(query: str) -> str:
    """Route query to correct collection using a cheap classification."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Is this query about technology/software OR HR/employee policies? Answer 'tech' or 'hr' only.\n\nQuery: {query}"
        }],
        max_tokens=5,
        temperature=0.0
    )
    label = response.choices[0].message.content.strip().lower()
    return "tech" if "tech" in label else "hr"

def smart_search(query: str, k: int = 2) -> list[dict]:
    collection_type = classify_query(query)
    collection = tech_col if collection_type == "tech" else hr_col

    results = collection.query(query_texts=[query], n_results=k,
                               include=["documents", "metadatas", "distances"])
    return [
        {"text": doc, "meta": meta, "similarity": 1 - dist, "collection": collection_type}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]

# Test
for query in [
    "How do I build a REST API?",
    "How many vacation days do I get?",
    "What's the best way to deploy my app?",
    "Does insurance cover my family?"
]:
    results = smart_search(query)
    print(f"\nQ: {query}")
    print(f"   Routed to: {results[0]['collection']}")
    print(f"   Top result: {results[0]['text'][:70]}...")
```

---

## Exercise 2 — Qdrant filtered search comparison (Main)

Measure how metadata filtering affects result quality and retrieval speed.

```python
import os
import time
import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(":memory:")

# Create a larger collection with mixed content
qdrant.recreate_collection(
    collection_name="mixed_corpus",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

CORPUS = [
    ("Python is used for data science and ML", "tech", "ai", 2024, 4.8),
    ("FastAPI makes building REST APIs easy", "tech", "web", 2024, 4.5),
    ("PostgreSQL is a powerful relational database", "tech", "database", 2023, 4.7),
    ("Docker enables consistent deployments", "tech", "devops", 2024, 4.6),
    ("Machine learning models improve with data", "tech", "ai", 2024, 4.9),
    ("The Eiffel Tower was built in 1889", "travel", "europe", 2023, 4.8),
    ("Tokyo is a city of contrasts", "travel", "asia", 2024, 4.7),
    ("RAG combines retrieval with generation", "tech", "ai", 2024, 4.5),
    ("Llama 3 is an open-source LLM", "tech", "ai", 2024, 4.6),
    ("Paris has excellent museums", "travel", "europe", 2023, 4.4),
]

texts = [c[0] for c in CORPUS]
resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
embeddings = [item.embedding for item in resp.data]

qdrant.upsert(
    collection_name="mixed_corpus",
    points=[
        PointStruct(
            id=i,
            vector=emb,
            payload={
                "text": text,
                "domain": domain,
                "subcategory": subcat,
                "year": year,
                "rating": rating
            }
        )
        for i, (text, domain, subcat, year, rating), emb
        in zip(range(len(CORPUS)), CORPUS, embeddings)
    ]
)

# Index filtered fields
from qdrant_client.models import PayloadSchemaType
for field, schema in [
    ("domain", PayloadSchemaType.KEYWORD),
    ("subcategory", PayloadSchemaType.KEYWORD),
    ("year", PayloadSchemaType.INTEGER),
    ("rating", PayloadSchemaType.FLOAT)
]:
    qdrant.create_payload_index("mixed_corpus", field, schema)

def search_qdrant_timed(
    query: str,
    k: int = 5,
    query_filter: Filter | None = None
) -> tuple[list[dict], float]:
    query_emb = client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    ).data[0].embedding

    start = time.perf_counter()
    results = qdrant.search(
        collection_name="mixed_corpus",
        query_vector=query_emb,
        limit=k,
        query_filter=query_filter,
        with_payload=True
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    return [{"score": r.score, **r.payload} for r in results], elapsed_ms

# Compare: unfiltered vs filtered searches
print("=== Unfiltered search: 'artificial intelligence Python' ===")
results, ms = search_qdrant_timed("artificial intelligence Python")
for r in results:
    print(f"  [{r['score']:.3f}] [{r['domain']}/{r['subcategory']}] {r['text'][:60]}...")
print(f"  Time: {ms:.2f}ms\n")

print("=== Filtered search: domain='tech' AND subcategory='ai' ===")
ai_filter = Filter(must=[
    FieldCondition(key="domain", match=MatchValue(value="tech")),
    FieldCondition(key="subcategory", match=MatchValue(value="ai"))
])
results_f, ms_f = search_qdrant_timed("artificial intelligence Python", query_filter=ai_filter)
for r in results_f:
    print(f"  [{r['score']:.3f}] {r['text'][:60]}...")
print(f"  Time: {ms_f:.2f}ms\n")

print("=== Filtered: year=2024 AND rating>=4.6 ===")
quality_filter = Filter(must=[
    FieldCondition(key="year", range=Range(gte=2024)),
    FieldCondition(key="rating", range=Range(gte=4.6))
])
results_q, ms_q = search_qdrant_timed("machine learning", query_filter=quality_filter)
for r in results_q:
    print(f"  [{r['score']:.3f}] rating={r['rating']} | {r['text'][:60]}...")
print(f"  Time: {ms_q:.2f}ms")
```

---

## Exercise 3 — Hybrid search evaluation (Stretch)

Compare dense-only, sparse-only, and hybrid retrieval on a labeled evaluation set.

```python
import os
import re
import math
from collections import Counter
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CORPUS = [
    {"id": "py1", "text": "Python 3.11 introduced significant performance improvements and better error messages"},
    {"id": "py2", "text": "Python supports object-oriented, functional, and procedural programming styles"},
    {"id": "ml1", "text": "Random forests ensemble multiple decision trees to improve prediction accuracy"},
    {"id": "ml2", "text": "Gradient boosting builds models sequentially, each correcting predecessor errors"},
    {"id": "api1", "text": "REST APIs use HTTP methods GET POST PUT DELETE to manipulate resources"},
    {"id": "api2", "text": "GraphQL provides a flexible query language as an alternative to REST"},
    {"id": "db1", "text": "PostgreSQL JSONB columns allow storing and querying semi-structured data"},
    {"id": "db2", "text": "Redis is an in-memory data store useful for caching and session management"},
]

EVAL_SET = [
    ("Python version 3.11 improvements", "py1"),
    ("decision tree ensemble methods", "ml1"),
    ("HTTP GET POST methods API", "api1"),
    ("caching in-memory store", "db2"),
    ("sequential boosting model correction", "ml2"),
    ("GraphQL alternative REST queries", "api2"),
    ("PostgreSQL JSON data", "db1"),
    ("Python programming paradigms", "py2"),
]

# Build BM25
class SimpleBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.docs = []; self.df = Counter(); self.idf = {}; self.avgdl = 0

    def tokenize(self, text):
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def fit(self, texts):
        self.docs = [self.tokenize(t) for t in texts]
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        for doc in self.docs:
            for term in set(doc): self.df[term] += 1
        N = len(self.docs)
        self.idf = {t: math.log((N - f + 0.5) / (f + 0.5) + 1) for t, f in self.df.items()}

    def score(self, query):
        qt = self.tokenize(query)
        return [sum(
            self.idf.get(t, 0) * (Counter(d)[t] * (self.k1 + 1)) /
            (Counter(d)[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            for t in qt if t in self.idf
        ) for d in self.docs]

texts = [d["text"] for d in CORPUS]
ids = [d["id"] for d in CORPUS]
bm25 = SimpleBM25(); bm25.fit(texts)

resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
embs = np.array([item.embedding for item in resp.data])

def dense_search(query, k):
    qe = np.array(client.embeddings.create(model="text-embedding-3-small", input=[query]).data[0].embedding)
    scores = embs @ qe / (np.linalg.norm(embs, axis=1) * np.linalg.norm(qe) + 1e-8)
    return [(ids[i], float(scores[i])) for i in np.argsort(scores)[::-1][:k]]

def sparse_search(query, k):
    scores = bm25.score(query)
    return [(ids[i], s) for i, s in sorted(enumerate(scores), key=lambda x: -x[1])[:k] if s > 0]

def rrf_merge(lists, k=60):
    scores = {}
    for lst in lists:
        for rank, (id, _) in enumerate(sorted(lst, key=lambda x: -x[1]), 1):
            scores[id] = scores.get(id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

def recall_at_k(results, expected_id, k):
    return expected_id in [id for id, _ in results[:k]]

print(f"{'Query':<40} {'Dense':>6} {'Sparse':>7} {'Hybrid':>7}")
print("-" * 65)
dense_hits = sparse_hits = hybrid_hits = 0
for query, expected in EVAL_SET:
    dense = dense_search(query, 5)
    sparse = sparse_search(query, 5)
    hybrid = rrf_merge([dense, sparse])

    d = recall_at_k(dense, expected, 3)
    s = recall_at_k(sparse, expected, 3)
    h = recall_at_k(hybrid, expected, 3)

    dense_hits += d; sparse_hits += s; hybrid_hits += h
    print(f"{query:<40} {'✓' if d else '✗':>6} {'✓' if s else '✗':>7} {'✓' if h else '✗':>7}")

n = len(EVAL_SET)
print(f"\nRecall@3: Dense={dense_hits/n:.0%}, Sparse={sparse_hits/n:.0%}, Hybrid={hybrid_hits/n:.0%}")
```

**Expected:** Hybrid should equal or outperform both dense and sparse alone. If sparse outperforms dense on many queries, your corpus has high term overlap with queries — consider domain-specific embedding models.

---

[[06-hybrid-search]] | [[08-interview-questions]]
