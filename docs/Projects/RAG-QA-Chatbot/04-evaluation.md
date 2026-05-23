# Evaluation — RAG Q&A Chatbot

## Evaluation strategy

Evaluate two things separately: retrieval quality and generation quality. A failure in retrieval will always cause a failure in generation — don't confuse them.

```
Retrieval quality → Are the right chunks being returned?
Generation quality → Is the answer faithful to the retrieved context?
```

## Building a test set

Before running any evaluation, define 20 test questions with:
- The question
- The expected source document(s) that contain the answer
- The key facts the answer must include

```python
# test_set.py
TEST_SET = [
    {
        "question": "What is your refund window?",
        "expected_sources": ["refund-policy.md"],
        "required_facts": ["30 days", "original payment method"],
    },
    {
        "question": "How do I contact customer support?",
        "expected_sources": ["contact.md"],
        "required_facts": ["support@company.com", "9am-5pm"],
    },
    # ... add 18 more
]
```

## Retrieval evaluation

```python
# eval_retrieval.py
import httpx
from test_set import TEST_SET

def recall_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Fraction of expected sources found in top-K retrieved."""
    hits = sum(1 for s in expected_sources if any(s in r for r in retrieved_sources))
    return hits / len(expected_sources) if expected_sources else 0.0

def run_retrieval_eval(base_url: str = "http://localhost:8000") -> dict:
    recall_scores = []
    with httpx.Client(timeout=30.0) as client:
        for item in TEST_SET:
            resp = client.post(f"{base_url}/chat",
                json={"question": item["question"], "use_cache": False})
            if resp.status_code != 200:
                print(f"FAIL: {item['question'][:40]} → {resp.status_code}")
                continue
            data = resp.json()
            recall = recall_at_k(data["sources"], item["expected_sources"])
            recall_scores.append(recall)
            status = "HIT" if recall == 1.0 else f"MISS ({recall:.0%})"
            print(f"  [{status}] {item['question'][:50]}")

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    print(f"\nRetrieval Recall@5: {avg_recall:.0%} over {len(recall_scores)} questions")
    return {"recall_at_5": avg_recall}

if __name__ == "__main__":
    run_retrieval_eval()
```

## RAGAS evaluation

RAGAS measures faithfulness and answer relevancy without requiring ground-truth answers.

```python
# eval_ragas.py
import os
import httpx
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from test_set import TEST_SET

def build_ragas_dataset(base_url: str = "http://localhost:8000") -> Dataset:
    rows = []
    with httpx.Client(timeout=30.0) as client:
        for item in TEST_SET:
            resp = client.post(f"{base_url}/chat",
                json={"question": item["question"], "n_results": 5, "use_cache": False})
            if resp.status_code != 200:
                continue
            data = resp.json()

            # Fetch the raw chunks used (requires an internal endpoint or log)
            rows.append({
                "question": item["question"],
                "answer": data["answer"],
                "contexts": [data["answer"]],  # Replace with actual retrieved chunks if accessible
                "ground_truth": " ".join(item["required_facts"]),
            })
    return Dataset.from_list(rows)

def run_ragas_eval(base_url: str = "http://localhost:8000") -> None:
    dataset = build_ragas_dataset(base_url)
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(f"\n=== RAGAS Results ===")
    print(f"Faithfulness:     {results['faithfulness']:.3f}")
    print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")

if __name__ == "__main__":
    run_ragas_eval()
```

## Latency benchmarking

```python
# bench.py
import time
import statistics
import httpx

QUESTIONS = [
    "What is your refund policy?",
    "How do I cancel my subscription?",
    "What payment methods do you accept?",
    "Is there a free trial?",
    "How do I export my data?",
]

def benchmark(base_url: str = "http://localhost:8000", n_runs: int = 3) -> dict:
    latencies = []
    with httpx.Client(timeout=30.0) as client:
        for _ in range(n_runs):
            for q in QUESTIONS:
                start = time.perf_counter()
                resp = client.post(f"{base_url}/chat",
                    json={"question": q, "use_cache": False})
                latency = (time.perf_counter() - start) * 1000
                if resp.status_code == 200:
                    latencies.append(latency)

    sorted_l = sorted(latencies)
    p50 = sorted_l[len(sorted_l) // 2]
    p95 = sorted_l[int(len(sorted_l) * 0.95)]
    print(f"P50: {p50:.0f}ms | P95: {p95:.0f}ms | Mean: {statistics.mean(latencies):.0f}ms")
    return {"p50_ms": p50, "p95_ms": p95}

if __name__ == "__main__":
    benchmark()
```

> [!success] Target metrics for portfolio quality
> - Retrieval Recall@5: ≥ 80%
> - RAGAS Faithfulness: ≥ 0.85
> - P95 Latency: < 3,000ms (cold cache)
> - Cache hit rate: 100% on repeated identical queries

---

[[03-advanced-features]] | [[05-deployment]]
