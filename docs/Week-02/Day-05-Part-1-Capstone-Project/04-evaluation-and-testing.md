# Evaluation and Testing

A demo that works once on a hand-picked input is not evidence that your system works. Evaluation means running your system on a representative set of inputs and measuring outcomes you defined before you saw the results.

## Learning objectives

- Define evaluation metrics appropriate to your project type
- Build a test suite that runs against your live API
- Measure latency, cache hit rate, and quality metrics quantitatively
- Interpret results and identify the most impactful improvement to make

---

## Evaluation by project type

### Option A — RAG service

The three metrics that matter for RAG:

| Metric | What it measures | How to measure |
|--------|-----------------|---------------|
| Faithfulness | Does the answer only use facts from retrieved context? | RAGAS `faithfulness` or LLM-as-judge |
| Answer relevance | Does the answer address the actual question? | RAGAS `answer_relevancy` |
| Retrieval precision | Are the retrieved chunks relevant? | Human label or MRR on a known Q&A set |

```python
# eval_rag.py
import json
import time
import httpx
from statistics import mean

# Define your test set: questions with known good answers
TEST_SET = [
    {
        "question": "How do I reset my password?",
        "expected_keywords": ["reset", "email", "link"],
    },
    {
        "question": "What payment methods do you accept?",
        "expected_keywords": ["credit card", "PayPal"],
    },
    # Add 15–20 more for a meaningful evaluation
]

def keyword_precision(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords) if keywords else 0.0

def run_rag_eval(base_url: str = "http://localhost:8000") -> dict:
    results = []
    latencies = []
    cache_hits = 0

    with httpx.Client(timeout=30.0) as client:
        # First pass: cold cache
        for item in TEST_SET:
            start = time.perf_counter()
            resp = client.post(f"{base_url}/chat", json={"question": item["question"], "use_cache": False})
            latency_ms = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                print(f"FAIL: {item['question'][:40]} → {resp.status_code}")
                continue

            data = resp.json()
            precision = keyword_precision(data["answer"], item["expected_keywords"])
            results.append({"question": item["question"], "precision": precision, "latency_ms": latency_ms})
            latencies.append(latency_ms)

        # Second pass: warm cache
        for item in TEST_SET[:5]:
            resp = client.post(f"{base_url}/chat", json={"question": item["question"], "use_cache": True})
            if resp.json().get("cached"):
                cache_hits += 1

    avg_precision = mean(r["precision"] for r in results) if results else 0.0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0

    print(f"\n=== RAG Evaluation Results ===")
    print(f"Questions evaluated: {len(results)}")
    print(f"Avg keyword precision: {avg_precision:.0%}")
    print(f"P95 latency: {p95_latency:.0f}ms")
    print(f"Cache hits (5 re-runs): {cache_hits}/5")

    for r in sorted(results, key=lambda x: x["precision"]):
        print(f"  {r['precision']:.0%} | {r['latency_ms']:.0f}ms | {r['question'][:60]}")

    return {"avg_precision": avg_precision, "p95_latency_ms": p95_latency, "cache_hits": cache_hits}

if __name__ == "__main__":
    run_rag_eval()
```

---

### Option B — LangGraph agent

Agents are harder to evaluate because output quality is subjective. Use a combination of structural checks (did it terminate? how many loops?) and LLM-as-judge for content quality.

```python
# eval_agent.py
import asyncio
import time
from research_agent import run_research

TEST_QUESTIONS = [
    "What are the tradeoffs between RAG and fine-tuning for LLM applications?",
    "How does the transformer attention mechanism work?",
    "What is the difference between LoRA and QLoRA?",
]

QUALITY_CRITERIA = [
    "Has an executive summary",
    "Cites specific technical details",
    "Has a clear conclusion",
    "Is longer than 300 words",
]

def structural_check(report: str) -> dict:
    checks = {
        "has_exec_summary": any(phrase in report.lower() for phrase in ["executive summary", "overview", "introduction"]),
        "has_conclusion": any(phrase in report.lower() for phrase in ["conclusion", "in summary", "in conclusion"]),
        "word_count_ok": len(report.split()) >= 300,
        "has_sections": report.count("##") >= 2 or report.count("**") >= 3,
    }
    return checks

async def run_agent_eval() -> None:
    total_attempts = 0
    total_score = 0.0
    results = []

    for question in TEST_QUESTIONS:
        start = time.perf_counter()
        result = await run_research(question)
        elapsed = time.perf_counter() - start

        checks = structural_check(result["report"])
        structural_score = sum(checks.values()) / len(checks)

        results.append({
            "question": question[:60],
            "quality_score": result["quality_score"],
            "structural_score": structural_score,
            "attempts": result["attempts"],
            "elapsed_s": round(elapsed, 1),
        })
        total_score += result["quality_score"]
        total_attempts += result["attempts"]

    print(f"\n=== Agent Evaluation Results ===")
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print(f"Avg quality score: {total_score/len(TEST_QUESTIONS):.2f}")
    print(f"Avg attempts (loops): {total_attempts/len(TEST_QUESTIONS):.1f}")
    for r in results:
        print(f"  Q={r['quality_score']:.2f} S={r['structural_score']:.2f} att={r['attempts']} t={r['elapsed_s']}s | {r['question']}")

asyncio.run(run_agent_eval())
```

---

### Option C — Document extraction

For extraction tasks, ground-truth comparison is the most rigorous evaluation.

```python
# eval_extraction.py
import json
import httpx

# Ground truth: manually verified extractions from test PDFs
GROUND_TRUTH = [
    {
        "pdf": "test_invoice_001.pdf",
        "expected": {
            "vendor_name": "Acme Corp",
            "total_amount": 1250.00,
            "currency": "USD",
            "invoice_number": "INV-2024-001",
        }
    },
    # Add more test cases
]

def field_accuracy(predicted: dict, expected: dict) -> float:
    correct = 0
    for field, expected_val in expected.items():
        pred_val = predicted.get(field)
        if isinstance(expected_val, float):
            correct += abs((pred_val or 0) - expected_val) < 0.01
        else:
            correct += str(pred_val).lower() == str(expected_val).lower()
    return correct / len(expected) if expected else 0.0

def run_extraction_eval(base_url: str = "http://localhost:8000") -> dict:
    results = []
    with httpx.Client(timeout=30.0) as client:
        for case in GROUND_TRUTH:
            with open(case["pdf"], "rb") as f:
                resp = client.post(
                    f"{base_url}/extract/invoice",
                    files={"file": (case["pdf"], f, "application/pdf")},
                )
            if resp.status_code != 200:
                print(f"FAIL: {case['pdf']} → {resp.status_code}: {resp.text}")
                continue

            predicted = resp.json()
            accuracy = field_accuracy(predicted, case["expected"])
            results.append({"file": case["pdf"], "accuracy": accuracy, "confidence": predicted.get("confidence", 0)})
            print(f"  {accuracy:.0%} accuracy | conf={predicted.get('confidence', 0):.2f} | {case['pdf']}")

    avg_accuracy = sum(r["accuracy"] for r in results) / len(results) if results else 0.0
    print(f"\nAverage field accuracy: {avg_accuracy:.0%}")
    return {"avg_accuracy": avg_accuracy, "n": len(results)}

if __name__ == "__main__":
    run_extraction_eval()
```

---

## API endpoint testing (all options)

Before running the eval suite, verify the endpoints themselves:

```python
# test_endpoints.py
import httpx

def test_health(base_url: str = "http://localhost:8000") -> bool:
    resp = httpx.get(f"{base_url}/health")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    data = resp.json()
    assert data["status"] == "ok"
    print(f"Health: OK — {data}")
    return True

def test_invalid_input(base_url: str = "http://localhost:8000") -> bool:
    # Empty message should return 422
    resp = httpx.post(f"{base_url}/chat", json={"question": ""})
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
    print("Invalid input: correctly rejected")

    # Message too long
    resp = httpx.post(f"{base_url}/chat", json={"question": "x" * 5000})
    assert resp.status_code == 422
    print("Oversized input: correctly rejected")
    return True

if __name__ == "__main__":
    test_health()
    test_invalid_input()
    print("All endpoint tests passed")
```

> [!success] Evaluation targets for capstone
> - RAG: avg keyword precision ≥ 70%, P95 latency < 3s, cache hit rate ≥ 50% on repeated queries
> - Agent: avg quality score ≥ 0.75, avg loops ≤ 2.0
> - Extraction: field accuracy ≥ 80% on ground-truth test set

> [!warning] Don't evaluate on your training/development inputs
> If you only test on examples you already checked while building, you're measuring memorization not generalization. Your eval set should contain inputs you haven't looked at during development.

---

[[03-implementation]] | [[05-presentation-guide]]
