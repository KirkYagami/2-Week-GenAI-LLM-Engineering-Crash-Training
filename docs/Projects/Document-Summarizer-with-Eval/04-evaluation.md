# Evaluation — Document Summarizer

## Evaluation metrics for summarization

| Metric | What it measures | Implementation |
|--------|-----------------|---------------|
| Coverage | Key facts from source present in summary | Word overlap or LLM judge |
| Faithfulness | No claims in summary that contradict the source | RAGAS or LLM judge |
| Compression ratio | How much was condensed | `len(summary) / len(source)` |
| Coherence | Does the summary read as a complete, logical text? | LLM judge (1–5 scale) |
| Format adherence | Does bullets output have bullets? Executive have TL;DR? | Rule-based checks |

## Ground truth test set

Create reference summaries for 10–15 documents. These are your ground truth — manually written or taken from article abstracts:

```python
# test_documents.py
EVAL_SET = [
    {
        "name": "Wikipedia: Python (programming language)",
        "text": "Python is a high-level, general-purpose programming language...",  # first 2000 chars
        "reference_summary": "Python is a versatile, interpreted programming language known for its readable syntax and large standard library. Created by Guido van Rossum in 1991, it supports multiple programming paradigms and is widely used in web development, data science, and AI.",
        "key_facts": ["Guido van Rossum", "1991", "interpreted", "multiple paradigms"],
    },
    # Add 9–14 more
]
```

## LLM-as-judge evaluation

```python
# eval.py
import os
import httpx
from openai import OpenAI
from test_documents import EVAL_SET
from dotenv import load_dotenv

load_dotenv()
judge = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def judge_faithfulness(summary: str, source: str) -> float:
    resp = judge.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Does this summary contain only information that can be found in the source text? "
                f"Rate 0.0 (contains false/hallucinated claims) to 1.0 (entirely faithful).\n"
                f"Respond with only a number.\n\n"
                f"Source (first 800 chars): {source[:800]}\n\n"
                f"Summary: {summary}"
            ),
        }],
        temperature=0.0, max_tokens=5,
    )
    try:
        return float(resp.choices[0].message.content.strip())
    except ValueError:
        return 0.5

def judge_coherence(summary: str) -> float:
    resp = judge.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Rate the coherence of this summary from 0.0 (incoherent, disjointed) to 1.0 (clear and logically structured).\n"
                f"Respond with only a number.\n\nSummary: {summary}"
            ),
        }],
        temperature=0.0, max_tokens=5,
    )
    try:
        return float(resp.choices[0].message.content.strip())
    except ValueError:
        return 0.5

def key_fact_coverage(summary: str, key_facts: list[str]) -> float:
    summary_lower = summary.lower()
    hits = sum(1 for fact in key_facts if fact.lower() in summary_lower)
    return hits / len(key_facts) if key_facts else 0.0

def run_evaluation(base_url: str = "http://localhost:8000") -> dict:
    results = []
    with httpx.Client(timeout=60.0) as client:
        for doc in EVAL_SET:
            resp = client.post(
                f"{base_url}/summarize/text",
                params={"format_type": "paragraph", "style": "concise"},
                json=doc["text"],
            )
            if resp.status_code != 200:
                print(f"FAIL: {doc['name']} → {resp.status_code}")
                continue

            summary = resp.json()["summary"]
            compression = len(summary) / len(doc["text"])
            faithfulness = judge_faithfulness(summary, doc["text"])
            coherence = judge_coherence(summary)
            coverage = key_fact_coverage(summary, doc.get("key_facts", []))

            results.append({
                "name": doc["name"][:40],
                "faithfulness": faithfulness,
                "coherence": coherence,
                "coverage": coverage,
                "compression": compression,
            })
            print(f"  F={faithfulness:.2f} C={coherence:.2f} Cov={coverage:.0%} | {doc['name'][:40]}")

    avg = lambda k: sum(r[k] for r in results) / len(results) if results else 0
    print(f"\n=== Summarizer Evaluation ===")
    print(f"Faithfulness:  {avg('faithfulness'):.2f}")
    print(f"Coherence:     {avg('coherence'):.2f}")
    print(f"Coverage:      {avg('coverage'):.0%}")
    print(f"Compression:   {avg('compression'):.0%} of original length")
    return {k: avg(k) for k in ["faithfulness", "coherence", "coverage", "compression"]}

if __name__ == "__main__":
    run_evaluation()
```

## Format adherence checks

```python
def check_format_adherence(summary: str, format_type: str) -> bool:
    if format_type == "bullets":
        lines = summary.strip().splitlines()
        bullet_lines = [l for l in lines if l.strip().startswith(("-", "•", "*"))]
        return len(bullet_lines) >= 3
    elif format_type == "executive":
        return any(phrase in summary.lower() for phrase in ["tldr", "tl;dr", "in brief", "summary:"])
    return True  # paragraph format has no strict structural requirement
```

> [!success] Target metrics
> - Faithfulness ≥ 0.85 (< 15% of summaries contain hallucinations)
> - Coherence ≥ 0.80
> - Key fact coverage ≥ 70%
> - Compression ratio: 5–15% of original length for paragraph summaries

---

[[03-advanced-features]] | [[05-deployment]]
