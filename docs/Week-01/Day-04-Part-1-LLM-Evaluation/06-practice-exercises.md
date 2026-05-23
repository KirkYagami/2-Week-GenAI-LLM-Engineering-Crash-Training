# Practice Exercises — LLM Evaluation

---

## Exercise 1 — Build a faithfulness checker (Warm-up)

Implement a faithfulness scoring function and test it on five question-answer-context triples that mix faithful and hallucinated answers. Report which answers fail the faithfulness check.

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TEST_CASES = [
    {
        "id": "case_1",
        "question": "When was the Eiffel Tower built?",
        "context": "The Eiffel Tower was constructed between 1887 and 1889 as the entrance arch for the 1889 World's Fair.",
        "answer": "The Eiffel Tower was built between 1887 and 1889 for the World's Fair.",
        # Expected: faithful
    },
    {
        "id": "case_2",
        "question": "What is Python used for?",
        "context": "Python is a high-level programming language known for its simplicity and readability.",
        "answer": "Python is used for web development, data science, machine learning, automation, and mobile app development.",
        # Expected: partially hallucinated (mobile apps not in context)
    },
    {
        "id": "case_3",
        "question": "Who founded OpenAI?",
        "context": "OpenAI was founded in December 2015 by a group including Sam Altman and Greg Brockman.",
        "answer": "OpenAI was founded in 2015 by Sam Altman, Elon Musk, Greg Brockman, and 12 other co-founders.",
        # Expected: hallucinated (12 co-founders not supported, Musk may or may not be in context)
    },
    {
        "id": "case_4",
        "question": "What temperature should chicken be cooked to?",
        "context": "The USDA recommends cooking whole chicken to an internal temperature of 165°F (74°C).",
        "answer": "Chicken should be cooked to at least 165°F according to USDA guidelines.",
        # Expected: faithful
    },
    {
        "id": "case_5",
        "question": "How fast does light travel?",
        "context": "Light travels at approximately 299,792 kilometers per second in a vacuum.",
        "answer": "Light travels at 300,000 km/s and can circle the Earth 7.5 times in one second.",
        # Expected: hallucinated (second fact not in context)
    },
]

def check_faithfulness(question: str, context: str, answer: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Check if this answer is faithful to the context.
A faithful answer only uses information explicitly in the context.

Context: {context}
Answer: {answer}

Return JSON:
{{"faithful": true/false, "score": 0.0-1.0, "issues": ["issue1"] or []}}
JSON only:"""
        }],
        temperature=0.0,
        max_tokens=150,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

print(f"{'ID':<8} {'Score':>6} {'Faithful':>9}  Issues")
print("-" * 70)
for case in TEST_CASES:
    result = check_faithfulness(case["question"], case["context"], case["answer"])
    status = "✓" if result["faithful"] else "✗"
    issues = "; ".join(result["issues"][:1]) if result["issues"] else "none"
    print(f"{case['id']:<8} {result['score']:>6.2f} {status:>9}  {issues}")
```

**Expected output pattern:**
```
ID       Score  Faithful  Issues
----------------------------------------------------------------------
case_1    1.00         ✓  none
case_2    0.60         ✗  mobile app development not in context
case_3    0.50         ✗  number of co-founders not in context
case_4    1.00         ✓  none
case_5    0.50         ✗  circling Earth claim not in context
```

---

## Exercise 2 — End-to-end RAG evaluation suite (Main)

Build a complete evaluation suite that takes a RAG system's outputs and scores them on faithfulness, answer relevancy, and retrieval quality. Output a structured report.

```python
import os
import json
import numpy as np
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simulated RAG pipeline output — replace with your actual pipeline
RAG_OUTPUTS = [
    {
        "question": "What is prompt caching in the Anthropic API?",
        "answer": "Prompt caching lets you mark parts of your prompt as cacheable. When the same prefix is reused, Anthropic charges only 10% of the normal input token cost.",
        "contexts": [
            "Prompt caching reduces costs by storing prompt prefixes. Cached tokens cost 10% of the standard input price.",
            "Cache entries have a 5-minute TTL. The minimum cacheable prefix is 1024 tokens for Claude claude-sonnet-4-5 and 2048 for Haiku."
        ],
        "ground_truth": "Prompt caching stores frequently-used prompt prefixes and charges 10% of the normal input price when cache is hit."
    },
    {
        "question": "How does HNSW improve search speed?",
        "answer": "HNSW builds a hierarchical graph structure. Queries start at the top (coarse) layer and navigate down to the fine layer, examining far fewer nodes than brute-force search.",
        "contexts": [
            "HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm that builds a multilayer graph.",
            "Queries traverse from the coarse top layer to the precise bottom layer, skipping most of the dataset."
        ],
        "ground_truth": "HNSW uses a multilayer graph where queries navigate from coarse to fine layers, examining only a fraction of the total vectors."
    },
    {
        "question": "What is the maximum context window of Claude claude-sonnet-4-5?",
        "answer": "Claude claude-sonnet-4-5 has a 200,000 token context window, which is one of the largest available.",
        "contexts": [
            "Claude claude-sonnet-4-5 features a 200K token context window and strong performance on coding and analysis tasks.",
        ],
        "ground_truth": "Claude claude-sonnet-4-5 has a 200,000 token context window."
    },
]

def run_ragas_evaluation(rag_outputs: list[dict]) -> dict:
    """Run RAGAS evaluation and return scored results."""
    dataset = Dataset.from_dict({
        "question": [item["question"] for item in rag_outputs],
        "answer": [item["answer"] for item in rag_outputs],
        "contexts": [item["contexts"] for item in rag_outputs],
        "ground_truth": [item["ground_truth"] for item in rag_outputs],
    })

    scores = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy])
    return dict(scores)

def score_retrieval_quality(rag_outputs: list[dict]) -> dict:
    """Check if ground truth answer is covered by retrieved contexts."""
    coverage_scores = []
    for item in rag_outputs:
        context_text = " ".join(item["contexts"])
        gt_words = set(item["ground_truth"].lower().split())
        context_words = set(context_text.lower().split())
        overlap = len(gt_words & context_words) / len(gt_words) if gt_words else 0
        coverage_scores.append(overlap)

    return {
        "avg_coverage": sum(coverage_scores) / len(coverage_scores),
        "per_item": coverage_scores
    }

def generate_eval_report(rag_outputs: list[dict]) -> str:
    ragas_scores = run_ragas_evaluation(rag_outputs)
    retrieval = score_retrieval_quality(rag_outputs)

    lines = [
        "# RAG Evaluation Report",
        f"Evaluated {len(rag_outputs)} examples\n",
        "## RAGAS Metrics",
        f"Faithfulness:     {ragas_scores.get('faithfulness', 0):.3f}",
        f"Answer Relevancy: {ragas_scores.get('answer_relevancy', 0):.3f}\n",
        "## Retrieval Quality",
        f"Avg context coverage: {retrieval['avg_coverage']:.3f}\n",
        "## Per-Item Coverage",
    ]
    for i, (item, cov) in enumerate(zip(rag_outputs, retrieval["per_item"]), 1):
        lines.append(f"  {i}. [{cov:.2f}] {item['question'][:60]}...")

    return "\n".join(lines)

# Run the full evaluation
report = generate_eval_report(RAG_OUTPUTS)
print(report)
```

---

## Exercise 3 — Build a regression detector (Stretch)

Build a system that compares two versions of a prompt (v1 vs v2) on an eval set and flags regressions: cases where v2 scores lower than v1 by more than a threshold.

```python
import os
import json
from openai import OpenAI
from dataclasses import dataclass

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_V1 = "You are a helpful assistant. Answer the user's question using the provided context."

SYSTEM_V2 = """You are a precise assistant. Answer ONLY using information from the provided context.
If the answer is not in the context, say so explicitly.
Keep answers concise and cite specific parts of the context."""

EVAL_SET = [
    {
        "question": "What is our refund policy?",
        "context": "Refunds are accepted within 14 days of purchase. Items must be unused.",
        "ground_truth": "Refunds accepted within 14 days, items must be unused."
    },
    {
        "question": "How do I contact support?",
        "context": "Contact support at support@company.com or call 1-800-HELP from 9am-5pm EST.",
        "ground_truth": "Email support@company.com or call 1-800-HELP (9am-5pm EST)."
    },
    {
        "question": "What payment methods do you accept?",
        "context": "We accept Visa, Mastercard, and PayPal. Amex is not accepted.",
        "ground_truth": "Visa, Mastercard, and PayPal. Amex not accepted."
    },
]

@dataclass
class VersionScore:
    question: str
    v1_answer: str
    v2_answer: str
    v1_score: float
    v2_score: float
    delta: float
    regression: bool

def get_answer(question: str, context: str, system_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        temperature=0.0,
        max_tokens=150
    )
    return response.choices[0].message.content

def judge_answer(question: str, answer: str, ground_truth: str) -> float:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Rate this answer vs ground truth. Score 0.0-1.0 for accuracy.
Question: {question}
Ground truth: {ground_truth}
Answer: {answer}
Return JSON: {{"score": 0.0-1.0}}"""
        }],
        temperature=0.0,
        max_tokens=50,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)["score"]

def detect_regressions(
    eval_set: list[dict],
    system_v1: str,
    system_v2: str,
    regression_threshold: float = 0.15
) -> list[VersionScore]:
    results = []
    for item in eval_set:
        v1_ans = get_answer(item["question"], item["context"], system_v1)
        v2_ans = get_answer(item["question"], item["context"], system_v2)

        v1_score = judge_answer(item["question"], v1_ans, item["ground_truth"])
        v2_score = judge_answer(item["question"], v2_ans, item["ground_truth"])
        delta = v2_score - v1_score

        results.append(VersionScore(
            question=item["question"],
            v1_answer=v1_ans,
            v2_answer=v2_ans,
            v1_score=v1_score,
            v2_score=v2_score,
            delta=delta,
            regression=delta < -regression_threshold
        ))
    return results

results = detect_regressions(EVAL_SET, SYSTEM_V1, SYSTEM_V2)

regressions = [r for r in results if r.regression]
improvements = [r for r in results if r.delta > 0.15]

print(f"Eval set: {len(results)} examples")
print(f"Regressions: {len(regressions)}")
print(f"Improvements: {len(improvements)}")
print(f"Avg v1 score: {sum(r.v1_score for r in results)/len(results):.3f}")
print(f"Avg v2 score: {sum(r.v2_score for r in results)/len(results):.3f}")

if regressions:
    print("\nRegressed cases:")
    for r in regressions:
        print(f"  [{r.delta:+.2f}] {r.question}")
        print(f"    V1: {r.v1_answer[:80]}...")
        print(f"    V2: {r.v2_answer[:80]}...")
```

**Challenge:** Extend this to automatically generate a GitHub comment or Slack notification when regressions are detected. Use the CI/CD pattern from the RAGAS section to run this on pull requests.

---

[[05-human-evals]] | [[07-interview-questions]]
