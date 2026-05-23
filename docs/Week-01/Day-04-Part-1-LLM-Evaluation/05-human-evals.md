# Human Evaluations

Automated metrics tell you whether the model is faithful and relevant. Human evaluation tells you whether users actually find the output helpful. Both matter — and calibrated human evals are harder to set up than most teams realize.

## Learning objectives

- Design a rubric-based human evaluation protocol
- Implement pairwise preference evaluation (A/B comparison)
- Measure and improve inter-rater agreement using Cohen's kappa
- Build a lightweight human eval pipeline in Python

---

## Why you still need humans

LLM-as-judge is powerful but has known failure modes:

- **Verbosity bias** — longer answers score higher even when less accurate
- **Self-preference** — GPT-4o rates GPT-4o outputs higher; Claude rates Claude outputs higher
- **Anchoring** — the order of presented options influences the score
- **Subtle quality** — helpfulness, trust, and tone require lived human experience

> [!quote] "Automated metrics are a proxy for the thing you actually care about, which is whether users succeed at their task." — Liang et al., HELM benchmark paper

Human evaluation is the ground truth. Automated metrics are approximations of it.

---

## Rubric design

A bad rubric: "Rate the answer 1–5." A good rubric: specific dimensions with anchored descriptions.

```python
from dataclasses import dataclass, field
from typing import Optional
import json

@dataclass
class RubricDimension:
    name: str
    description: str
    anchors: dict[int, str]  # score → description

# Production-grade rubric for a RAG Q&A system
QA_RUBRIC = [
    RubricDimension(
        name="correctness",
        description="Is the factual content of the answer accurate?",
        anchors={
            1: "Contains significant factual errors",
            2: "Mostly incorrect or misleading",
            3: "Partially correct with notable gaps",
            4: "Mostly correct, minor inaccuracies",
            5: "Fully accurate and verifiable"
        }
    ),
    RubricDimension(
        name="completeness",
        description="Does the answer cover all aspects of the question?",
        anchors={
            1: "Misses the main point of the question",
            2: "Addresses only one part of a multi-part question",
            3: "Covers the main point, misses secondary aspects",
            4: "Covers most aspects, minor omissions",
            5: "Complete answer addressing all aspects"
        }
    ),
    RubricDimension(
        name="clarity",
        description="Is the answer easy to understand and well-structured?",
        anchors={
            1: "Confusing, contradictory, or unreadable",
            2: "Hard to follow, poor organization",
            3: "Clear but could be better structured",
            4: "Clear and well-organized",
            5: "Exceptionally clear, easy to skim and act on"
        }
    ),
    RubricDimension(
        name="citation_quality",
        description="Are sources cited appropriately and accurately?",
        anchors={
            1: "No citations where needed, or all citations wrong",
            2: "Citations present but mostly irrelevant",
            3: "Some correct citations, some missing",
            4: "Good citations, minor issues",
            5: "All claims backed by accurate, specific citations"
        }
    ),
]

def format_rubric_for_annotator(rubric: list[RubricDimension]) -> str:
    lines = ["## Scoring rubric\n"]
    for dim in rubric:
        lines.append(f"### {dim.name.replace('_', ' ').title()}")
        lines.append(f"{dim.description}\n")
        for score, desc in sorted(dim.anchors.items()):
            lines.append(f"- **{score}** — {desc}")
        lines.append("")
    return "\n".join(lines)

print(format_rubric_for_annotator(QA_RUBRIC[:2]))
```

---

## Pairwise preference evaluation

Absolute scores are noisy; humans are bad at calibrating across sessions. Pairwise comparisons ("which is better, A or B?") are more reliable and reveal real differences.

```python
@dataclass
class PairwiseTask:
    question: str
    answer_a: str
    answer_b: str
    context: Optional[str] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class PairwiseResult:
    task_id: str
    rater_id: str
    preference: str        # "A", "B", or "tie"
    confidence: int        # 1 (slight) to 3 (clear)
    reasoning: str
    dimension_scores: dict[str, dict[str, int]]  # {"A": {"correctness": 4}, "B": {...}}

def create_pairwise_prompt(task: PairwiseTask) -> str:
    context_section = f"\n\nContext available to the assistant:\n{task.context}" if task.context else ""
    return f"""Compare these two answers to the same question. Evaluate them on correctness, completeness, and clarity.

Question: {task.question}{context_section}

Answer A:
{task.answer_a}

Answer B:
{task.answer_b}

Which answer is better overall? Respond with:
- preference: "A", "B", or "tie"
- confidence: 1 (slight preference) to 3 (clearly better)
- reasoning: one sentence
- dimension_scores: {{"A": {{"correctness": 1-5, "completeness": 1-5, "clarity": 1-5}}, "B": {{...}}}}

Return JSON only."""

def llm_pairwise_judge(task: PairwiseTask, task_id: str) -> PairwiseResult:
    from openai import OpenAI
    client = OpenAI(api_key=__import__("os").getenv("OPENAI_API_KEY"))

    # Randomize order to reduce position bias — swap A and B 50% of the time
    import random
    swapped = random.random() > 0.5
    if swapped:
        task_for_eval = PairwiseTask(
            question=task.question,
            answer_a=task.answer_b,
            answer_b=task.answer_a,
            context=task.context
        )
    else:
        task_for_eval = task

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": create_pairwise_prompt(task_for_eval)}],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)

    # Un-swap if needed
    if swapped and result["preference"] in ("A", "B"):
        result["preference"] = "B" if result["preference"] == "A" else "A"

    return PairwiseResult(
        task_id=task_id,
        rater_id="gpt-4o",
        preference=result["preference"],
        confidence=result["confidence"],
        reasoning=result["reasoning"],
        dimension_scores=result.get("dimension_scores", {})
    )
```

---

## Inter-rater agreement

When multiple humans (or LLM judges) rate the same examples, they will disagree. Inter-rater agreement measures how often they agree — and whether agreement is better than chance.

```python
from collections import Counter
import math

def cohens_kappa(ratings_a: list, ratings_b: list) -> float:
    """
    Cohen's kappa for two raters on the same items.
    kappa = (P_o - P_e) / (1 - P_e)
    where P_o = observed agreement, P_e = expected agreement by chance.
    """
    assert len(ratings_a) == len(ratings_b), "Same number of items required"
    n = len(ratings_a)
    categories = sorted(set(ratings_a) | set(ratings_b))

    # Observed agreement
    p_observed = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b) / n

    # Expected agreement by chance
    counts_a = Counter(ratings_a)
    counts_b = Counter(ratings_b)
    p_expected = sum(
        (counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n)
        for c in categories
    )

    if p_expected == 1.0:
        return 1.0

    return (p_observed - p_expected) / (1 - p_expected)

def interpret_kappa(kappa: float) -> str:
    if kappa < 0:
        return "Worse than chance"
    if kappa < 0.20:
        return "Slight agreement"
    if kappa < 0.40:
        return "Fair agreement"
    if kappa < 0.60:
        return "Moderate agreement"
    if kappa < 0.80:
        return "Substantial agreement"
    return "Almost perfect agreement"

# Simulate two raters scoring 10 answers on correctness (1-5)
rater_1 = [5, 4, 3, 5, 2, 4, 3, 5, 4, 3]
rater_2 = [5, 4, 2, 5, 2, 3, 3, 4, 4, 3]

kappa = cohens_kappa(rater_1, rater_2)
print(f"Cohen's kappa: {kappa:.3f} — {interpret_kappa(kappa)}")
# Cohen's kappa: 0.712 — Substantial agreement

# Target: kappa > 0.6 before using raters for production evaluation
# If kappa < 0.4: run calibration session — raters review disagreements together
```

> [!tip] Calibration beats training
> The fastest way to improve inter-rater agreement is a 30-minute calibration session where raters score the same 10 examples independently, then discuss disagreements. Calibrated raters reach kappa > 0.7 much faster than raters who read a rubric and start scoring.

---

## Building a lightweight human eval pipeline

```python
import csv
import os
from datetime import datetime

class HumanEvalPipeline:
    """
    Generates annotation tasks as CSV for human raters.
    Collects results and computes aggregate scores.
    """

    def __init__(self, output_dir: str = "eval_tasks"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_tasks(
        self,
        eval_examples: list[dict],
        dimensions: list[str]
    ) -> str:
        """Write annotation task CSV. One row per example."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"tasks_{timestamp}.csv")

        fieldnames = ["id", "question", "answer", "context"] + dimensions + ["notes"]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ex in eval_examples:
                row = {
                    "id": ex["id"],
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "context": ex.get("context", ""),
                }
                row.update({dim: "" for dim in dimensions})
                row["notes"] = ""
                writer.writerow(row)

        print(f"Tasks written to: {path}")
        return path

    def load_results(self, results_path: str) -> list[dict]:
        with open(results_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def aggregate_results(
        self,
        results: list[dict],
        dimensions: list[str]
    ) -> dict:
        scores = {dim: [] for dim in dimensions}

        for row in results:
            for dim in dimensions:
                val = row.get(dim, "").strip()
                if val.isdigit():
                    scores[dim].append(int(val))

        summary = {}
        for dim, vals in scores.items():
            if vals:
                summary[dim] = {
                    "mean": sum(vals) / len(vals),
                    "n": len(vals),
                    "pass_rate": sum(1 for v in vals if v >= 4) / len(vals)
                }

        return summary

# Usage
pipeline = HumanEvalPipeline(output_dir="human_evals")

examples = [
    {"id": "q1", "question": "What is our return policy?",
     "answer": "Returns are accepted within 30 days.", "context": "Policy doc text..."},
    {"id": "q2", "question": "How do I cancel my subscription?",
     "answer": "Go to Settings → Subscription → Cancel.", "context": "Help center text..."},
]

task_path = pipeline.generate_tasks(
    examples,
    dimensions=["correctness", "completeness", "clarity"]
)
# Raters open the CSV, fill in scores 1-5, save

# After collection:
# results = pipeline.load_results("human_evals/tasks_20240101_120000_filled.csv")
# summary = pipeline.aggregate_results(results, ["correctness", "completeness", "clarity"])
# print(summary)
```

---

## When to use human vs automated evaluation

| Signal | Use automated eval | Use human eval |
|--------|------------------|---------------|
| Correctness (objective) | ✓ Reference-based metrics | ✓ Spot check |
| Faithfulness | ✓ NLI, LLM-as-judge | ✓ For high-stakes content |
| Helpfulness | ✗ Hard to automate | ✓ Required |
| Tone / trust | ✗ Unreliable | ✓ Required |
| Regression detection | ✓ Fast automated suite | ✗ Too slow |
| New capability launch | ✗ No baseline | ✓ Required |
| A/B comparison | ✓ Pairwise LLM-judge | ✓ Ground truth |

> [!success] Practical rule
> Automate 90% of your evaluation. Use human eval for: every major prompt change, every new feature launch, and random 1% sampling of production traffic. This coverage costs ~$50–200/month in annotation time and catches the things automated metrics miss.

---

[[04-relevance-metrics]] | [[06-practice-exercises]]
