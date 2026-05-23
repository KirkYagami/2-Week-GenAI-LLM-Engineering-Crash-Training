# Evaluation — LangGraph Research Agent

## Metrics for agent evaluation

| Metric | What it measures | How to measure |
|--------|-----------------|---------------|
| Quality score | Critic's assessment of final report | Already tracked in state |
| Loop count | How many writer iterations needed | `state["attempts"]` |
| Sub-question coverage | Did the report address all sub-questions? | LLM judge |
| Structural quality | Does the report have all required sections? | Rule-based |
| Total latency | End-to-end time including all LLM calls | `time.perf_counter()` |
| Token cost | Total tokens across all nodes | Sum from LangSmith or manual counting |

## Evaluation script

```python
# eval.py
import asyncio
import time
from agent import run

TEST_QUESTIONS = [
    "What are the main approaches to reducing hallucination in LLMs?",
    "How does retrieval-augmented generation compare to fine-tuning?",
    "What is the ReAct framework for AI agents and when should you use it?",
    "What are the key differences between LoRA and QLoRA fine-tuning?",
    "How do vector databases work and what makes a good retrieval system?",
]

def check_structure(report: str) -> dict:
    """Verify required sections are present."""
    report_lower = report.lower()
    return {
        "has_executive_summary": any(p in report_lower for p in ["executive summary", "overview"]),
        "has_key_findings": any(p in report_lower for p in ["key findings", "findings", "## "]),
        "has_conclusion": any(p in report_lower for p in ["conclusion", "in summary"]),
        "word_count_ok": len(report.split()) >= 250,
    }

async def run_evaluation() -> dict:
    all_scores = []
    all_attempts = []
    all_latencies = []
    structure_scores = []

    for question in TEST_QUESTIONS:
        print(f"\nResearching: {question[:60]}...")
        start = time.perf_counter()
        result = await run(question)
        elapsed = time.perf_counter() - start

        struct = check_structure(result["report"])
        struct_score = sum(struct.values()) / len(struct)

        all_scores.append(result["quality_score"])
        all_attempts.append(result["attempts"])
        all_latencies.append(elapsed)
        structure_scores.append(struct_score)

        print(f"  Quality: {result['quality_score']:.2f} | Attempts: {result['attempts']} | "
              f"Structure: {struct_score:.0%} | {elapsed:.1f}s")

    avg_q = sum(all_scores) / len(all_scores)
    avg_a = sum(all_attempts) / len(all_attempts)
    avg_l = sum(all_latencies) / len(all_latencies)
    avg_s = sum(structure_scores) / len(structure_scores)

    print(f"\n=== Research Agent Evaluation ===")
    print(f"Questions evaluated: {len(TEST_QUESTIONS)}")
    print(f"Avg quality score:   {avg_q:.2f}")
    print(f"Avg loop count:      {avg_a:.1f}")
    print(f"Avg structure score: {avg_s:.0%}")
    print(f"Avg latency:         {avg_l:.1f}s")

    return {"avg_quality": avg_q, "avg_attempts": avg_a, "avg_latency_s": avg_l}

if __name__ == "__main__":
    asyncio.run(run_evaluation())
```

## Comparing prompt versions

Run evaluation twice — once with the original prompts and once with modified prompts — and compare the quality scores. This is an A/B test for prompts:

```python
# Version A: current writer prompt
# Version B: more detailed structure requirements in writer prompt

# Change the writer node's system prompt and re-run
# If avg quality rises without increasing avg attempts, the new prompt wins
```

> [!success] Target metrics
> - Average quality score ≥ 0.78
> - Average loop count ≤ 2.0 (high-quality prompts need fewer revision loops)
> - Structure score ≥ 90% (nearly all reports have required sections)
> - Average latency < 30s (4 nodes × ~2s each × 1–3 loops)

---

[[03-advanced-features]] | [[05-deployment]]
