# Evaluation — AI Writing Assistant

## What to measure

Writing quality is partly subjective, but these objective metrics give you numbers for your portfolio:

| Metric | What it captures | How to measure |
|--------|-----------------|---------------|
| Style adherence | Does the output match the requested style? | LLM-as-judge (0–1 scale) |
| Instruction following | Did the output cover all requested sections? | Section count check |
| Word count accuracy | Is the output close to the requested length? | `abs(actual - target) / target` |
| Coherence | Does the text flow logically? | LLM-as-judge |
| Latency | How long does each step take? | `time.perf_counter()` |

## LLM-as-judge evaluation

```python
# eval.py
import os
import time
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
judge = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TEST_CASES = [
    {"topic": "Benefits of meditation", "style": "professional", "target_words": 300},
    {"topic": "Getting started with Python", "style": "casual", "target_words": 400},
    {"topic": "Quantum computing explained", "style": "technical", "target_words": 500},
    {"topic": "Why you should exercise daily", "style": "persuasive", "target_words": 350},
]

def judge_style(text: str, style: str) -> float:
    """Score 0.0–1.0: does the text match the requested style?"""
    resp = judge.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Rate how well this text matches a '{style}' writing style on a scale of 0.0 to 1.0.\n"
                f"0.0 = completely wrong style, 1.0 = perfect match.\n"
                f"Respond with only a number.\n\nText:\n{text[:800]}"
            ),
        }],
        temperature=0.0,
        max_tokens=5,
    )
    try:
        return float(resp.choices[0].message.content.strip())
    except ValueError:
        return 0.5

def evaluate_pipeline(base_url: str = "http://localhost:8000") -> dict:
    style_scores = []
    word_count_errors = []
    latencies = []

    with httpx.Client(timeout=60.0) as client:
        for case in TEST_CASES:
            start = time.perf_counter()
            resp = client.post(f"{base_url}/write", json=case)
            latency = (time.perf_counter() - start) * 1000

            if resp.status_code != 200:
                print(f"FAIL: {case['topic'][:40]} → {resp.status_code}")
                continue

            data = resp.json()
            final = data["final"]
            actual_words = len(final.split())
            word_error = abs(actual_words - case["target_words"]) / case["target_words"]
            style_score = judge_style(final, case["style"])

            style_scores.append(style_score)
            word_count_errors.append(word_error)
            latencies.append(latency)

            print(f"  style={style_score:.2f} | word_error={word_error:.0%} | {latency:.0f}ms | {case['topic'][:40]}")

    avg_style = sum(style_scores) / len(style_scores) if style_scores else 0
    avg_word_error = sum(word_count_errors) / len(word_count_errors) if word_count_errors else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    print(f"\n=== Writing Assistant Evaluation ===")
    print(f"Avg style score:     {avg_style:.2f} / 1.0")
    print(f"Avg word count error: {avg_word_error:.0%}")
    print(f"P95 latency:         {p95_latency:.0f}ms")

    return {"avg_style_score": avg_style, "avg_word_error": avg_word_error, "p95_ms": p95_latency}

if __name__ == "__main__":
    evaluate_pipeline()
```

## Style adherence by style type

Run each style separately to identify which styles the pipeline handles better or worse:

```python
STYLE_TEST = [
    {"topic": "Remote work", "style": "professional"},
    {"topic": "Remote work", "style": "casual"},
    {"topic": "Remote work", "style": "technical"},
    {"topic": "Remote work", "style": "persuasive"},
]
```

Using the same topic across styles lets you compare directly without topic difficulty being a confound.

## Step-level latency breakdown

```python
import time

async def timed_pipeline(topic: str, style: str) -> dict:
    timings = {}
    llm = app.state.llm
    parser = StrOutputParser()

    t0 = time.perf_counter()
    outline = await (OUTLINE_PROMPT | llm | parser).ainvoke({"topic": topic, "style": style, "audience": "general", "section_count": 3})
    timings["outline_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    draft = await (DRAFT_PROMPT | llm | parser).ainvoke({"outline": outline, "style": style, "target_words": 300})
    timings["draft_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    refined = await (REFINE_PROMPT | llm | parser).ainvoke({"draft": draft})
    timings["refine_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    final = await (STYLE_CHECK_PROMPT | llm | parser).ainvoke({"text": refined, "style": style})
    timings["style_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = sum(timings.values())
    return timings
```

> [!success] Target metrics
> - Style score: ≥ 0.80 for all four style options
> - Word count error: ≤ 20% of target
> - Total pipeline latency: < 15s for 400-word output (4 sequential LLM calls)

---

[[03-advanced-features]] | [[05-deployment]]
