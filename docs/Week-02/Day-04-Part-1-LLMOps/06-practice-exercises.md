# Practice Exercises — LLMOps

---

## Exercise 1 — Cost comparison (Warm-up)

Build a tool that runs the same prompt against three models and produces a cost-quality tradeoff table.

```python
import os
import time
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}

TEST_CASES = [
    {
        "id": "sentiment",
        "prompt": "Classify the sentiment of: 'This product is absolutely terrible, waste of money!' Reply: positive, negative, or neutral.",
        "expected": "negative",
    },
    {
        "id": "entity",
        "prompt": "Extract the company name from: 'Sarah Chen from OpenAI announced the new model.' Reply with just the company name.",
        "expected": "openai",
    },
    {
        "id": "math",
        "prompt": "What is 15% of 240? Reply with just the number.",
        "expected": "36",
    },
]

def benchmark_model(model: str) -> dict:
    costs, latencies, correct = [], [], 0

    for tc in TEST_CASES:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": tc["prompt"]}],
            temperature=0.0,
            max_tokens=20,
        )
        latency = (time.perf_counter() - start) * 1000

        usage = response.usage
        price = PRICING.get(model, {"input": 0.15, "output": 0.60})
        cost = (usage.prompt_tokens / 1e6 * price["input"]) + (usage.completion_tokens / 1e6 * price["output"])
        costs.append(cost)
        latencies.append(latency)

        answer = response.choices[0].message.content.strip().lower()
        if tc["expected"].lower() in answer:
            correct += 1

    return {
        "model": model,
        "accuracy": f"{correct}/{len(TEST_CASES)}",
        "avg_cost_usd": sum(costs) / len(costs),
        "total_cost_usd": sum(costs),
        "avg_latency_ms": sum(latencies) / len(latencies),
    }

print(f"{'Model':<20} {'Accuracy':>10} {'Avg Cost($)':>14} {'Avg Latency(ms)':>18}")
print("-" * 65)

results = []
for model in PRICING.keys():
    r = benchmark_model(model)
    results.append(r)
    print(f"{r['model']:<20} {r['accuracy']:>10} {r['avg_cost_usd']:>14.8f} {r['avg_latency_ms']:>18.0f}")

# Cost ratio
if len(results) >= 2:
    ratio = results[1]["avg_cost_usd"] / results[0]["avg_cost_usd"]
    print(f"\nCost ratio (gpt-4o / gpt-4o-mini): {ratio:.1f}x")
```

---

## Exercise 2 — LLM monitoring dashboard (Main)

Build a real-time monitoring system that tracks cost, latency, and error rate across a simulated production workload.

```python
import os
import time
import json
import random
import math
from openai import OpenAI
from collections import defaultdict
from dataclasses import dataclass, field

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class CallRecord:
    timestamp: float
    model: str
    function_name: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    success: bool
    error_type: str = ""

class LLMMonitor:
    def __init__(self, budget_daily_usd: float = 2.00):
        self.records: list[CallRecord] = []
        self.budget = budget_daily_usd

    def record(self, rec: CallRecord) -> None:
        self.records.append(rec)
        self._check_budget()

    def _check_budget(self) -> None:
        total = sum(r.cost_usd for r in self.records)
        if total > self.budget * 0.8:
            pct = total / self.budget * 100
            print(f"  [ALERT] Budget {pct:.0f}% used: ${total:.4f}/${self.budget:.2f}")

    def _percentile(self, values: list[float], p: int) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = max(0, math.ceil(p / 100 * len(sorted_vals)) - 1)
        return sorted_vals[idx]

    def dashboard(self) -> None:
        if not self.records:
            print("No records yet.")
            return

        total_calls = len(self.records)
        successes = [r for r in self.records if r.success]
        errors = [r for r in self.records if not r.success]

        latencies = [r.latency_ms for r in successes]
        total_cost = sum(r.cost_usd for r in self.records)
        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in self.records)

        print(f"\n{'=' * 55}")
        print(f"LLM MONITORING DASHBOARD")
        print(f"{'=' * 55}")
        print(f"Total calls:    {total_calls}")
        print(f"Success rate:   {len(successes)/total_calls*100:.1f}%")
        print(f"Error rate:     {len(errors)/total_calls*100:.1f}%")
        print(f"\nLatency (successful calls):")
        print(f"  p50: {self._percentile(latencies, 50):.0f}ms")
        print(f"  p95: {self._percentile(latencies, 95):.0f}ms")
        print(f"  p99: {self._percentile(latencies, 99):.0f}ms")
        print(f"\nCost:")
        print(f"  Total:     ${total_cost:.6f}")
        print(f"  Per call:  ${total_cost/total_calls:.8f}")
        print(f"  Budget:    {total_cost/self.budget*100:.1f}% used")
        print(f"\nTokens:")
        print(f"  Total:     {total_tokens:,}")
        print(f"  Per call:  {total_tokens//total_calls:,}")

        # By function
        by_fn: dict[str, list[CallRecord]] = defaultdict(list)
        for r in self.records:
            by_fn[r.function_name].append(r)
        print(f"\nBy function:")
        for fn, recs in sorted(by_fn.items()):
            fn_cost = sum(r.cost_usd for r in recs)
            fn_errors = sum(1 for r in recs if not r.success)
            print(f"  {fn:<25} calls={len(recs):3d} cost=${fn_cost:.6f} errors={fn_errors}")

monitor = LLMMonitor(budget_daily_usd=0.50)

PRICING = {"gpt-4o-mini": {"input": 0.15, "output": 0.60}}
FUNCTIONS = ["classify_ticket", "extract_entities", "generate_response", "summarize"]
QUESTIONS = ["What is AI?", "Summarize this text.", "Classify this ticket.", "Extract names from this."]

print("Simulating 20 LLM calls...")
for i in range(20):
    fn = random.choice(FUNCTIONS)
    q = random.choice(QUESTIONS)
    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": q}],
            max_tokens=50,
        )
        latency = (time.perf_counter() - start) * 1000
        usage = response.usage
        cost = (usage.prompt_tokens / 1e6 * 0.15) + (usage.completion_tokens / 1e6 * 0.60)

        monitor.record(CallRecord(
            timestamp=time.time(), model="gpt-4o-mini", function_name=fn,
            latency_ms=latency, prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens, cost_usd=cost, success=True,
        ))
    except Exception as e:
        monitor.record(CallRecord(
            timestamp=time.time(), model="gpt-4o-mini", function_name=fn,
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=0, completion_tokens=0, cost_usd=0,
            success=False, error_type=type(e).__name__,
        ))

monitor.dashboard()
```

---

[[05-observability]] | [[07-interview-questions]]
