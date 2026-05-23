# Observability

Logging individual traces is necessary but insufficient. Observability means you can answer operational questions at scale: What's my p95 latency this week? Which user is responsible for 40% of my token spend? Did this model upgrade increase or decrease my error rate? This note covers metric aggregation, alerting, and dashboard design for LLM applications.

## Learning objectives

- Define the key metrics for LLM observability: latency, cost, error rate, quality
- Build a metrics aggregator that computes percentiles and rates
- Implement budget alerts before costs hit your limit
- Design a dashboard structure for LLM applications
- Set up Prometheus-compatible metrics counters

---

## The four LLM observability signals

```python
# 1. LATENCY: How long does each call take?
#    Key metrics: p50, p95, p99, time-to-first-token
#    Alert when: p95 exceeds SLA (e.g., > 5 seconds)

# 2. COST: How much are we spending?
#    Key metrics: cost per request, cost per user, daily/monthly total
#    Alert when: daily spend exceeds budget threshold

# 3. ERROR RATE: How often do calls fail?
#    Key metrics: % API errors, % timeout errors, % malformed outputs
#    Alert when: error rate > 1% sustained over 5 minutes

# 4. QUALITY: Are outputs good?
#    Key metrics: LLM-as-judge scores, user feedback rate, task success rate
#    Alert when: quality score drops > 5% vs baseline
```

---

## Metrics aggregator

```python
import time
import math
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class MetricPoint:
    timestamp: float
    value: float
    labels: dict

class MetricsStore:
    def __init__(self):
        self._metrics: dict[str, list[MetricPoint]] = defaultdict(list)

    def record(self, metric_name: str, value: float, labels: dict = None) -> None:
        self._metrics[metric_name].append(
            MetricPoint(timestamp=time.time(), value=value, labels=labels or {})
        )

    def _recent(self, metric_name: str, window_seconds: int = 3600) -> list[float]:
        cutoff = time.time() - window_seconds
        return [p.value for p in self._metrics[metric_name] if p.timestamp > cutoff]

    def percentile(self, metric_name: str, p: float, window_seconds: int = 3600) -> float:
        values = sorted(self._recent(metric_name, window_seconds))
        if not values:
            return 0.0
        idx = math.ceil(p / 100 * len(values)) - 1
        return values[max(0, idx)]

    def mean(self, metric_name: str, window_seconds: int = 3600) -> float:
        values = self._recent(metric_name, window_seconds)
        return sum(values) / len(values) if values else 0.0

    def count(self, metric_name: str, window_seconds: int = 3600) -> int:
        return len(self._recent(metric_name, window_seconds))

    def sum(self, metric_name: str, window_seconds: int = 3600) -> float:
        return sum(self._recent(metric_name, window_seconds))

    def report(self) -> None:
        print("\n=== Metrics Report (last hour) ===")
        for metric, points in self._metrics.items():
            values = [p.value for p in points]
            if not values:
                continue
            print(f"\n{metric}:")
            print(f"  count={len(values)}, sum={sum(values):.4f}")
            sorted_vals = sorted(values)
            p50_idx = max(0, math.ceil(50 / 100 * len(sorted_vals)) - 1)
            p95_idx = max(0, math.ceil(95 / 100 * len(sorted_vals)) - 1)
            print(f"  p50={sorted_vals[p50_idx]:.2f}, p95={sorted_vals[p95_idx]:.2f}")

# Instrument your LLM wrapper
metrics = MetricsStore()

def instrumented_call(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.perf_counter()
    error = False
    try:
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=200)
        reply = response.choices[0].message.content
        usage = response.usage

        metrics.record("llm.latency_ms", (time.perf_counter() - start) * 1000, {"model": model})
        metrics.record("llm.prompt_tokens", usage.prompt_tokens, {"model": model})
        metrics.record("llm.completion_tokens", usage.completion_tokens, {"model": model})

        price = {"gpt-4o-mini": 0.15, "gpt-4o": 2.50}.get(model, 0.15)
        cost = usage.total_tokens / 1e6 * price
        metrics.record("llm.cost_usd", cost, {"model": model})
        metrics.record("llm.requests_total", 1, {"model": model, "status": "success"})

        return reply

    except Exception as e:
        error = True
        metrics.record("llm.requests_total", 1, {"model": model, "status": "error"})
        metrics.record("llm.latency_ms", (time.perf_counter() - start) * 1000, {"model": model})
        raise

# Run some calls and view metrics
import os
questions = ["What is Python?", "What is RAG?", "What is fine-tuning?"]
for q in questions:
    instrumented_call([{"role": "user", "content": q}])

metrics.report()
print(f"\np95 latency: {metrics.percentile('llm.latency_ms', 95):.0f}ms")
print(f"Total cost: ${metrics.sum('llm.cost_usd'):.6f}")
```

---

## Budget alert system

```python
import os
import time
from threading import Timer
from typing import Callable

class BudgetAlert:
    def __init__(
        self,
        daily_budget_usd: float,
        alert_thresholds: list[float] = [0.5, 0.75, 0.9, 1.0],
        on_alert: Callable[[str], None] = print,
    ):
        self.daily_budget = daily_budget_usd
        self.thresholds = sorted(alert_thresholds)
        self.on_alert = on_alert
        self._triggered: set[float] = set()
        self._daily_spend = 0.0
        self._day_start = time.time()

    def record_spend(self, amount_usd: float) -> None:
        # Reset daily counter if day has passed
        if time.time() - self._day_start > 86400:
            self._daily_spend = 0.0
            self._day_start = time.time()
            self._triggered.clear()

        self._daily_spend += amount_usd
        pct = self._daily_spend / self.daily_budget

        for threshold in self.thresholds:
            if pct >= threshold and threshold not in self._triggered:
                self._triggered.add(threshold)
                self.on_alert(
                    f"[BUDGET ALERT] {threshold*100:.0f}% of daily budget used: "
                    f"${self._daily_spend:.4f} / ${self.daily_budget:.2f}"
                )

# Example
def send_alert(message: str) -> None:
    print(f"🚨 {message}")
    # In production: send to Slack, PagerDuty, email, etc.

alert = BudgetAlert(daily_budget_usd=5.00, on_alert=send_alert)
for cost in [1.0, 1.5, 1.0, 0.75, 0.5]:  # Simulated spend over the day
    alert.record_spend(cost)
```

---

## Prometheus metrics (production-grade)

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
llm_requests = Counter("llm_requests_total", "Total LLM API calls", ["model", "function", "status"])
llm_latency = Histogram("llm_latency_seconds", "LLM call latency", ["model"], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
llm_tokens = Counter("llm_tokens_total", "Total tokens consumed", ["model", "type"])
llm_cost = Counter("llm_cost_usd_total", "Total cost in USD", ["model"])

def prometheus_instrumented_call(messages: list, model: str = "gpt-4o-mini", function_name: str = "unknown") -> str:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=200)
        latency = time.perf_counter() - start

        llm_requests.labels(model=model, function=function_name, status="success").inc()
        llm_latency.labels(model=model).observe(latency)
        llm_tokens.labels(model=model, type="prompt").inc(response.usage.prompt_tokens)
        llm_tokens.labels(model=model, type="completion").inc(response.usage.completion_tokens)

        return response.choices[0].message.content

    except Exception as e:
        llm_requests.labels(model=model, function=function_name, status="error").inc()
        raise

# Start Prometheus metrics server on port 8001
# start_http_server(8001)
# Metrics available at http://localhost:8001/metrics
```

> [!info] Prometheus + Grafana is the standard stack for LLM observability
> Export metrics from `prometheus_client`, scrape with Prometheus, visualize in Grafana. Use `histogram_quantile(0.95, ...)` in Prometheus for p95 latency. Alert with AlertManager when thresholds are crossed.

---

[[04-latency-optimization]] | [[06-practice-exercises]]
