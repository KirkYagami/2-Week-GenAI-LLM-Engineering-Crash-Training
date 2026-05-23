# Cost Tracking

LLM API costs compound quickly. A classifier running at 1000 queries/day with a 500-token average prompt on GPT-4o adds up to thousands of dollars monthly. Cost tracking lets you measure this before it shows up on your bill, allocate costs to specific features or users, and catch regressions when a code change doubles your token usage.

## Learning objectives

- Count tokens accurately using tiktoken before making API calls
- Build a cost ledger that tracks spending by model, function, and user
- Set budget alerts that fire before you overspend
- Compare model cost vs performance tradeoffs
- Identify the highest-cost operations in your application

---

## Token counting with tiktoken

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens for a string using the model's actual tokenizer."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # Default for most OpenAI models
    return len(enc.encode(text))

def count_message_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """Count tokens for a messages list (including role tokens)."""
    enc = tiktoken.encoding_for_model(model)
    tokens_per_message = 3   # <|start|>{role}\n{content}<|end|>
    tokens_per_reply = 3     # reply primed with <|start|>assistant<|message|>

    total = tokens_per_reply
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(enc.encode(value))
    return total

# Test
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the difference between RAG and fine-tuning in 2 sentences."},
]
tokens = count_message_tokens(messages)
print(f"Estimated prompt tokens: {tokens}")
# → Estimated prompt tokens: ~45

# Cost estimate before the call
COST_PER_1M = {"gpt-4o-mini": {"input": 0.15, "output": 0.60}}
estimated_cost = tokens / 1e6 * COST_PER_1M["gpt-4o-mini"]["input"]
print(f"Estimated input cost: ${estimated_cost:.8f}")
```

---

## Cost ledger

```python
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PRICING = {
    "gpt-4o":          {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":     {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo":   {"input": 0.50,  "output": 1.50},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}

@dataclass
class CostEntry:
    model: str
    function_name: str
    user_id: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)

class CostLedger:
    def __init__(self, budget_usd: float = 10.0):
        self.entries: list[CostEntry] = []
        self.budget_usd = budget_usd
        self._by_function: dict[str, float] = defaultdict(float)
        self._by_user: dict[str, float] = defaultdict(float)
        self._by_model: dict[str, float] = defaultdict(float)

    def record(self, model: str, function_name: str, user_id: str,
               prompt_tokens: int, completion_tokens: int) -> float:
        price = PRICING.get(model, {"input": 0.15, "output": 0.60})
        cost = (prompt_tokens / 1e6 * price["input"]) + (completion_tokens / 1e6 * price["output"])

        entry = CostEntry(model, function_name, user_id, prompt_tokens, completion_tokens, cost)
        self.entries.append(entry)
        self._by_function[function_name] += cost
        self._by_user[user_id] += cost
        self._by_model[model] += cost

        total = sum(e.cost_usd for e in self.entries)
        if total > self.budget_usd * 0.9:
            print(f"[ALERT] 90% of budget used: ${total:.4f} / ${self.budget_usd}")
        return cost

    @property
    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.prompt_tokens + e.completion_tokens for e in self.entries)

    def report(self) -> None:
        print(f"\n{'=' * 50}")
        print(f"Total: ${self.total_cost:.6f} | {self.total_tokens:,} tokens")
        print(f"\nBy function:")
        for fn, cost in sorted(self._by_function.items(), key=lambda x: -x[1]):
            print(f"  {fn:<30} ${cost:.6f}")
        print(f"\nBy model:")
        for model, cost in sorted(self._by_model.items(), key=lambda x: -x[1]):
            print(f"  {model:<25} ${cost:.6f}")
        print(f"\nBy user:")
        for user, cost in sorted(self._by_user.items(), key=lambda x: -x[1]):
            print(f"  {user:<20} ${cost:.6f}")

# Usage with the OpenAI client
ledger = CostLedger(budget_usd=1.00)

def tracked_completion(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    function_name: str = "unknown",
    user_id: str = "anon",
    **kwargs,
) -> str:
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    ledger.record(
        model=model,
        function_name=function_name,
        user_id=user_id,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    return response.choices[0].message.content

# Test calls
for i in range(3):
    tracked_completion(
        [{"role": "user", "content": f"Question {i}: What is machine learning?"}],
        model="gpt-4o-mini",
        function_name="answer_question",
        user_id=f"user_{i % 2}",
    )

ledger.report()
```

---

## Model cost vs performance comparison

```python
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODELS_TO_TEST = ["gpt-4o-mini", "gpt-4o"]
TEST_PROMPT = [
    {"role": "system", "content": "You are a sentiment classifier. Reply with one word: positive, negative, or neutral."},
    {"role": "user", "content": "The new product update completely broke my workflow. I'm furious."},
]
EXPECTED = "negative"

print(f"{'Model':<20} {'Tokens':>8} {'Cost($)':>12} {'Latency(ms)':>14} {'Correct':>8}")
print("-" * 65)

for model in MODELS_TO_TEST:
    start = time.perf_counter()
    response = client.chat.completions.create(model=model, messages=TEST_PROMPT, temperature=0.0, max_tokens=5)
    latency_ms = (time.perf_counter() - start) * 1000

    usage = response.usage
    price = PRICING.get(model, {"input": 0.15, "output": 0.60})
    cost = (usage.prompt_tokens / 1e6 * price["input"]) + (usage.completion_tokens / 1e6 * price["output"])
    answer = response.choices[0].message.content.strip().lower()
    correct = EXPECTED in answer

    print(f"{model:<20} {usage.total_tokens:>8} {cost:>12.8f} {latency_ms:>14.0f} {'✓' if correct else '✗':>8}")
```

> [!tip] gpt-4o-mini handles most classification and extraction tasks as well as gpt-4o at 1/15th the cost
> The cost ratio between gpt-4o and gpt-4o-mini is approximately 15–25x for input tokens. For high-volume, structured tasks (classification, extraction, summarization), measure gpt-4o-mini's accuracy before defaulting to the larger model.

---

[[02-langsmith]] | [[04-latency-optimization]]
