# Practice Exercises — How LLMs Work

Three levels: warm-up (tweak a parameter), main (build a mini-pipeline), stretch (evaluate and measure).

---

## Warm-up: Temperature explorer

**Goal:** Build intuition for how temperature affects output diversity.

```python
import openai
import os

client = openai.OpenAI()  # reads OPENAI_API_KEY from env

PROMPT = "Complete this sentence in an interesting way: 'The scientist opened the box and found'"

def generate_at_temperature(prompt: str, temperature: float, n: int = 3) -> list[str]:
    """Generate n responses at the given temperature."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheaper for experiments
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=50,
        n=n,  # generate n candidates in one call
    )
    return [choice.message.content for choice in response.choices]

# Run it
for temp in [0.0, 0.5, 1.0, 1.5]:
    outputs = generate_at_temperature(PROMPT, temperature=temp)
    print(f"\n=== Temperature {temp} ===")
    for i, out in enumerate(outputs, 1):
        print(f"  {i}. {out}")
```

**Questions to answer:**
- At temperature 0, are all three outputs identical?
- At temperature 1.5, do outputs start to degrade in coherence?
- Which temperature produces the most useful output for this task?

---

## Warm-up: Token counter

**Goal:** Count tokens for a realistic RAG prompt across OpenAI and Anthropic.

```python
import tiktoken
import anthropic
import os

# OpenAI token count
def count_openai_tokens(messages: list[dict], model: str = "gpt-4o") -> dict:
    enc = tiktoken.get_encoding("o200k_base")
    counts = {}
    for msg in messages:
        role = msg["role"]
        tokens = len(enc.encode(msg["content"])) + 4  # +4 for message overhead
        counts[role] = tokens
    counts["total"] = sum(counts.values()) + 2  # +2 for reply priming
    return counts

# Anthropic token count
def count_anthropic_tokens(messages: list[dict], system: str = "") -> int:
    client = anthropic.Anthropic()
    response = client.messages.count_tokens(
        model="claude-sonnet-4-6",
        system=system,
        messages=messages,
    )
    return response.input_tokens

# A realistic RAG prompt
system_prompt = """You are a helpful assistant. Answer questions based only on the provided context.
If the answer is not in the context, say "I don't know."
Always cite the specific section you're drawing from."""

retrieved_context = """
Section 2.1: Return Policy
Products can be returned within 30 days of purchase with original receipt.
Items must be in original condition. Digital downloads are non-refundable.

Section 2.2: Exchange Policy
Exchanges are accepted within 60 days. No receipt required for exchanges under $50.
""" * 3  # simulate 3 retrieved chunks

user_query = "Can I exchange a product without a receipt?"

messages = [
    {"role": "user", "content": f"Context:\n{retrieved_context}\n\nQuestion: {user_query}"},
]

openai_counts = count_openai_tokens(
    [{"role": "system", "content": system_prompt}] + messages
)
anthropic_count = count_anthropic_tokens(messages, system=system_prompt)

print("OpenAI token breakdown:", openai_counts)
print(f"Anthropic total: {anthropic_count} tokens")

# Estimate cost
input_tokens = openai_counts["total"]
cost_gpt4o = (input_tokens / 1_000_000) * 2.50 + (200 / 1_000_000) * 10.00
cost_claude = (anthropic_count / 1_000_000) * 3.00 + (200 / 1_000_000) * 15.00
print(f"\nEstimated cost (GPT-4o): ${cost_gpt4o:.5f}")
print(f"Estimated cost (Claude Sonnet 4.6): ${cost_claude:.5f}")
```

---

## Main exercise: Sampling strategy comparator

**Goal:** Systematically compare sampling strategies and measure their effect on output quality for a specific task type.

```python
import openai
import json
from dataclasses import dataclass

client = openai.OpenAI()

@dataclass
class SamplingConfig:
    name: str
    temperature: float
    top_p: float
    max_tokens: int

CONFIGS = [
    SamplingConfig("greedy",      temperature=0.0, top_p=1.0, max_tokens=200),
    SamplingConfig("conservative", temperature=0.3, top_p=0.9, max_tokens=200),
    SamplingConfig("balanced",    temperature=0.7, top_p=1.0, max_tokens=200),
    SamplingConfig("creative",    temperature=1.1, top_p=0.95, max_tokens=200),
]

TASKS = {
    "extraction": 'Extract the person name, date, and amount from: "Alice paid $234.50 on March 7, 2024."',
    "summarization": "Summarize in one sentence: Transformers use self-attention to process all tokens in parallel, enabling training on massive datasets with GPU parallelism. Unlike RNNs, they can relate distant tokens directly.",
    "creative": "Write the first sentence of a short story about a robot learning to feel emotions.",
    "code": "Write a Python function that returns the nth Fibonacci number using memoization.",
}

results = {}
for task_name, task_prompt in TASKS.items():
    results[task_name] = {}
    for config in CONFIGS:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": task_prompt}],
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )
        results[task_name][config.name] = response.choices[0].message.content

# Print results in a readable format
for task, outputs in results.items():
    print(f"\n{'='*60}")
    print(f"TASK: {task.upper()}")
    print("="*60)
    for config_name, output in outputs.items():
        print(f"\n--- {config_name} ---")
        print(output[:300])

# Reflect: which config performs best per task? Why?
```

**What to observe and note down:**
- Extraction task: does creative temperature break JSON structure?
- Code task: does temperature affect correctness?
- Creative task: does greedy produce something generic vs. interesting?

---

## Main exercise: Context window audit tool

**Goal:** Build a utility that warns you before you exceed the context limit.

```python
import tiktoken
import openai

client = openai.OpenAI()

CONTEXT_LIMITS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
}

def context_audit(
    system: str,
    messages: list[dict],
    expected_output_tokens: int,
    model: str = "gpt-4o",
) -> None:
    enc = tiktoken.get_encoding("o200k_base")
    limit = CONTEXT_LIMITS.get(model, 128_000)

    system_tokens = len(enc.encode(system)) + 4
    message_tokens = sum(len(enc.encode(m["content"])) + 4 for m in messages) + 2
    total_input = system_tokens + message_tokens
    total_with_output = total_input + expected_output_tokens

    print(f"\n📊 Context Window Audit — {model}")
    print(f"  System prompt:        {system_tokens:8,} tokens")
    print(f"  Messages:             {message_tokens:8,} tokens")
    print(f"  Expected output:      {expected_output_tokens:8,} tokens")
    print(f"  ─────────────────────────────────")
    print(f"  Total:                {total_with_output:8,} tokens")
    print(f"  Limit:                {limit:8,} tokens")
    print(f"  Remaining:            {limit - total_with_output:8,} tokens")

    usage_pct = (total_with_output / limit) * 100
    if usage_pct > 90:
        print(f"\n⚠️  WARNING: Using {usage_pct:.1f}% of context window!")
    elif usage_pct > 70:
        print(f"\n⚡ CAUTION: Using {usage_pct:.1f}% of context window.")
    else:
        print(f"\n✅ OK: Using {usage_pct:.1f}% of context window.")

# Test with a realistic scenario
context_audit(
    system="You are a helpful assistant. Answer based only on the provided documents.",
    messages=[
        {"role": "user", "content": "Document: " + "word " * 5000},  # ~5000 tokens
        {"role": "assistant", "content": "I have read the document."},
        {"role": "user", "content": "What is the main topic?"},
    ],
    expected_output_tokens=500,
    model="gpt-4o",
)
```

---

## Stretch: The lost-in-the-middle experiment

**Goal:** Empirically verify the lost-in-the-middle effect. Insert a fact at different positions in a long context and measure retrieval accuracy.

```python
import openai
import random

client = openai.OpenAI()

FILLER = "This is filler text to pad the context. " * 50  # ~250 tokens per block
SECRET_FACT = "The secret password for this exercise is BANANA-7734."
QUESTION = "What is the secret password mentioned in the document?"

def create_context(secret_position: str, num_filler_blocks: int = 20) -> str:
    """Place the secret fact at the beginning, middle, or end."""
    fillers = [FILLER for _ in range(num_filler_blocks)]

    if secret_position == "beginning":
        parts = [SECRET_FACT] + fillers
    elif secret_position == "end":
        parts = fillers + [SECRET_FACT]
    elif secret_position == "middle":
        mid = len(fillers) // 2
        parts = fillers[:mid] + [SECRET_FACT] + fillers[mid:]
    else:
        raise ValueError("position must be 'beginning', 'middle', or 'end'")

    return "\n\n".join(parts)

results = {}
for position in ["beginning", "middle", "end"]:
    correct = 0
    trials = 5
    for _ in range(trials):
        context = create_context(position)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Document:\n{context}\n\nQuestion: {QUESTION}"},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        answer = response.choices[0].message.content
        if "BANANA-7734" in answer:
            correct += 1

    results[position] = correct / trials * 100
    print(f"Position: {position:10} → Accuracy: {results[position]:.0f}%")

# Expected: beginning ≈ 95-100%, middle ≈ 60-80%, end ≈ 90-100%
# This validates the "lost in the middle" effect empirically
```

**Reflect:** What does this tell you about how to order chunks in your RAG system?

---

## Expected outputs

By the end of these exercises you should have:

- ✅ Clear intuition for temperature values (0.0 vs. 0.7 vs. 1.2)
- ✅ A working token counter for both OpenAI and Anthropic
- ✅ A context audit tool you can drop into any project
- ✅ Empirical evidence for the lost-in-the-middle effect

[[04-how-llms-generate-text]] | [[06-interview-questions]]
