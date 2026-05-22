# Chain-of-Thought Prompting

Chain-of-thought (CoT) prompting asks the model to reason step-by-step before producing its final answer. This single technique unlocks capabilities that seem unavailable in direct prompting — complex math, multi-step logical deduction, and nuanced judgment calls.

## Learning objectives

- Understand why reasoning before answering improves accuracy
- Apply zero-shot CoT, few-shot CoT, and self-consistency
- Know when CoT helps and when it's unnecessary overhead
- Use extended thinking (Claude) and reasoning effort (o-series) for maximum accuracy

---

## Why CoT works

When the model generates its answer directly, it commits to an answer in the first token. If that answer is wrong, the entire response follows a wrong path — there's no going back.

CoT changes the information available at answer time:

```
Without CoT:
[Question] → [Answer token 1] → [Answer token 2] → ...
              ↑ committed here

With CoT:
[Question] → [Reasoning step 1] → [Reasoning step 2] → [... N steps] → [Answer]
                                                                         ↑ conditioned on all reasoning tokens
```

The reasoning tokens are in the context window when the model produces the answer. It can "look back" at its own work.

---

## Zero-shot CoT

The simplest form: append a reasoning trigger to your prompt.

```python
import openai

client = openai.OpenAI()

def ask(prompt: str, cot: bool = False, temperature: float = 0.0) -> str:
    if cot:
        prompt = prompt + "\n\nThink step by step before giving your final answer."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content

# Arithmetic problem
question = """A store sells apples for $0.75 each and oranges for $1.25 each.
If Alice buys 4 apples and 3 oranges, and pays with a $10 bill, how much change does she get?"""

without_cot = ask(question, cot=False)
with_cot = ask(question, cot=True)

print("Without CoT:", without_cot)
print("\nWith CoT:", with_cot)

# Without CoT: may produce the wrong answer directly
# With CoT: computes 4×0.75=3.00, 3×1.25=3.75, total=6.75, change=3.25
```

**Effective zero-shot CoT triggers:**

```python
ZERO_SHOT_COT_TRIGGERS = [
    "Think step by step.",                          # Original (Wei et al., 2022)
    "Let's reason through this carefully.",
    "Break this problem down into steps.",
    "Before answering, work through the logic.",
    "Let's think about this systematically.",
]

# For Anthropic/Claude specifically — let thinking come first
CLAUDE_COT = "Reason through this before giving your final answer."
```

---

## Few-shot CoT

Provide examples that include the reasoning chain, not just the answer.

```python
import anthropic

client = anthropic.Anthropic()

FEW_SHOT_COT_PROMPT = """Classify the customer sentiment and explain your reasoning.

Example 1:
Review: "The product stopped working after two days. Customer service took a week to respond."
Reasoning: The product failed quickly (negative technical experience) and support was slow (negative service experience). No positive elements mentioned.
Sentiment: NEGATIVE

Example 2:
Review: "Incredible quality, though it took a while to arrive."
Reasoning: "Incredible quality" is strongly positive. "Took a while to arrive" is mildly negative. The positive clearly outweighs the negative.
Sentiment: POSITIVE

Example 3:
Review: "Works as advertised. Nothing special but does the job."
Reasoning: "Works as advertised" is neutral-positive. "Nothing special" indicates no excitement. "Does the job" is functional satisfaction. No strong sentiment either way.
Sentiment: NEUTRAL

Now classify:
Review: "{review}"
Reasoning:"""

reviews = [
    "Absolutely love it! Best purchase of the year by far.",
    "It's fine I guess. Does what it's supposed to.",
    "Arrived broken. Replacement also broke. Never buying again.",
    "Good product, terrible instructions. Took 3 hours to set up.",
]

for review in reviews:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": FEW_SHOT_COT_PROMPT.format(review=review)}],
    )
    print(f"Review: {review[:60]}")
    print(response.content[0].text)
    print()
```

---

## Self-consistency

Run the same CoT prompt multiple times with higher temperature, then take the majority vote. Significantly more accurate than a single CoT pass on tasks with clear right/wrong answers.

```python
from collections import Counter

def self_consistent_answer(
    question: str,
    n_samples: int = 7,
    temperature: float = 0.7,
) -> tuple[str, float]:
    """
    Generate n reasoning chains, extract final answers, return majority vote.
    Returns: (most_common_answer, confidence_ratio)
    """
    cot_prompt = f"""{question}

Think step by step. End your response with:
FINAL ANSWER: [your answer here]"""

    answers = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": cot_prompt}],
            temperature=temperature,
            max_tokens=512,
        )
        text = response.choices[0].message.content
        # Extract the final answer
        if "FINAL ANSWER:" in text:
            answer = text.split("FINAL ANSWER:")[-1].strip().split("\n")[0]
            answers.append(answer.strip())

    if not answers:
        return "No answer extracted", 0.0

    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    return most_common[0], most_common[1] / len(answers)

# Example
question = "A train leaves London at 9:00 AM traveling at 120 km/h. Another train leaves Paris (450 km away) at 10:00 AM traveling at 150 km/h toward London. At what time do they meet?"

answer, confidence = self_consistent_answer(question, n_samples=5)
print(f"Answer: {answer}")
print(f"Confidence: {confidence:.0%}")
```

> [!tip] Self-consistency cost
> 7 samples × cost per call = 7× the cost. Use self-consistency only when accuracy is critical and you can afford the latency and cost. For production: use reasoning models (o3, o4-mini) instead — they achieve similar accuracy at lower overhead.

---

## Structured CoT: separating reasoning from answer

For production systems, you often want the reasoning hidden from the end user but available for debugging.

```python
import anthropic
import json

client = anthropic.Anthropic()

def structured_cot(problem: str) -> dict:
    """
    Separate reasoning and answer into distinct fields.
    Use XML tags to force the model to structure its output.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Solve the following problem. 

First, write your reasoning inside <reasoning> tags.
Then write your final answer inside <answer> tags.

Problem: {problem}""",
        }],
    )

    text = response.content[0].text

    # Parse reasoning and answer
    reasoning = ""
    answer = ""

    if "<reasoning>" in text and "</reasoning>" in text:
        reasoning = text.split("<reasoning>")[1].split("</reasoning>")[0].strip()
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[1].split("</answer>")[0].strip()

    return {"reasoning": reasoning, "answer": answer, "raw": text}

result = structured_cot(
    "A company's revenue grew 15% in 2023 and 20% in 2024. Starting from $1M in 2022, what is the 2024 revenue?"
)
print("REASONING:", result["reasoning"])
print("\nANSWER:", result["answer"])
# Log reasoning for debugging, show only answer to user
```

---

## Extended thinking — Claude

Claude's extended thinking lets the model think internally before responding. The thinking tokens are separate from the response and have a dedicated token budget.

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 8000,  # how many tokens Claude can use for thinking
    },
    messages=[{
        "role": "user",
        "content": """A factory produces widgets at a rate that doubles every 6 hours.
It starts with 100 widgets. After how many hours will it have produced more than 10,000 widgets total?
Include the production at each 6-hour interval.""",
    }],
)

for block in response.content:
    if block.type == "thinking":
        print("=== INTERNAL REASONING (debug only) ===")
        print(block.thinking[:500], "...")
        print(f"\nThinking tokens used: ~{len(block.thinking.split()):,} words")
    elif block.type == "text":
        print("\n=== FINAL ANSWER ===")
        print(block.text)
```

---

## Reasoning models — OpenAI o-series

OpenAI's o3, o3-mini, o4-mini generate a hidden chain-of-thought before responding. You don't need to write a CoT prompt — the model always reasons.

```python
import openai

client = openai.OpenAI()

# o4-mini: reasoning_effort controls how much thinking to do
response = client.chat.completions.create(
    model="o4-mini",
    reasoning_effort="high",   # "low" | "medium" | "high"
    messages=[{
        "role": "user",
        "content": "If a ladder leans against a wall at a 60° angle and the base is 3m from the wall, how long is the ladder?",
    }],
    max_completion_tokens=4000,
)

print(response.choices[0].message.content)
# Reasoning is done internally — you only see the final answer
# To see reasoning summary: check response.choices[0].message.reasoning_content (if enabled)
```

**When to use which:**

| Approach | Latency | Cost | When |
|----------|---------|------|------|
| Zero-shot CoT | Low | Low | Most tasks — start here |
| Few-shot CoT | Low | Medium | Custom format/domain |
| Self-consistency | High | High × N | Critical accuracy, no budget for reasoning models |
| Extended thinking (Claude) | High | Medium-High | Complex analysis, long-horizon tasks |
| o3 / o4-mini | Medium-High | High | Math, code, logic puzzles — when right answer is critical |

---

## When CoT does NOT help

- **Simple factual Q&A:** "What is the capital of Japan?" — CoT adds latency for no gain
- **Creative writing:** Reasoning about creativity kills it. Use temperature, not CoT.
- **High-latency-sensitive applications:** CoT adds 300ms–3s to each response
- **Classification with obvious labels:** The model already knows; forcing it to reason adds noise

```python
# BAD: CoT on a simple factual lookup
bad_prompt = "Think step by step: What is 2 + 2?"
# Model wastes tokens on: "First, I'll consider the operands..."

# GOOD: CoT on a multi-step problem
good_prompt = """Think step by step:

A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"""
# Without CoT: most people (and models) say $0.10 — wrong
# With CoT: $0.05 (ball) + $1.05 (bat) = $1.10 ✓
```

---

> [!success] Key takeaway
> Add "Think step by step" to any prompt where the model needs to reason before answering. Measure whether it helps. For maximum accuracy on reasoning tasks, use o4-mini or Claude extended thinking instead of manual CoT.

[[01-zero-shot-and-few-shot]] | [[03-structured-output]]
