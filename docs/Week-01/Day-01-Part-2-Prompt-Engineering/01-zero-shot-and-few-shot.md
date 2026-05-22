# Zero-Shot and Few-Shot Prompting

The first question when building any LLM feature is: how much does the model already know? Zero-shot measures the baseline. Few-shot gives the model examples to calibrate from. Getting this right is cheaper and faster than any other technique.

## Learning objectives

- Write effective zero-shot prompts using clarity, specificity, and output format instructions
- Design few-shot examples that generalize rather than overfit
- Know when to add examples and when they're unnecessary overhead
- Measure whether examples actually help using a quick evaluation

---

## Zero-shot prompting

Zero-shot means no examples — just the instruction. Modern frontier models (GPT-4o, Claude Sonnet 4.6) handle a remarkable range of tasks zero-shot.

The key to effective zero-shot prompting is **specificity**:

```python
import openai

client = openai.OpenAI()

def prompt(user_message: str, system: str = "", temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
        max_tokens=512,
    )
    return response.choices[0].message.content

# BAD — vague instruction
bad = prompt("Tell me about the customer review: 'Great product, fast shipping!'")
print("Bad:", bad)
# Could produce anything: summary, analysis, praise, critique...

# GOOD — specific instruction with output format
good = prompt("""Classify the sentiment of this customer review as POSITIVE, NEGATIVE, or NEUTRAL.
Respond with only the label, nothing else.

Review: 'Great product, fast shipping!'""")
print("Good:", good)
# Output: POSITIVE
```

### The specificity ladder

Vague → precise prompts:

| Version | Prompt | Problem |
|---------|--------|---------|
| Vague | "Summarize this email." | Unclear length, format, audience |
| Better | "Summarize this email in 2–3 sentences." | Length specified |
| Good | "Summarize this email in 2–3 sentences for a busy executive who needs to know: action required, deadline, and key stakeholders." | Purpose + content requirements specified |
| Excellent | Add: "Use plain language. No jargon. Start with the action." | Style constraints added |

---

## Anatomy of a strong zero-shot prompt

```python
def classify_support_ticket(ticket_text: str) -> str:
    """
    Zero-shot classifier using all five components of a strong prompt.
    """
    prompt_text = f"""You are a customer support routing system.

## Task
Classify the support ticket below into exactly one category.

## Categories
- BILLING: Payment issues, refunds, invoices, subscription changes
- TECHNICAL: Bugs, errors, crashes, performance issues, integrations
- ACCOUNT: Login, password, permissions, account settings
- FEATURE_REQUEST: Suggestions for new features or improvements
- OTHER: Anything that doesn't fit the above categories

## Rules
- Respond with ONLY the category label (e.g., BILLING)
- Do not explain your reasoning
- If multiple categories apply, pick the most specific one

## Ticket
{ticket_text}

## Category"""
    return prompt(prompt_text)

# Test
tickets = [
    "I was charged twice for my subscription last month",
    "The export to CSV button throws a 500 error",
    "Can you add dark mode to the dashboard?",
    "I can't log in after resetting my password",
]
for ticket in tickets:
    label = classify_support_ticket(ticket)
    print(f"  [{label}] {ticket[:60]}")
```

**The five components** of a strong zero-shot prompt:
1. **Role / context** — who the model is
2. **Task** — what to do, precisely
3. **Constraints** — format, length, style, what NOT to do
4. **Input** — the actual content to process
5. **Output anchor** — start the output structure (e.g., `## Category`)

---

## Few-shot prompting

Few-shot provides worked examples inside the prompt. The model uses them as templates to calibrate format, tone, and reasoning style.

```python
import anthropic

client = anthropic.Anthropic()

def extract_structured_data(text: str) -> str:
    """
    Few-shot extraction — examples teach the output format.
    """
    system = "You extract structured information from text. Return only the requested data."

    few_shot_prompt = f"""Extract the person's name, company, and role from the text.
Format: Name | Company | Role

Text: "Hi, I'm Sarah Chen, Senior Product Manager at DataFlow Inc."
Extract: Sarah Chen | DataFlow Inc. | Senior Product Manager

Text: "This is Mike Rodriguez from the engineering team at CloudBase."
Extract: Mike Rodriguez | CloudBase | Unknown

Text: "Dr. Emily Watson, Chief Medical Officer, HealthPath Systems."
Extract: Emily Watson | HealthPath Systems | Chief Medical Officer

Text: "{text}"
Extract:"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        system=system,
        messages=[{"role": "user", "content": few_shot_prompt}],
    )
    return response.content[0].text.strip()

test_cases = [
    "I'm Alex Kim, the lead data scientist at NeuralLabs.",
    "Please connect me with James Thompson, he's a partner at Summit Ventures.",
    "Our CTO, Lisa Park from Quantix, will join the call.",
]
for text in test_cases:
    result = extract_structured_data(text)
    print(f"Input: {text}")
    print(f"Output: {result}\n")
```

---

## Writing good few-shot examples

Bad examples hurt more than no examples. Follow these rules:

### Rule 1: Cover the edge cases

```python
# BAD: all easy cases — model won't know how to handle edges
bad_examples = [
    ("I love this product!", "POSITIVE"),
    ("This is terrible.", "NEGATIVE"),
    ("It's okay.", "NEUTRAL"),
]

# GOOD: include the hard cases
good_examples = [
    ("I love this product!", "POSITIVE"),
    ("This is terrible.", "NEGATIVE"),
    ("It's okay.", "NEUTRAL"),
    ("The product is great but shipping took forever.", "MIXED"),  # edge case
    ("I returned it.", "NEUTRAL"),  # no sentiment expressed
    ("Don't buy this. Seriously.", "NEGATIVE"),  # sarcasm-adjacent
]
```

### Rule 2: Match the distribution of real inputs

```python
def build_few_shot_prompt(examples: list[tuple[str, str]], new_input: str) -> str:
    """Build a few-shot prompt from (input, output) pairs."""
    lines = []
    for inp, out in examples:
        lines.append(f"Input: {inp}")
        lines.append(f"Output: {out}")
        lines.append("")  # blank line separator
    lines.append(f"Input: {new_input}")
    lines.append("Output:")
    return "\n".join(lines)

# The number of examples: use 3–8. Diminishing returns after 8 for most tasks.
# Research shows: quality of examples >> quantity of examples
```

### Rule 3: Put the most relevant example last

The final example before the new input has the strongest priming effect. Put your best or most representative example there.

---

## Zero-shot vs few-shot: when to use which

| Situation | Use |
|-----------|-----|
| Standard NLP task (sentiment, classification) | Zero-shot — frontier models are pre-trained on these |
| Custom output format (your company's JSON schema) | Few-shot — the model hasn't seen your format |
| Complex multi-field extraction | Few-shot — 2–3 examples clarify expectations |
| Creative generation with specific style | Few-shot — 1–2 samples of your desired style |
| Simple Q&A, summarization | Zero-shot — examples rarely help |
| Task with many edge cases | Few-shot with curated edge case examples |

> [!tip] Calibration test
> Add one example at a time and measure output quality. If adding the third example doesn't improve results, stop. Each example costs tokens.

---

## Dynamic few-shot selection

Static examples embedded in a prompt are easy. Dynamic selection — retrieving the most relevant examples for each input — is more powerful and is covered in the RAG section.

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# Embed examples and query to find most similar ones
def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def select_examples(
    example_pool: list[tuple[str, str]],
    query: str,
    k: int = 3,
) -> list[tuple[str, str]]:
    """Return the k most similar examples to the query."""
    query_emb = get_embedding(query)
    scored = [
        (cosine_similarity(query_emb, get_embedding(inp)), inp, out)
        for inp, out in example_pool
    ]
    scored.sort(reverse=True)
    return [(inp, out) for _, inp, out in scored[:k]]

# In practice: pre-compute and store example embeddings in a vector DB
# See: Week-01/Day-03-Part-2-Vector-Databases
```

---

## Common mistakes

> [!warning] Inconsistent examples confuse the model
> If your examples use different formats — sometimes JSON, sometimes plain text — the model will pick one unpredictably. Use exactly one format across all examples.

> [!warning] Example labels that are wrong
> A mislabeled example is worse than no example. The model will learn from it. Curate your examples carefully — treat them like training data.

> [!warning] Too many examples = hidden context cost
> 10 examples × 200 tokens each = 2,000 tokens of prompt. For a high-volume classification task making 100,000 calls/day, that's 200M extra input tokens ≈ $500/day at GPT-4o pricing.

---

> [!success] Key takeaway
> Start with zero-shot. Add examples only when zero-shot fails, and test empirically that they help. The quality of your examples matters more than the quantity.

[[00-agenda]] | [[02-chain-of-thought]]
