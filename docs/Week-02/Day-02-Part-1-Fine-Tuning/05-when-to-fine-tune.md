# When to Fine-Tune vs RAG vs Prompting

Fine-tuning is a tool, not a default. It's expensive to run once and expensive to repeat when requirements change. Before committing to a training run, exhaust cheaper options. The question isn't "should we fine-tune?" — it's "what is actually failing and what's the cheapest fix?"

## Learning objectives

- Apply the decision framework: prompting → RAG → fine-tuning
- Identify which failure modes each technique addresses
- Calculate when fine-tuning is cost-justified vs API usage
- Know the three situations where fine-tuning is clearly the right choice

---

## The decision framework

```python
def choose_technique(problem: str) -> str:
    """
    Decision tree for choosing between prompting, RAG, and fine-tuning.
    Not a real function — use it as a checklist.
    """
    # Step 1: Have you tried prompting first?
    if not tried_prompting:
        return "Try few-shot prompting with 3–5 examples first"

    # Step 2: Is the problem knowledge (facts) or behavior (format/style)?
    if problem == "model_lacks_knowledge":
        if knowledge_changes_frequently:
            return "RAG — inject knowledge at query time"
        else:
            return "RAG or fine-tuning — depends on query volume"

    if problem == "model_behavior":
        # Behavior: format, style, task-following, domain-specific output
        if examples_available >= 100:
            return "Fine-tuning (LoRA)"
        else:
            return "Improve your prompt; collect more examples"

    # Step 3: Is latency or cost driving the decision?
    if cost_per_query_too_high:
        # Fine-tuned smaller model often cheaper than API calls to frontier model
        return "Fine-tuning a smaller model to replace large-model API calls"

    return "RAG or prompting"
```

---

## What each technique actually fixes

```python
TECHNIQUE_FIT = {
    "prompting": {
        "fixes": [
            "Model ignores your format (add explicit format instructions)",
            "Model output is too long/short (add length constraint)",
            "Model refuses valid requests (adjust system prompt)",
            "Model uses wrong tone (add tone examples in system prompt)",
        ],
        "does_not_fix": [
            "Model lacks domain knowledge",
            "Consistent behavior across hundreds of edge cases",
            "Task too complex for zero/few-shot",
        ],
        "cost": "$0 extra",
        "time_to_deploy": "Minutes",
    },
    "RAG": {
        "fixes": [
            "Model answers with outdated information",
            "Model doesn't know company-specific data",
            "Model hallucinates facts that exist in your documents",
            "Model needs to cite sources",
        ],
        "does_not_fix": [
            "Model output format/style inconsistency",
            "Model behavior on edge cases",
            "Latency-sensitive applications (RAG adds retrieval time)",
        ],
        "cost": "Vector DB + retrieval compute",
        "time_to_deploy": "Hours to days",
    },
    "fine_tuning": {
        "fixes": [
            "Consistent output format across all inputs",
            "Domain-specific tone and vocabulary",
            "Complex task-following that few-shot can't handle",
            "Reduce prompt length (bake instructions into weights)",
            "Replace expensive large model with fine-tuned small model",
        ],
        "does_not_fix": [
            "Lack of factual knowledge (use RAG for this)",
            "Need for real-time information",
            "Problems solvable with better prompting",
        ],
        "cost": "$10–500 for QLoRA; months of data collection",
        "time_to_deploy": "Days to weeks",
    },
}

for tech, info in TECHNIQUE_FIT.items():
    print(f"\n{'=' * 40}")
    print(f"Technique: {tech}")
    print(f"Fixes: {info['fixes'][0]}, ...")
    print(f"Cost: {info['cost']}")
```

---

## Cost comparison: fine-tuned model vs API

When your query volume is high, a fine-tuned small model often beats API calls to a frontier model:

```python
from dataclasses import dataclass

@dataclass
class CostScenario:
    queries_per_month: int
    avg_input_tokens: int
    avg_output_tokens: int

def api_cost_per_month(scenario: CostScenario) -> float:
    """GPT-4o pricing: $2.50/1M input, $10/1M output (as of 2025)."""
    input_cost = (scenario.queries_per_month * scenario.avg_input_tokens / 1e6) * 2.50
    output_cost = (scenario.queries_per_month * scenario.avg_output_tokens / 1e6) * 10.0
    return input_cost + output_cost

def fine_tuned_cost_per_month(
    training_cost: float = 50.0,      # One-time QLoRA training on cloud GPU
    inference_cost_per_month: float = 200.0,  # Self-hosted or dedicated endpoint
    amortize_months: int = 12
) -> float:
    return (training_cost / amortize_months) + inference_cost_per_month

scenario = CostScenario(
    queries_per_month=100_000,
    avg_input_tokens=500,
    avg_output_tokens=100,
)

api_monthly = api_cost_per_month(scenario)
ft_monthly = fine_tuned_cost_per_month()

print(f"API (GPT-4o):          ${api_monthly:,.0f}/month")
print(f"Fine-tuned (self-host): ${ft_monthly:,.0f}/month")
print(f"Break-even at: {ft_monthly / (api_monthly / 100_000):.0f} queries/month")
```

---

## The three clear cases for fine-tuning

```
Case 1: CONSISTENT BEHAVIOR AT SCALE
  The task has a specific output format that prompting achieves 80% of the time
  but you need 98%+. Edge cases break prompt-only solutions.
  Example: structured JSON extraction from noisy text.

Case 2: LATENCY + COST REDUCTION
  You're using GPT-4o for a simple classification that a fine-tuned phi-3-mini
  could do equally well. Fine-tuning the smaller model cuts cost by 50–100x.
  Example: spam detection, ticket routing, entity tagging.

Case 3: PROPRIETARY STYLE OR DOMAIN
  Your company has a unique writing style or highly specialized domain language
  that no amount of prompting reliably replicates.
  Example: medical report generation with specific formatting requirements.
```

---

## Combining techniques

Fine-tuning and RAG are complementary, not competing:

```python
# Pattern: Fine-tuned model as the RAG reader
# - RAG retrieves relevant context (solves the knowledge problem)
# - Fine-tuned model generates consistent output format (solves the behavior problem)

def rag_with_fine_tuned_reader(
    question: str,
    retriever,
    fine_tuned_model,
    tokenizer
) -> str:
    # Step 1: RAG retrieval
    context_chunks = retriever.retrieve(question, k=5)
    context = "\n".join(context_chunks)

    # Step 2: Fine-tuned model reads and answers
    prompt = f"### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(fine_tuned_model.device)

    import torch
    with torch.no_grad():
        outputs = fine_tuned_model.generate(
            **inputs, max_new_tokens=200, do_sample=False
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

> [!success] Default to prompting, add RAG for knowledge, fine-tune for behavior
> This ordering minimizes wasted effort. Every technique higher in the stack is cheaper and faster to implement. Only move down when the higher-level technique is genuinely insufficient.

---

## Red flags: when fine-tuning will fail

> [!warning] Don't fine-tune to fix these
> - **Insufficient data:** Less than 50 examples for classification is almost never enough
> - **Noisy labels:** Inconsistent labeling (even 10% noise) degrades model quality significantly
> - **Task too broad:** "Be better at customer support" is not a trainable task; "classify ticket category" is
> - **Moving target:** If the task definition changes frequently, fine-tuning creates maintenance debt
> - **Hallucination prevention:** Fine-tuning does not reliably prevent hallucination; use RAG + grounding prompts

---

[[04-training-data]] | [[06-practice-exercises]]
