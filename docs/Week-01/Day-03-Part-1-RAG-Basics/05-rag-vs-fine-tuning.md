# RAG vs Fine-Tuning

Both RAG and fine-tuning adapt a model's behavior. They solve different problems and they're not mutually exclusive. The wrong choice costs you weeks of work or production failures.

## Learning objectives

- Articulate the key differences between RAG and fine-tuning
- Apply a decision framework to choose the right approach
- Know the hybrid patterns that combine both
- Identify the cases where neither helps

---

## What each approach does

**RAG:** Keeps the base model frozen. Injects external documents into the context window at query time. The model reasons over the provided text.

**Fine-tuning:** Updates the model's weights on a curated dataset. The new knowledge or behavior is baked into the model's parameters.

```
RAG architecture:
User query → Retrieve documents → Inject into prompt → LLM generates → Answer

Fine-tuning architecture:
Training examples → Gradient updates → Updated weights → LLM generates → Answer
```

---

## Side-by-side comparison

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| **Updates knowledge** | Yes — update the vector index | Requires retraining |
| **Cites sources** | Yes — you know what was retrieved | No — knowledge is implicit in weights |
| **Time to implement** | Hours to days | Days to weeks |
| **Data required** | Documents (no labels needed) | 100s–1000s of labeled (input, output) pairs |
| **Compute cost** | Inference + vector index | Training cost (GPUs, days) |
| **Inference cost** | Higher (more context tokens) | Lower (shorter prompts needed) |
| **Handles proprietary data** | Yes — store in your own index | Yes — train on it |
| **Changes model style/format** | Limited (prompt engineering) | Yes — strong style control |
| **Interpretable** | Yes — see what was retrieved | No — knowledge is in weights |
| **Hallucination on retrieval** | Possible if retrieval fails | Possible if training data has errors |

---

## Decision framework

```python
def should_use_rag(problem: dict) -> str:
    """
    decision_factors keys:
        knowledge_changes_frequently: bool
        need_source_attribution: bool
        have_labeled_training_data: bool (100+ examples)
        need_consistent_output_format: bool
        high_volume_task: bool (millions of calls/month)
        domain_specific_vocabulary: bool
        knowledge_is_proprietary: bool
    """
    factors = problem

    # Strong RAG signals
    if factors.get("knowledge_changes_frequently"):
        return "RAG — knowledge changes make fine-tuning impractical"
    if factors.get("need_source_attribution"):
        return "RAG — fine-tuning can't cite sources"
    if not factors.get("have_labeled_training_data"):
        return "RAG — no labeled training data available"

    # Strong fine-tuning signals
    if factors.get("need_consistent_output_format") and factors.get("have_labeled_training_data"):
        return "Fine-tuning — format consistency requires weight-level training"
    if factors.get("high_volume_task") and factors.get("domain_specific_vocabulary"):
        return "Fine-tuning — cost reduction at scale + domain vocabulary"

    # Default
    return "RAG — lower barrier, faster iteration, easier to debug"
```

---

## Scenario analysis

### Scenario 1: Customer support chatbot for a SaaS product

**Situation:** Users ask about features, pricing, and troubleshooting. Docs are updated weekly with new features.

**Choice: RAG**

- Knowledge changes weekly — retraining every week is impractical
- Answers must cite specific doc sections for compliance
- No labeled (question, answer) pairs exist yet

**Implementation:** Ingest product docs + changelogs into a vector index. Update the index when docs are updated. Query with k=4.

---

### Scenario 2: Medical coding assistant

**Situation:** Convert free-text physician notes into ICD-10 billing codes. Very consistent input/output format. 10,000 labeled examples from internal records.

**Choice: Fine-tuning (LoRA on a medical LLM)**

- Output format is extremely consistent (code → category → description)
- Proprietary medical vocabulary the base model doesn't know well
- 10,000 labeled examples available
- High volume: 50,000 notes/day — inference cost matters

---

### Scenario 3: Legal document summarization

**Situation:** Summarize case law and contracts. Documents are novel each time. Need accurate citations to specific clauses.

**Choice: RAG**

- Every document is unique — there's no general "legal summary" knowledge to fine-tune in
- Citations to specific clause numbers are critical
- Source attribution is a hard requirement

---

### Scenario 4: Code completion in a proprietary framework

**Situation:** Autocomplete code for an internal Python framework with custom APIs the model has never seen.

**Choice: RAG + Fine-tuning**

- **RAG:** Retrieve relevant framework docs, example code, and API references
- **Fine-tuning:** Adapt the model to the framework's style and conventions
- Hybrid is better than either alone — RAG brings fresh context, fine-tuning brings stylistic adherence

---

## The hybrid approach

RAG and fine-tuning combine naturally:

```
┌────────────────────────────────────────────────────────┐
│ 1. Fine-tune on domain style and format                │
│    (teach the model HOW to respond for your domain)    │
├────────────────────────────────────────────────────────┤
│ 2. RAG for dynamic knowledge                           │
│    (give the fine-tuned model the right context)       │
└────────────────────────────────────────────────────────┘
```

```python
# Hybrid RAG + fine-tuned model pattern (pseudocode)
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def hybrid_answer(question: str, retrieved_context: str) -> str:
    response = client.chat.completions.create(
        # A fine-tuned model trained on your domain data
        model="ft:gpt-4o-mini:your-org:model-id:version",
        messages=[
            {
                "role": "system",
                "content": "You are a medical coding specialist. Use the provided context to assign ICD-10 codes."
            },
            {
                "role": "user",
                # Context comes from RAG; format knowledge comes from fine-tuning
                "content": f"Patient notes:\n{retrieved_context}\n\nAssign ICD-10 codes."
            }
        ],
        max_tokens=300,
        temperature=0.0
    )
    return response.choices[0].message.content
```

---

## Cost comparison at scale

```python
def cost_comparison(
    daily_queries: int,
    avg_input_tokens_rag: int = 2500,   # system + context + question
    avg_input_tokens_ft: int = 500,     # system + question only (shorter prompts)
    avg_output_tokens: int = 300
) -> dict:
    # OpenAI pricing
    GPT4O_MINI_INPUT = 0.15 / 1_000_000
    GPT4O_MINI_OUTPUT = 0.60 / 1_000_000
    GPT4O_FT_EXTRA = 0.30 / 1_000_000  # fine-tuned surcharge per token

    monthly_queries = daily_queries * 30

    # RAG with base model
    rag_monthly = monthly_queries * (
        avg_input_tokens_rag * GPT4O_MINI_INPUT +
        avg_output_tokens * GPT4O_MINI_OUTPUT
    )

    # Fine-tuned model (shorter prompts, but extra per-token cost)
    ft_monthly = monthly_queries * (
        avg_input_tokens_ft * (GPT4O_MINI_INPUT + GPT4O_FT_EXTRA) +
        avg_output_tokens * GPT4O_MINI_OUTPUT
    )

    return {
        "daily_queries": daily_queries,
        "rag_monthly_usd": rag_monthly,
        "ft_monthly_usd": ft_monthly,
        "ft_cheaper": ft_monthly < rag_monthly,
        "savings_if_ft": max(0, rag_monthly - ft_monthly)
    }

for volume in [1_000, 10_000, 100_000, 1_000_000]:
    result = cost_comparison(volume)
    print(f"{volume:>8,} queries/day | RAG: ${result['rag_monthly_usd']:>8,.2f}/mo | FT: ${result['ft_monthly_usd']:>8,.2f}/mo | FT cheaper: {result['ft_cheaper']}")
```

> [!success] Decision rule
> At < 100K daily queries, RAG's flexibility outweighs its higher token cost. At > 1M daily queries with a stable task, fine-tuning saves tens of thousands of dollars per month. The crossover point depends on your token counts and the fine-tuning surcharge.

---

## When neither helps

> [!warning] RAG doesn't fix reasoning failures
> If GPT-4o-mini can't solve a problem even when given the exact relevant text, RAG won't fix it. The issue is reasoning capacity, not retrieval. Upgrade the model.

> [!warning] Fine-tuning doesn't fix missing knowledge
> Fine-tuning teaches style and format, not new facts reliably. If your model consistently gets specific factual answers wrong (dates, names, numbers), fine-tuning on examples may help with common cases but won't generalize to novel facts. Use RAG for factual grounding.

> [!warning] Neither solves bad data quality
> Garbage in, garbage out. RAG with outdated or incorrect documents gives wrong answers confidently. Fine-tuning on noisy labels gives a model that's confidently wrong. Data quality is always the prerequisite.

---

[[04-rag-pipeline-end-to-end]] | [[06-practice-exercises]]
