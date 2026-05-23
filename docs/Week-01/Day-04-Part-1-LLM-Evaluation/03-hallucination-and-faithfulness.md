# Hallucination and Faithfulness

Hallucination is not a bug you can patch — it's a fundamental property of how language models work. Understanding what causes it, how to measure it, and how to suppress it is the difference between a demo and a production system.

## Learning objectives

- Distinguish hallucination types (intrinsic vs extrinsic, factual vs faithful)
- Implement faithfulness scoring without RAGAS
- Build a hallucination detection pipeline using NLI
- Apply prompt and retrieval techniques that reduce hallucination rate

---

## What hallucination actually is

Language models generate the most probable next token given prior context. They do not look up facts — they interpolate from training data. When the model reaches a gap in its training knowledge, it fills the gap with statistically plausible text, producing confident-sounding falsehoods.

```
                  What the model knows
                  ┌────────────────────────────────────────┐
                  │  General world knowledge (from pretraining)  │
                  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
                  └────────────────────────────────────────┘
                         ↓
    Your question asks about THIS specific fact ─────► model guesses ─► hallucination
```

### Two types of hallucination

**Intrinsic hallucination** — the answer contradicts the retrieved context.

> Context: "The model was released in March 2024."
> Answer: "The model was released in **April** 2024."

**Extrinsic hallucination** — the answer adds information not present in the context.

> Context: "The model was released in March 2024."
> Answer: "The model was released in March 2024 **by a team of 50 researchers in San Francisco**."

Both are faithfulness failures. Intrinsic is worse (factually wrong); extrinsic is sneakier (can't be verified from the provided context).

> [!warning] High confidence ≠ high accuracy
> Models state hallucinations with the same fluency and confidence as correct answers. Never use the model's tone as a proxy for correctness.

---

## Faithfulness scoring without RAGAS

You can implement a faithfulness check directly using an LLM judge. The judge asks: "Can every claim in the answer be supported by the provided context?"

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is faithful to the provided context.

A faithful answer contains ONLY information that can be verified in the context.
An unfaithful answer adds facts, dates, names, or claims not present in the context.

Context:
{context}

Answer to evaluate:
{answer}

Decompose the answer into individual factual claims, then check each against the context.

Return JSON:
{{
  "claims": [
    {{"claim": "...", "supported": true/false, "evidence": "quote from context or 'not found'"}}
  ],
  "faithfulness_score": 0.0-1.0,
  "reasoning": "one sentence"
}}

JSON only:"""

def score_faithfulness(answer: str, context: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        }],
        temperature=0.0,
        max_tokens=500,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# Test: intrinsic hallucination
context = "GPT-4 was released by OpenAI in March 2023. It introduced vision capabilities."
answer = "GPT-4 was released in April 2023 and supports voice input."

result = score_faithfulness(answer, context)
print(f"Faithfulness: {result['faithfulness_score']:.2f}")
for claim in result["claims"]:
    status = "✓" if claim["supported"] else "✗"
    print(f"  {status} {claim['claim']}")
    if not claim["supported"]:
        print(f"    Evidence: {claim['evidence']}")
# Output:
# Faithfulness: 0.00
#   ✗ GPT-4 was released in April 2023
#     Evidence: not found (context says March 2023)
#   ✗ GPT-4 supports voice input
#     Evidence: not found (context mentions vision, not voice)
```

---

## NLI-based faithfulness (no API cost)

Natural Language Inference models classify whether a hypothesis is entailed by, neutral to, or contradicts a premise. You can use them to check each answer sentence against retrieved context without LLM API calls.

```python
from transformers import pipeline

# Load a cross-encoder NLI model (runs locally, ~500MB)
nli = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=-1  # CPU; use 0 for GPU
)

def sentence_faithfulness_nli(answer: str, context: str) -> dict:
    """Check each sentence of the answer against context using NLI."""
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]

    results = []
    for sent in sentences:
        label_scores = nli(f"{context} [SEP] {sent}", top_k=3)
        label_map = {item["label"]: item["score"] for item in label_scores}

        entailment = label_map.get("ENTAILMENT", 0)
        contradiction = label_map.get("CONTRADICTION", 0)
        neutral = label_map.get("NEUTRAL", 0)

        results.append({
            "sentence": sent,
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
            "verdict": "faithful" if entailment > 0.5 else "hallucinated" if contradiction > 0.3 else "unverifiable"
        })

    faithful_count = sum(1 for r in results if r["verdict"] == "faithful")
    return {
        "score": faithful_count / len(results) if results else 0.0,
        "sentences": results
    }

# Example
context = "Python was created by Guido van Rossum and first released in 1991."
answer = "Python was created by Guido van Rossum in 1991. It is the most popular language in the world."

result = sentence_faithfulness_nli(answer, context)
print(f"NLI Faithfulness: {result['score']:.2f}")
for s in result["sentences"]:
    print(f"  [{s['verdict']:12}] {s['sentence']}")
# Output:
# NLI Faithfulness: 0.50
#   [faithful     ] Python was created by Guido van Rossum in 1991.
#   [unverifiable ] It is the most popular language in the world.
```

> [!tip] NLI vs LLM judge tradeoffs
> NLI: ~5ms per sentence, runs locally, no API cost, but less precise on complex multi-part claims.
> LLM judge: accurate on subtle hallucinations, understands paraphrase, but costs $0.002–$0.01 per check and adds 1–3s latency. Use NLI for high-volume filtering; LLM judge for flagged cases.

---

## Hallucination rate measurement

Track hallucination rate across your eval set, not just individual examples.

```python
from dataclasses import dataclass
import statistics

@dataclass
class HallucinationResult:
    question: str
    answer: str
    context: str
    faithfulness_score: float
    hallucinated_claims: list[str]

def measure_hallucination_rate(
    examples: list[dict],
    threshold: float = 0.8
) -> dict:
    """
    examples: list of {"question", "answer", "context"} dicts
    threshold: faithfulness below this is flagged as hallucination
    """
    results = []
    for ex in examples:
        scored = score_faithfulness(ex["answer"], ex["context"])
        bad_claims = [
            c["claim"] for c in scored.get("claims", [])
            if not c["supported"]
        ]
        results.append(HallucinationResult(
            question=ex["question"],
            answer=ex["answer"],
            context=ex["context"],
            faithfulness_score=scored["faithfulness_score"],
            hallucinated_claims=bad_claims
        ))

    scores = [r.faithfulness_score for r in results]
    hallucinated = [r for r in results if r.faithfulness_score < threshold]

    return {
        "n": len(results),
        "hallucination_rate": len(hallucinated) / len(results),
        "avg_faithfulness": statistics.mean(scores),
        "p10_faithfulness": sorted(scores)[int(len(scores) * 0.10)],
        "worst_cases": sorted(hallucinated, key=lambda r: r.faithfulness_score)[:3]
    }

# Example
examples = [
    {
        "question": "When was Python released?",
        "answer": "Python was first released in 1991 by Guido van Rossum.",
        "context": "Python is a programming language created by Guido van Rossum, first released in 1991."
    },
    {
        "question": "What is RAG?",
        "answer": "RAG is a technique that combines retrieval with generation. It was invented at MIT in 2019.",
        "context": "RAG (Retrieval-Augmented Generation) combines retrieval systems with language model generation."
    }
]

# report = measure_hallucination_rate(examples)
# print(f"Hallucination rate: {report['hallucination_rate']:.1%}")
```

---

## Suppressing hallucination: prompt engineering

The single most effective intervention is a strong grounding instruction in the system prompt.

```python
GROUNDING_SYSTEM_PROMPT = """You are a precise assistant that answers ONLY from provided context.

Rules:
1. Answer only using information explicitly stated in the context.
2. If the context does not contain enough information, say: "The provided context does not contain information about this."
3. Do not add background knowledge, general facts, or assumptions.
4. Quote the context directly when stating specific facts.
5. Never invent dates, names, numbers, or statistics.

Violating these rules is worse than saying "I don't know."
"""

def grounded_answer(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.0,
        max_tokens=300
    )
    return response.choices[0].message.content

# Compare with and without grounding
question = "What programming languages did the company use?"
context = "The company built their backend API using Python and Go."

# Without grounding: model may add "they also used JavaScript for the frontend"
# With grounding: model says "Python and Go" and nothing else
answer = grounded_answer(question, context)
print(answer)
# "According to the context, the company built their backend API using Python and Go."
```

---

## Suppressing hallucination: retrieval quality

Faithfulness problems often start upstream in retrieval. A faithfulness score below 0.75 usually means one of these:

```python
def diagnose_faithfulness_failure(
    faithfulness_score: float,
    context_precision: float,
    context_recall: float
) -> str:
    if faithfulness_score < 0.7 and context_precision > 0.8:
        return "Model is ignoring good context — strengthen grounding prompt"
    if faithfulness_score < 0.7 and context_precision < 0.6:
        return "Irrelevant chunks retrieved — model fills gaps with parametric knowledge"
    if faithfulness_score < 0.7 and context_recall < 0.6:
        return "Key context missing — model guesses the missing parts"
    if faithfulness_score < 0.7:
        return "Low faithfulness, unclear cause — check both prompt and retrieval"
    return "Faithfulness acceptable"

# Diagnosis helps route the fix:
# - Bad retrieval? → better embedding model, reranking, hybrid search
# - Good retrieval, bad generation? → stronger prompt, lower temperature, smaller context window
print(diagnose_faithfulness_failure(0.62, 0.55, 0.80))
# "Irrelevant chunks retrieved — model fills gaps with parametric knowledge"
```

> [!warning] Faithfulness vs accuracy
> A faithful answer is one that only uses the context — it can still be factually wrong if the context is wrong. Faithfulness measures adherence to context, not ground truth. Both matter and they are measured separately.

---

[[02-ragas-framework]] | [[04-relevance-metrics]]
