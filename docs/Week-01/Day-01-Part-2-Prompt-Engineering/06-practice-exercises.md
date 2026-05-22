# Practice Exercises — Prompt Engineering

Three levels: warm-up (tweak an existing prompt), main (build a pipeline), stretch (evaluate and iterate).

---

## Warm-up: Prompt surgery

**Goal:** Fix broken prompts — identify the failure mode and correct it.

```python
import openai

client = openai.OpenAI()

def test_prompt(system: str, user: str, expected_format: str = "") -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    output = response.choices[0].message.content
    print(f"Output: {output[:200]}")
    return output

# Exercise 1: Fix this vague classification prompt
# Problem: the model returns different formats each time
BAD_SYSTEM_1 = "Classify emails."
BAD_USER_1 = "Subject: RE: Q3 Report - Urgent Review Needed\nBody: Please review the attached Q3 report ASAP. The board meeting is tomorrow."

print("=== BAD prompt output ===")
test_prompt(BAD_SYSTEM_1, BAD_USER_1)

# YOUR FIX: rewrite GOOD_SYSTEM_1 and GOOD_USER_1 here
GOOD_SYSTEM_1 = """Classify this email into exactly one category.
Categories: URGENT_ACTION | REVIEW_NEEDED | FYI | SPAM | OTHER
Respond with ONLY the category label."""

GOOD_USER_1 = BAD_USER_1

print("\n=== GOOD prompt output ===")
test_prompt(GOOD_SYSTEM_1, GOOD_USER_1)

# Exercise 2: Fix this extraction prompt that mixes content with instructions
BAD_PROMPT_2 = """
Extract the name from this text and also the email and the phone number
if there is one and also the company. The person's name is John Smith
and he works at Acme and his email is john@acme.com.
"""
# Problem: user data is mixed with schema instructions

print("\n=== Exercise 2: Fix the structure ===")
# Write a version using XML tags to separate schema from content
GOOD_PROMPT_2 = """Extract contact information from the text.

<schema>
name: string
email: string or null
phone: string or null
company: string or null
</schema>

<text>
The person's name is John Smith and he works at Acme and his email is john@acme.com.
</text>

Respond with JSON only."""

test_prompt("", GOOD_PROMPT_2)
```

---

## Main exercise: Product review analyzer

**Goal:** Build a complete review analysis pipeline combining zero-shot classification, structured output, and CoT for the borderline cases.

```python
import openai
import anthropic
from pydantic import BaseModel
from typing import Literal, Optional
import json

openai_client = openai.OpenAI()
anthropic_client = anthropic.Anthropic()

# Step 1: Define the output schema
class ReviewAnalysis(BaseModel):
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]
    rating_out_of_5: int
    key_positives: list[str]
    key_negatives: list[str]
    product_category: str
    would_recommend: bool
    one_line_summary: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]

# Step 2: Build the analyzer
def analyze_review(review_text: str) -> ReviewAnalysis:
    """
    Analyze a product review using structured output.
    Uses Claude for analysis with CoT for low-confidence cases.
    """
    # First pass: structured extraction
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a product review analyst. Extract structured insights from customer reviews.

For sentiment:
- POSITIVE: clearly satisfied overall
- NEGATIVE: clearly dissatisfied overall
- NEUTRAL: no strong sentiment
- MIXED: significant positives AND negatives

For confidence:
- HIGH: clear, unambiguous review
- MEDIUM: some ambiguity
- LOW: very short, sarcastic, or contradictory

For rating_out_of_5: infer from sentiment and content (not just star rating if mentioned)""",
            },
            {"role": "user", "content": f"Analyze this review:\n\n{review_text}"},
        ],
        response_format=ReviewAnalysis,
        temperature=0.0,
    )

    result = response.choices[0].message.parsed

    # If low confidence, use Claude with CoT for a second opinion
    if result.confidence == "LOW":
        print("Low confidence — running CoT verification...")
        verification = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"""Analyze this ambiguous product review carefully.

<review>
{review_text}
</review>

Think step by step:
1. What is the overall tone?
2. What specific positives and negatives are mentioned?
3. What would a reasonable person conclude about their satisfaction?

End with: FINAL VERDICT: [POSITIVE/NEGATIVE/NEUTRAL/MIXED] | Rating: [1-5]""",
            }],
        )
        print("CoT verification:", verification.content[0].text[:300])

    return result

# Step 3: Test with diverse reviews
reviews = [
    "Absolutely amazing product! Changed my life. 10/10 would recommend to everyone.",
    "Arrived damaged. Customer service refused to help. Worst experience ever.",
    "It's a product. It does what it's supposed to do. Nothing more.",
    "Love the design and build quality but the battery life is terrible. Pretty torn.",
    "fine i guess",  # Low confidence case
    "If you enjoy disappointment, this is the product for you!",  # Sarcasm case
]

for review in reviews:
    print(f"\n{'='*60}")
    print(f"Review: {review[:80]}")
    analysis = analyze_review(review)
    print(f"Sentiment: {analysis.sentiment} | Rating: {analysis.rating_out_of_5}/5 | Confidence: {analysis.confidence}")
    print(f"Summary: {analysis.one_line_summary}")
    if analysis.key_positives:
        print(f"Positives: {', '.join(analysis.key_positives)}")
    if analysis.key_negatives:
        print(f"Negatives: {', '.join(analysis.key_negatives)}")
```

---

## Main exercise: Prompt A/B tester

**Goal:** Build a systematic framework to compare prompt variants using real outputs.

```python
import openai
import random
from dataclasses import dataclass

client = openai.OpenAI()

@dataclass
class PromptVariant:
    name: str
    system: str
    user_template: str  # use {input} as placeholder

def run_ab_test(
    variants: list[PromptVariant],
    test_inputs: list[str],
    temperature: float = 0.0,
    n_trials: int = 1,
) -> dict:
    """
    Run each variant against each test input n_trials times.
    Returns all results for manual or automated evaluation.
    """
    results = {v.name: {} for v in variants}

    for test_input in test_inputs:
        for variant in variants:
            outputs = []
            for _ in range(n_trials):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": variant.system},
                        {"role": "user", "content": variant.user_template.format(input=test_input)},
                    ],
                    temperature=temperature,
                    max_tokens=200,
                )
                outputs.append(response.choices[0].message.content)
            results[variant.name][test_input] = outputs

    return results

# Define variants
v1 = PromptVariant(
    name="basic",
    system="You classify text.",
    user_template="Classify: {input}",
)

v2 = PromptVariant(
    name="structured",
    system="""You classify customer feedback into one of: PRAISE | COMPLAINT | QUESTION | SUGGESTION
Respond with only the category label.""",
    user_template="Customer feedback: {input}",
)

v3 = PromptVariant(
    name="few_shot",
    system="""You classify customer feedback. Respond with only: PRAISE | COMPLAINT | QUESTION | SUGGESTION

Examples:
"Love the new interface!" → PRAISE
"Why can't I export to Excel?" → QUESTION
"The app crashes on startup" → COMPLAINT
"Would be great to add dark mode" → SUGGESTION""",
    user_template="Feedback: {input}",
)

test_cases = [
    "This is the best app I've ever used!",
    "When will you add offline mode?",
    "My account data disappeared after the update.",
    "You should add a bulk export feature.",
    "Meh.",
]

results = run_ab_test([v1, v2, v3], test_cases)

# Print comparison
print(f"\n{'Input':<45} {'basic':<15} {'structured':<15} {'few_shot':<15}")
print("-" * 90)
for test_input in test_cases:
    row = f"{test_input[:44]:<45}"
    for variant_name in ["basic", "structured", "few_shot"]:
        output = results[variant_name][test_input][0][:14]
        row += f" {output:<15}"
    print(row)
```

---

## Stretch: Build an automated prompt evaluator

**Goal:** Use the model to evaluate its own outputs against a rubric. Known as LLM-as-judge.

```python
import openai
from pydantic import BaseModel

client = openai.OpenAI()

class EvaluationResult(BaseModel):
    accuracy_score: int       # 1-5
    clarity_score: int        # 1-5
    completeness_score: int   # 1-5
    format_score: int         # 1-5
    overall_score: int        # 1-5
    reasoning: str
    specific_issues: list[str]

def evaluate_response(
    prompt: str,
    response: str,
    rubric: str,
) -> EvaluationResult:
    """Use GPT-4o to score a model response against a rubric."""
    evaluation = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an objective evaluator of AI responses. Score responses strictly and identify specific issues.",
            },
            {
                "role": "user",
                "content": f"""Evaluate this AI response against the rubric.

<prompt>
{prompt}
</prompt>

<response>
{response}
</response>

<rubric>
{rubric}
</rubric>

Score each dimension 1-5 (5 = perfect, 1 = completely fails).""",
            },
        ],
        response_format=EvaluationResult,
        temperature=0.0,
    )
    return evaluation.choices[0].message.parsed

# Test
test_prompt = "Summarize the benefits of RAG in 3 bullet points for a non-technical audience."
test_response = """RAG stands for Retrieval-Augmented Generation. It works by:
• Searching a knowledge base for relevant information
• Feeding that information to the AI as context
• Generating a response grounded in your specific documents

Benefits include reduced hallucination, up-to-date information, and source attribution."""

rubric = """
- Accuracy: Are the stated benefits correct?
- Clarity: Is it understandable for a non-technical audience?
- Completeness: Does it cover 3 distinct benefits as requested?
- Format: Does it use bullet points as implied by the request?
"""

result = evaluate_response(test_prompt, test_response, rubric)
print(f"Accuracy:     {result.accuracy_score}/5")
print(f"Clarity:      {result.clarity_score}/5")
print(f"Completeness: {result.completeness_score}/5")
print(f"Format:       {result.format_score}/5")
print(f"Overall:      {result.overall_score}/5")
print(f"\nReasoning: {result.reasoning}")
if result.specific_issues:
    print(f"Issues: {result.specific_issues}")
```

---

## Expected outcomes

By the end of these exercises you should have:

- ✅ Identified and fixed 3 common prompt failure modes
- ✅ A working multi-step review analysis pipeline
- ✅ A prompt A/B testing framework you can reuse in any project
- ✅ An LLM-as-judge evaluation system

[[05-prompt-patterns]] | [[07-interview-questions]]
