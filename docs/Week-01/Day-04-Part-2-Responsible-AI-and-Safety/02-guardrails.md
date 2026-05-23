# Guardrails

Guardrails are the safety checks that run before and after every LLM call. They are not a single tool but a pipeline: validate input → run model → validate output → return (or block and explain).

## Learning objectives

- Implement input guardrails using moderation APIs and custom logic
- Build output guardrails for length, format, and content checks
- Design a guardrail pipeline with configurable thresholds
- Use NeMo Guardrails or Llama Guard as production alternatives

---

## The guardrail pipeline

```
User input
    │
    ▼
┌─────────────────────┐
│  Input guardrails   │ ← Block harmful inputs before paying for the LLM call
│  - Moderation API   │
│  - PII detection    │
│  - Length/format    │
│  - Adversarial scan │
└────────┬────────────┘
         │ Passes
         ▼
┌─────────────────────┐
│   LLM call          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Output guardrails  │ ← Catch problems before the user sees them
│  - Content policy   │
│  - Format validation│
│  - PII in output    │
│  - Factual sanity   │
└────────┬────────────┘
         │ Passes
         ▼
    User sees response
```

---

## Input guardrail: OpenAI Moderation API

OpenAI's Moderation API is free, fast (~100ms), and catches obvious harmful content. Use it as the first layer.

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def check_moderation(text: str) -> dict:
    """Run OpenAI moderation. Returns violation details if flagged."""
    response = client.moderations.create(input=text)
    result = response.results[0]

    flagged_categories = {
        cat: score
        for cat, score in result.category_scores.__dict__.items()
        if score > 0.05  # Only return categories with non-trivial scores
    }

    return {
        "flagged": result.flagged,
        "categories": {cat: result.categories.__dict__[cat] for cat in result.categories.__dict__ if result.categories.__dict__[cat]},
        "top_scores": dict(sorted(flagged_categories.items(), key=lambda x: -x[1])[:3])
    }

# Test
test_inputs = [
    "What is the best way to cook salmon?",
    "How do I whittle a knife?",   # Often falsely flagged — good test case
]

for text in test_inputs:
    result = check_moderation(text)
    status = "FLAGGED" if result["flagged"] else "PASS"
    print(f"[{status}] {text}")
    if result["categories"]:
        print(f"  Categories: {result['categories']}")
    if result["top_scores"]:
        print(f"  Top scores: {result['top_scores']}")
```

> [!warning] Moderation API false positive rate
> The moderation API has notable false positive rates on cooking, hunting, and security education content. Always log false positives and tune your response — blocking "how do I whittle a knife?" is a bad user experience. Consider asking users to rephrase before blocking outright.

---

## Input guardrail: PII detection

Before sending user input to an external LLM API, strip Personally Identifiable Information to avoid data privacy violations.

```python
import re
from dataclasses import dataclass

@dataclass
class PIIResult:
    original: str
    anonymized: str
    found_types: list[str]

# Lightweight regex-based PII detector
PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone_us": r'\b(\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
}

REPLACEMENTS = {
    "email": "[EMAIL]",
    "phone_us": "[PHONE]",
    "ssn": "[SSN]",
    "credit_card": "[CREDIT_CARD]",
    "ip_address": "[IP_ADDRESS]",
}

def detect_and_anonymize_pii(text: str) -> PIIResult:
    anonymized = text
    found_types = []

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found_types.append(pii_type)
            anonymized = re.sub(pattern, REPLACEMENTS[pii_type], anonymized, flags=re.IGNORECASE)

    return PIIResult(original=text, anonymized=anonymized, found_types=found_types)

# Test
test_text = "My email is john.doe@company.com and phone is 555-867-5309. SSN: 123-45-6789."
result = detect_and_anonymize_pii(test_text)
print(f"Found PII: {result.found_types}")
print(f"Anonymized: {result.anonymized}")
# Found PII: ['email', 'phone_us', 'ssn']
# Anonymized: My email is [EMAIL] and phone is [PHONE]. SSN: [SSN].
```

For production, use Microsoft Presidio (`presidio-analyzer`) which handles 20+ entity types with named entity recognition:

```python
# pip install presidio-analyzer presidio-anonymizer
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def presidio_anonymize(text: str, language: str = "en") -> str:
    results = analyzer.analyze(text=text, language=language)
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text

# text = presidio_anonymize("John Smith, DOB 1985-03-15, lives at 123 Main St.")
# "PERSON, DATE_TIME, lives at LOCATION."
```

---

## Output guardrail: format and length validation

```python
import json
from typing import Any

def validate_output_format(
    output: str,
    expected_format: str = "text",
    max_length: int = 2000,
    min_length: int = 10
) -> dict:
    """Validate model output meets format and length requirements."""
    issues = []

    # Length checks
    if len(output) < min_length:
        issues.append(f"Response too short ({len(output)} chars, min {min_length})")
    if len(output) > max_length:
        issues.append(f"Response too long ({len(output)} chars, max {max_length})")

    # Format-specific checks
    if expected_format == "json":
        try:
            json.loads(output)
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON: {e}")

    if expected_format == "markdown_table":
        if "|" not in output or "---" not in output:
            issues.append("Expected markdown table format not found")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "length": len(output)
    }

def validate_no_pii_in_output(output: str) -> dict:
    """Ensure the model didn't include PII in its response."""
    result = detect_and_anonymize_pii(output)
    return {
        "pii_detected": bool(result.found_types),
        "types": result.found_types
    }
```

---

## Building a complete guardrail pipeline

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GuardrailConfig:
    check_input_moderation: bool = True
    check_input_pii: bool = True
    strip_pii_before_llm: bool = True
    max_input_length: int = 4000
    check_output_pii: bool = True
    check_output_moderation: bool = False  # Usually too slow for real-time
    max_output_length: int = 2000
    block_message: str = "I'm unable to respond to that request."

@dataclass
class GuardrailResult:
    blocked: bool
    block_reason: Optional[str]
    final_input: str    # Possibly anonymized
    final_output: Optional[str]
    checks_run: list[str]

def run_with_guardrails(
    user_input: str,
    system_prompt: str,
    config: GuardrailConfig = None
) -> GuardrailResult:
    if config is None:
        config = GuardrailConfig()

    checks_run = []
    processed_input = user_input

    # --- Input guardrails ---

    # Length check
    if len(user_input) > config.max_input_length:
        return GuardrailResult(
            blocked=True,
            block_reason=f"Input too long ({len(user_input)} chars)",
            final_input=user_input,
            final_output=None,
            checks_run=["input_length"]
        )
    checks_run.append("input_length")

    # Moderation
    if config.check_input_moderation:
        mod_result = check_moderation(user_input)
        checks_run.append("input_moderation")
        if mod_result["flagged"]:
            return GuardrailResult(
                blocked=True,
                block_reason=f"Content policy violation: {list(mod_result['categories'].keys())}",
                final_input=user_input,
                final_output=None,
                checks_run=checks_run
            )

    # PII detection
    if config.check_input_pii:
        pii_result = detect_and_anonymize_pii(user_input)
        checks_run.append("input_pii")
        if pii_result.found_types and config.strip_pii_before_llm:
            processed_input = pii_result.anonymized

    # --- LLM call ---
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": processed_input}
        ],
        temperature=0.0,
        max_tokens=500
    )
    output = response.choices[0].message.content
    checks_run.append("llm_call")

    # --- Output guardrails ---
    format_check = validate_output_format(output, max_length=config.max_output_length)
    checks_run.append("output_format")
    if not format_check["valid"]:
        # Don't block — return with warning instead
        output = output[:config.max_output_length]

    if config.check_output_pii:
        out_pii = validate_no_pii_in_output(output)
        checks_run.append("output_pii")
        if out_pii["pii_detected"]:
            output = detect_and_anonymize_pii(output).anonymized

    return GuardrailResult(
        blocked=False,
        block_reason=None,
        final_input=processed_input,
        final_output=output,
        checks_run=checks_run
    )

# Usage
config = GuardrailConfig(
    check_input_moderation=True,
    check_input_pii=True,
    strip_pii_before_llm=True,
    max_input_length=2000
)

result = run_with_guardrails(
    user_input="My email is john@example.com. What is RAG?",
    system_prompt="You are a helpful AI assistant.",
    config=config
)

print(f"Blocked: {result.blocked}")
print(f"Checks: {result.checks_run}")
print(f"Input sent to LLM: {result.final_input}")
print(f"Output: {result.final_output}")
# Input sent to LLM: "My email is [EMAIL]. What is RAG?"
```

> [!tip] Fail open vs fail closed
> Decide upfront: if a guardrail check fails (throws an exception), do you block the request (fail closed) or let it through (fail open)? For content safety: fail closed. For format checks: fail open and log. Never silently drop requests — always return a message or log the block.

---

[[01-jailbreaks-and-prompt-injection]] | [[03-content-filtering]]
