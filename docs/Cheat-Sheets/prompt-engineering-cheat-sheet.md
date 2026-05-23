# Prompt Engineering Cheat Sheet

Core patterns for reliable LLM outputs.

---

## Zero-shot

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Classify this email as spam or not spam:\n{email}"},
]
```

## Few-shot

```python
messages = [
    {"role": "system", "content": "Classify emails as spam or not spam."},
    {"role": "user", "content": "Win $1000 now! Click here!"},
    {"role": "assistant", "content": "spam"},
    {"role": "user", "content": "Meeting at 3pm tomorrow"},
    {"role": "assistant", "content": "not spam"},
    {"role": "user", "content": "{new_email}"},  # The actual query
]
```

## Chain-of-thought

```python
SYSTEM = """Solve the problem step by step:
1. Identify what is known
2. Identify what needs to be found
3. Set up the solution
4. Calculate the answer
5. Verify the answer makes sense"""
```

## Structured output

```python
SYSTEM = """Extract information and return ONLY valid JSON matching this schema:
{
  "name": string,
  "date": "YYYY-MM-DD",
  "amount": number
}
Do not include any explanation or markdown formatting."""
```

## Role prompting

```python
roles = {
    "expert": "You are a senior Python engineer with 10 years of experience. Be specific and concise.",
    "teacher": "You are a patient teacher explaining to a beginner. Use analogies and simple examples.",
    "critic": "You are a rigorous code reviewer. Point out issues, don't praise strengths.",
}
```

## Constraints that work

```python
# Length control
"Respond in exactly 3 bullet points."
"Write a summary in 2-3 sentences maximum."
"Answer in under 50 words."

# Format control
"Return ONLY a Python dict, no explanation."
"Use markdown formatting with headers."
"Do not use the words 'certainly', 'absolutely', or 'of course'."

# Negative constraints
"Do not hallucinate. If you don't know, say 'I don't know'."
"Do not speculate beyond what is stated in the context."
```

## Temperature guide

| Task | Temperature |
|------|------------|
| Classification, extraction | 0.0 |
| Q&A, factual tasks | 0.0–0.2 |
| Chat, balanced responses | 0.3–0.5 |
| Creative writing, brainstorming | 0.7–1.0 |
| Poetry, highly creative | 1.0–1.5 |

## Prompt debugging checklist

When the model gives wrong outputs:
1. Is the instruction unambiguous? Rephrase it.
2. Does the model see a contradiction in the instructions?
3. Add a few-shot example that demonstrates the expected output.
4. Move the most important instruction to the end (recency bias).
5. Try temperature=0 to remove sampling randomness from the equation.
6. Try a larger model to check if the issue is model capability vs. prompt.

## Anti-patterns to avoid

```python
# Vague — don't do this
"Write a good summary."

# Specific — do this instead
"Write a 3-sentence summary focusing on the main argument, key evidence, and conclusion."

# Contradictory — don't do this
"Be concise and comprehensive."

# Clear priority — do this instead
"Be concise (under 100 words). Cover only the most critical point."
```
