# System Prompts

The system prompt is your most powerful lever. It runs before every user message, shaping the model's persona, scope, style, and constraints. A well-designed system prompt turns a general-purpose model into a focused specialist. A poorly written one creates unpredictable, inconsistent behavior at scale.

## Learning objectives

- Understand how system prompts differ from user messages architecturally
- Design system prompts that are robust, specific, and hard to bypass
- Separate stable instructions (system) from variable content (user)
- Write system prompts for three common application types: assistant, extractor, and classifier

---

## What the system prompt is (and isn't)

In the OpenAI and Anthropic APIs, the messages array has three roles: `system`, `user`, and `assistant`. The system role is processed first, with higher priority, before the user message.

```python
import openai

client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",   # ← highest priority, processed first
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",     # ← user's message
            "content": "Who are you?",
        },
    ],
    max_tokens=100,
)
```

**What the system prompt is:**
- A persistent instruction that frames every subsequent interaction
- The place to specify persona, constraints, output format, and scope
- Processed at the same attention level as user content — it is NOT a hard override

**What the system prompt is NOT:**
- An unbreakable security barrier. Determined users can often work around it.
- A way to permanently hide information from the model (it's all in the context window)
- Separate from "memory" — it's just tokens, processed the same way as everything else

---

## System prompt anatomy

```python
SYSTEM_PROMPT_TEMPLATE = """
{PERSONA / ROLE}
You are [name], a [role description] for [company/context].

{TASK SCOPE}
Your job is to [primary function]. You help users with:
- [capability 1]
- [capability 2]
- [capability 3]

{CONSTRAINTS}
You do NOT:
- [restriction 1]
- [restriction 2]

{BEHAVIOR RULES}
When [situation], always [action].
When [edge case], respond with [specific behavior].

{OUTPUT FORMAT}
Always respond in [format]. Use [structure].
Keep responses [length guideline].

{TONE}
Your tone is [adjectives]. You [style guidance].
"""
```

---

## Example: Customer support assistant

```python
import anthropic

client = anthropic.Anthropic()

SUPPORT_SYSTEM_PROMPT = """You are Aria, a customer support specialist for ShopFlow, an e-commerce platform.

Your responsibilities:
- Answer questions about orders, shipping, returns, and account issues
- Help customers track packages using their order ID
- Process refund requests by collecting required information
- Escalate billing disputes to the billing team

You do NOT:
- Access or modify actual order data (you don't have database access)
- Make promises about refund timelines beyond our 5–7 business day policy
- Discuss competitor products or pricing
- Provide technical support for third-party integrations

When a customer is frustrated, acknowledge their frustration before addressing the issue.
When you cannot resolve an issue, say: "Let me connect you with our specialized team."
When asked for an order status, ask for the order ID before attempting to help.

Keep responses concise — 2–3 short paragraphs maximum. Use plain language.
Do not use jargon or internal terminology. Be warm but professional."""

def support_chat(user_message: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SUPPORT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text

# Test
test_messages = [
    "Where is my order?",
    "I've been waiting 3 weeks and nothing has arrived. This is unacceptable.",
    "Can you help me hack into someone's account?",
    "How does ShopFlow compare to Shopify?",
]
for msg in test_messages:
    print(f"\nUser: {msg}")
    print(f"Aria: {support_chat(msg)[:200]}")
```

---

## Example: Structured data extractor

For extraction pipelines, the system prompt defines the schema contract.

```python
import openai
import json

client = openai.OpenAI()

EXTRACTOR_SYSTEM = """You are a data extraction engine. Your only job is to extract structured information from text.

Output rules:
- Always respond with valid JSON only — no prose, no explanation, no markdown formatting
- Use null for any field that cannot be found in the text
- Do not infer or guess values — only extract what is explicitly stated
- For dates, use ISO 8601 format (YYYY-MM-DD)
- For monetary values, extract the numeric value only (no currency symbols)

If the text contains no extractable information, return: {"error": "no_extractable_content"}"""

def extract(text: str, schema: dict) -> dict:
    prompt = f"""Extract the following fields from the text.

Schema: {json.dumps(schema, indent=2)}

Text: {text}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EXTRACTOR_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return json.loads(response.choices[0].message.content)

# Test
result = extract(
    text="Meeting with John Smith (john.smith@acme.com) on March 15, 2024 at 2:30 PM. Budget: $5,000.",
    schema={
        "person_name": "string",
        "email": "string",
        "meeting_date": "date",
        "meeting_time": "string",
        "budget": "number"
    },
)
print(json.dumps(result, indent=2))
```

---

## Injecting dynamic context into system prompts

Don't put everything in the user message. Use the system prompt for context that applies to the whole session.

```python
from datetime import datetime

def build_system_prompt(user_id: str, user_tier: str, user_name: str) -> str:
    """Build a personalized system prompt at request time."""
    return f"""You are a financial advisor assistant for WealthPath.

Current context:
- User: {user_name} (ID: {user_id})
- Account tier: {user_tier}
- Current date: {datetime.now().strftime("%B %d, %Y")}

{'You have access to premium features including portfolio analysis.' if user_tier == 'premium' else 'You can answer general questions. For portfolio analysis, recommend upgrading to Premium.'}

Compliance rules:
- Do not make specific investment recommendations (e.g., "buy stock X")
- Always recommend consulting a licensed financial advisor for major decisions
- Never discuss specific client account balances or transaction history"""

# Different prompts for different users
for user in [
    {"id": "usr_001", "tier": "premium", "name": "Alice"},
    {"id": "usr_002", "tier": "free", "name": "Bob"},
]:
    prompt = build_system_prompt(**user)
    print(f"\nSystem prompt for {user['name']} ({user['tier']}):")
    print(prompt[:200] + "...")
```

---

## Multi-turn conversations with system prompts

The system prompt persists across all turns. Build it for the entire session, not just the first message.

```python
import anthropic

client = anthropic.Anthropic()

def run_conversation(system: str) -> None:
    """Simple multi-turn chat loop with a system prompt."""
    messages = []

    print("Chat started (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,  # system prompt is always present
            messages=messages,
        )

        assistant_message = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_message})
        print(f"\nAssistant: {assistant_message}")

# Uncomment to run interactively:
# run_conversation("You are a Python tutor. Explain concepts with code examples.")
```

---

## Prompt caching system prompts (cost optimization)

Long system prompts cost tokens on every API call. Cache them to pay only once.

```python
import anthropic

client = anthropic.Anthropic()

LONG_SYSTEM_PROMPT = """[A 2,000-token system prompt with detailed instructions...]
""" * 50  # Simulating a long prompt

# Without caching: pay for 2,000 tokens × every API call
# With caching: pay for 2,000 tokens once, then ~10% on subsequent calls

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": LONG_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # mark for caching
        }
    ],
    messages=[{"role": "user", "content": "Hello, how can you help me?"}],
)

print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
print(f"Cache read tokens:     {response.usage.cache_read_input_tokens}")
print(f"Regular input tokens:  {response.usage.input_tokens}")
# Second call: cache_creation = 0, cache_read = ~2000 (90% cheaper)
```

---

## Common system prompt mistakes

> [!warning] Vague instructions produce inconsistent behavior
> "Be helpful and professional" is not actionable. "Respond in 2–3 sentences. Use active voice. Never use the word 'certainly'." is.

> [!warning] Over-restricting kills usefulness
> A system prompt that says "don't discuss anything controversial" will cause the model to refuse legitimate questions. Be specific about what is restricted and why.

> [!warning] Putting variable content in the system prompt
> If your system prompt changes with every request (different user name, different document), you lose prompt caching benefits. Put stable content in system, variable content in user messages.

> [!warning] Assuming system prompts are secret
> Users can often extract system prompt content through clever prompting. Don't put API keys, sensitive business logic, or anything truly confidential in the system prompt.

---

> [!success] Key takeaway
> Write system prompts like you're writing a job description for a specialist. Be specific about role, responsibilities, what to do, what NOT to do, and the output format. Test it against adversarial inputs before shipping.

[[03-structured-output]] | [[05-prompt-patterns]]
