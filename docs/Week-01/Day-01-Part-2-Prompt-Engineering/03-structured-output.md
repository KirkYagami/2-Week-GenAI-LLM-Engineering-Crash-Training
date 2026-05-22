# Structured Output

Free-form text answers are fine for chat. For production pipelines — data extraction, classification, API responses — you need output you can parse reliably. Structured output eliminates the `json.loads()` try/except pattern that breaks at 2 AM.

## Learning objectives

- Extract JSON reliably using prompt techniques and native API support
- Use Pydantic models with OpenAI and Anthropic structured output APIs
- Handle output validation and retry logic
- Choose the right approach for your use case and model

---

## Why structured output is hard (and how APIs solved it)

Without enforcement, LLMs produce valid-looking-but-broken JSON:

```
Common failures:
- Trailing commas: {"name": "Alice",}
- Single quotes: {'name': 'Alice'}
- Missing quotes: {name: "Alice"}
- Truncated mid-value (hit max_tokens): {"name": "Al
- Wrapped in markdown: ```json\n{"name": "Alice"}\n```
- Extra prose: "Here's the JSON: {...}"
```

Modern APIs solve this at the grammar level — the model's token sampling is constrained to only produce valid tokens at each position in the JSON structure.

---

## Method 1: Prompt-based JSON extraction

Works with any model. Less reliable, but sufficient for simple cases.

```python
import openai
import json
import re

client = openai.OpenAI()

def extract_json_prompt(text: str, schema_description: str) -> dict:
    """
    Prompt the model to extract structured data.
    Include schema and output instructions explicitly.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You extract structured data from text. Respond with valid JSON only. No markdown, no explanation.",
            },
            {
                "role": "user",
                "content": f"""Extract the following fields from the text below.

Schema:
{schema_description}

Text:
{text}

Respond with a JSON object containing exactly these fields. Use null for missing values.""",
            },
        ],
        temperature=0.0,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()

    # Clean up common prompt-based issues
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)

schema = """
{
  "name": "string (person's full name)",
  "email": "string (email address) or null",
  "company": "string (company name) or null",
  "role": "string (job title) or null",
  "sentiment": "POSITIVE | NEGATIVE | NEUTRAL"
}"""

text = "Thanks for reaching out! I'm Jennifer Walsh, VP of Engineering at CloudStack (jwelsh@cloudstack.io). Really excited about the partnership."

result = extract_json_prompt(text, schema)
print(json.dumps(result, indent=2))
# {
#   "name": "Jennifer Walsh",
#   "email": "jwelsh@cloudstack.io",
#   "company": "CloudStack",
#   "role": "VP of Engineering",
#   "sentiment": "POSITIVE"
# }
```

---

## Method 2: OpenAI Structured Outputs (native API)

OpenAI's structured outputs use Pydantic models or JSON Schema to constrain generation at the grammar level — guaranteed valid output.

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
import openai

client = openai.OpenAI()

# Define your schema with Pydantic
class ContactInfo(BaseModel):
    name: str = Field(description="Full name of the person")
    email: Optional[str] = Field(default=None, description="Email address")
    company: Optional[str] = Field(default=None, description="Company or organization")
    role: Optional[str] = Field(default=None, description="Job title or role")
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        description="Overall sentiment of the message"
    )

def extract_contact_openai(text: str) -> ContactInfo:
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract contact information and sentiment from the text."},
            {"role": "user", "content": text},
        ],
        response_format=ContactInfo,  # Pydantic model as schema
        temperature=0.0,
    )
    return response.choices[0].message.parsed  # Already a ContactInfo instance

# Usage
contact = extract_contact_openai(
    "Hi, this is Marcus Green from TechVenture (m.green@techventure.io). Looking forward to our call."
)
print(contact.name)       # Marcus Green
print(contact.email)      # m.green@techventure.io
print(contact.sentiment)  # POSITIVE
print(type(contact))      # <class 'ContactInfo'>

# Access as dict if needed
print(contact.model_dump())
```

### JSON Schema without Pydantic

```python
import json

# Define schema manually (useful for dynamic schemas or non-Python environments)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract invoice data from the text."},
        {"role": "user", "content": "Invoice #INV-2024-8821, $1,250.00, due March 15, 2024. Client: Acme Corp."},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "invoice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "invoice_number": {"type": "string"},
                    "amount": {"type": "number"},
                    "due_date": {"type": "string"},
                    "client_name": {"type": "string"},
                },
                "required": ["invoice_number", "amount", "due_date", "client_name"],
                "additionalProperties": False,
            },
        },
    },
    temperature=0.0,
)

invoice = json.loads(response.choices[0].message.content)
print(json.dumps(invoice, indent=2))
```

---

## Method 3: Anthropic Structured Outputs (native API)

Anthropic released native structured outputs (generally available as of late 2025) using `client.messages.parse()`.

```python
from pydantic import BaseModel
from typing import Optional, Literal
import anthropic

client = anthropic.Anthropic()

class SupportTicket(BaseModel):
    category: Literal["BILLING", "TECHNICAL", "ACCOUNT", "FEATURE_REQUEST", "OTHER"]
    priority: Literal["LOW", "MEDIUM", "HIGH", "URGENT"]
    summary: str
    customer_name: Optional[str] = None
    needs_callback: bool

def classify_ticket(ticket_text: str) -> SupportTicket:
    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Classify this support ticket and extract key information.

Ticket:
{ticket_text}""",
            }
        ],
        output_format=SupportTicket,
    )
    return response.parsed_output

# Test it
ticket = """
Hi, I'm Sarah from Acme. Our entire team can't access the dashboard since this morning.
We have a board presentation in 2 hours and we NEED this fixed immediately.
Please call me at 555-0192. This is costing us real money.
"""

result = classify_ticket(ticket)
print(f"Category: {result.category}")      # TECHNICAL
print(f"Priority: {result.priority}")      # URGENT
print(f"Summary: {result.summary}")
print(f"Customer: {result.customer_name}") # Sarah
print(f"Callback: {result.needs_callback}")# True
```

---

## Complex nested schemas

```python
from pydantic import BaseModel
from typing import Optional

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class Invoice(BaseModel):
    invoice_number: str
    client_name: str
    client_email: Optional[str] = None
    issue_date: str
    due_date: str
    line_items: list[LineItem]
    subtotal: float
    tax_rate: float
    tax_amount: float
    total_amount: float
    paid: bool

invoice_text = """
INVOICE #2024-0892
Acme Corporation | billing@acme.com
Issued: January 15, 2024 | Due: February 14, 2024

Services:
- API Development (40 hrs @ $150/hr): $6,000.00
- UI Design (20 hrs @ $120/hr): $2,400.00
- DevOps Setup (8 hrs @ $180/hr): $1,440.00

Subtotal: $9,840.00
Tax (10%): $984.00
Total: $10,824.00
Status: UNPAID
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract invoice data accurately."},
        {"role": "user", "content": invoice_text},
    ],
    response_format=Invoice,
    temperature=0.0,
)

invoice = response.choices[0].message.parsed
print(f"Invoice: {invoice.invoice_number}")
print(f"Total: ${invoice.total_amount:,.2f}")
print(f"Line items: {len(invoice.line_items)}")
for item in invoice.line_items:
    print(f"  - {item.description}: ${item.total:,.2f}")
```

---

## Handling validation and retries

Even with native structured output, you should handle edge cases:

```python
import time

def extract_with_retry(
    text: str,
    schema: type[BaseModel],
    max_retries: int = 3,
    model: str = "gpt-4o",
) -> BaseModel | None:
    """Extract structured data with retry on parse failure."""
    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": "Extract the requested structured data."},
                    {"role": "user", "content": text},
                ],
                response_format=schema,
                temperature=0.0,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                # Model refused (stop_reason == "refusal")
                print(f"Model refused on attempt {attempt + 1}")
                return None
            return parsed
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            else:
                print(f"All {max_retries} attempts failed: {e}")
                return None
```

---

## Prompt-based vs. native structured output: which to use?

| | Prompt-based | Native (OpenAI/Anthropic) |
|--|---|---|
| **Reliability** | ~95% valid JSON | ~100% |
| **Model support** | Any model | GPT-4o, Claude Sonnet/Opus 4.x |
| **Complex nested schemas** | Breaks often | Handles reliably |
| **Custom validation logic** | Add manually | Via Pydantic validators |
| **Cost** | Standard | Slight token overhead from schema |
| **Use when** | Prototyping, simple schemas, open-source models | Production, complex schemas |

> [!warning] `response_format={"type": "json_object"}` is not structured output
> OpenAI's old JSON mode guarantees valid JSON but not schema compliance. Use `json_schema` or Pydantic via `client.beta.chat.completions.parse()` for schema-constrained output.

> [!warning] Structured output + streaming requires careful handling
> Streaming with structured output means you receive partial JSON. Only parse after the stream completes. Don't process partial JSON tokens.

---

> [!success] Key takeaway
> For production extraction pipelines, use Anthropic's `client.messages.parse()` or OpenAI's `client.beta.chat.completions.parse()` with a Pydantic model. Prompt-based JSON is for prototypes and open-source models.

[[02-chain-of-thought]] | [[04-system-prompts]]
