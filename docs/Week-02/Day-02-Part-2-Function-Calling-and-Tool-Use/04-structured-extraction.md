# Structured Extraction

Extraction is the most common use case for tool calling in production: take a messy email, ticket, document, or web page and pull out typed, validated fields. This note covers two approaches — raw tool calling and LangChain's `with_structured_output` wrapper — and builds a realistic extraction pipeline.

## Learning objectives

- Build Pydantic models for extraction schemas
- Use `with_structured_output` for clean, type-safe extraction
- Handle optional fields, nested objects, and lists
- Implement batch extraction with error recovery
- Validate and post-process extracted data

---

## Pydantic models as extraction schemas

```python
from pydantic import BaseModel, Field
from typing import Optional

class Address(BaseModel):
    street: Optional[str] = None
    city: str
    state: Optional[str] = None
    country: str = "US"

class ContactInfo(BaseModel):
    full_name: str = Field(description="Person's full name")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number with country code")
    company: Optional[str] = Field(default=None, description="Employer or organization")
    address: Optional[Address] = Field(default=None, description="Physical address if mentioned")
    role: Optional[str] = Field(default=None, description="Job title or role")

class Invoice(BaseModel):
    vendor: str
    invoice_number: Optional[str] = None
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    due_date: Optional[str] = Field(default=None, description="Payment due date in YYYY-MM-DD")
    total: float = Field(description="Total amount in USD")
    items: list[str] = Field(default_factory=list, description="List of line item descriptions")
    paid: bool = False
```

---

## `with_structured_output` — the clean approach

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, Literal

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

class SupportTicket(BaseModel):
    """Extracted fields from a customer support ticket."""
    category: Literal["billing", "technical", "account", "shipping", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    product_mentioned: Optional[str] = Field(default=None, description="Product name if mentioned")
    order_id: Optional[str] = Field(default=None, description="Order or transaction ID if present")
    sentiment: Literal["positive", "neutral", "negative", "very_negative"]
    summary: str = Field(description="One-sentence summary of the issue")
    action_required: bool = Field(description="Whether immediate action is needed")

extractor = llm.with_structured_output(SupportTicket)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured data from customer support tickets."),
    ("human", "{ticket}")
])

chain = prompt | extractor

tickets = [
    "ORDER-4521: My laptop stopped charging after 2 weeks. I need a replacement ASAP or I'm disputing the charge!",
    "Hi, I just wanted to say your team was super helpful yesterday. Issue resolved, thanks!",
    "Can't log into my account since the password reset yesterday. No email received.",
]

for t in tickets:
    result = chain.invoke({"ticket": t})
    print(f"\nTicket: {t[:60]}...")
    print(f"  Category: {result.category} | Priority: {result.priority}")
    print(f"  Sentiment: {result.sentiment} | Action: {result.action_required}")
    print(f"  Summary: {result.summary}")
    if result.order_id:
        print(f"  Order ID: {result.order_id}")
```

---

## Batch extraction with error recovery

```python
import os
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Optional
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ProductMention(BaseModel):
    name: str
    price: Optional[float] = None
    sentiment: str  # "positive", "negative", "neutral"
    features_mentioned: list[str] = []

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_product_mention",
        "description": "Extract product information from text.",
        "parameters": ProductMention.model_json_schema()  # Generate schema from Pydantic
    }
}

def extract_with_retry(text: str, max_retries: int = 2) -> ProductMention | None:
    messages = [
        {"role": "system", "content": "Extract product mention details from the text."},
        {"role": "user", "content": text}
    ]

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=[EXTRACT_TOOL],
                tool_choice={"type": "function", "function": {"name": "extract_product_mention"}},
            )
            tool_call = response.choices[0].message.tool_calls[0]
            raw = json.loads(tool_call.function.arguments)
            return ProductMention(**raw)  # Pydantic validates types

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            if attempt < max_retries:
                messages.append({"role": "user", "content": f"The previous output was malformed: {e}. Try again."})
                continue
            print(f"  Extraction failed after {max_retries + 1} attempts: {e}")
            return None

texts = [
    "The Sony WH-1000XM5 headphones at $349 have incredible noise cancellation. Best purchase I've made.",
    "Bought this $25 cable and it stopped working in a week. Total garbage.",
    "Just upgraded my home office setup today — very happy with the monitor.",
]

print(f"{'Text':<55} {'Product':<25} {'Price':>8} {'Sentiment'}")
print("-" * 100)
for t in texts:
    result = extract_with_retry(t)
    if result:
        print(f"{t[:55]:<55} {result.name[:25]:<25} {str(result.price or 'N/A'):>8} {result.sentiment}")
```

---

## Nested extraction: complex schemas

```python
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

class LineItem(BaseModel):
    description: str
    quantity: int = 1
    unit_price: float
    total: float

class InvoiceExtraction(BaseModel):
    vendor_name: str
    vendor_email: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: str = Field(description="Date in YYYY-MM-DD format")
    due_date: Optional[str] = Field(default=None, description="Due date in YYYY-MM-DD format")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax_rate: Optional[float] = Field(default=None, description="Tax rate as decimal, e.g. 0.08 for 8%")
    total_amount: float
    currency: str = "USD"
    payment_status: str = Field(description="paid, pending, or overdue")

extractor = llm.with_structured_output(InvoiceExtraction)

invoice_text = """
INVOICE #INV-2025-0847
From: Acme Supplies Inc. (billing@acme.com)
Date: 2025-05-15 | Due: 2025-06-15

Items:
- Standing Desk (x2) @ $349.00 each = $698.00
- Monitor Arm (x2) @ $89.99 each = $179.98
- Cable Management Kit (x1) @ $24.99 = $24.99

Subtotal: $902.97
Tax (8%): $72.24
TOTAL: $975.21

Status: PENDING
"""

result = extractor.invoke(invoice_text)
print(f"Vendor: {result.vendor_name}")
print(f"Invoice: {result.invoice_number} | Date: {result.invoice_date} | Due: {result.due_date}")
print(f"Items: {len(result.line_items)}")
for item in result.line_items:
    print(f"  - {item.description}: {item.quantity}x ${item.unit_price} = ${item.total}")
print(f"Total: ${result.total_amount} {result.currency} | Status: {result.payment_status}")
```

---

## Choosing a schema design approach

| Approach | Best for | Tradeoffs |
|----------|----------|-----------|
| `with_structured_output` + Pydantic | LangChain pipelines, clean code | LangChain dependency |
| Raw tool calling + `json.loads` | Minimal dependencies, full control | More boilerplate |
| JSON mode + manual parsing | Simple schemas | No type guarantees |
| `model_json_schema()` from Pydantic | Reuse Pydantic models as tool schemas | Verbose schema output |

> [!success] Prefer `with_structured_output` for extraction-only tasks
> It eliminates the two-round conversation pattern (no need to send tool results back), handles schema generation from Pydantic automatically, and raises `OutputParserException` cleanly when extraction fails.

---

[[03-anthropic-tool-use]] | [[05-parallel-tool-calls]]
