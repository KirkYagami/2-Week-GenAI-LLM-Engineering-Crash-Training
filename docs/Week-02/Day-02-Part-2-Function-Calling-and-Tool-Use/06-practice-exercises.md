# Practice Exercises — Function Calling and Tool Use

---

## Exercise 1 — Contact extractor (Warm-up)

Build an extractor that pulls contact information from unstructured text and validates the output with Pydantic.

```python
import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ContactInfo(BaseModel):
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        if v and "@" not in v:
            return None  # Reject invalid emails silently
        return v

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        digits = re.sub(r"\D", "", v)
        return digits if len(digits) >= 7 else None

TOOL = {
    "type": "function",
    "function": {
        "name": "extract_contact",
        "description": "Extract contact information from text.",
        "parameters": ContactInfo.model_json_schema()
    }
}

def extract_contact(text: str) -> ContactInfo | None:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract contact information. Return null for fields not present."},
                {"role": "user", "content": text}
            ],
            tools=[TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_contact"}},
            temperature=0.0,
        )
        raw = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        return ContactInfo(**raw)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None

TEST_CASES = [
    "Hi, I'm Dr. Maria Santos from BioTech Labs. You can reach me at m.santos@biotech.io or +1-617-555-0142.",
    "Contact SALES: sales@company.com for pricing.",
    "James O'Brien, VP Engineering - james.obrien@startup.com - (212) 555-0100",
    "No contact information in this sentence about machine learning.",
]

print(f"{'Input':<55} {'Name':<20} {'Email':<30} {'Phone'}")
print("-" * 115)
for text in TEST_CASES:
    result = extract_contact(text)
    if result:
        print(f"{text[:55]:<55} {(result.full_name or 'N/A')[:20]:<20} {(result.email or 'N/A')[:30]:<30} {result.phone or 'N/A'}")
    else:
        print(f"{text[:55]:<55} EXTRACTION FAILED")
```

---

## Exercise 2 — Multi-tool research assistant (Main)

Build a research assistant that uses multiple tools to answer complex questions about companies.

```python
import os
import json
from openai import OpenAI
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simulated tool implementations
def get_company_info(company_name: str) -> dict:
    companies = {
        "OpenAI": {"founded": 2015, "employees": 1700, "valuation_usd_bn": 157, "hq": "San Francisco"},
        "Anthropic": {"founded": 2021, "employees": 800, "valuation_usd_bn": 18, "hq": "San Francisco"},
        "DeepMind": {"founded": 2010, "employees": 2500, "valuation_usd_bn": 30, "hq": "London"},
    }
    return companies.get(company_name, {"error": f"Company {company_name} not found"})

def get_recent_news(company_name: str, max_results: int = 3) -> list[dict]:
    news = {
        "OpenAI": [
            {"title": "OpenAI releases o3 model", "date": "2025-04-15"},
            {"title": "ChatGPT reaches 200M users", "date": "2025-03-10"},
        ],
        "Anthropic": [
            {"title": "Claude 4 outperforms GPT-4o on coding benchmarks", "date": "2025-05-01"},
            {"title": "Anthropic raises $4B Series D", "date": "2025-02-20"},
        ],
    }
    return news.get(company_name, [{"title": "No recent news found", "date": "N/A"}])[:max_results]

def compare_companies(company_a: str, company_b: str, metric: str) -> dict:
    info_a = get_company_info(company_a)
    info_b = get_company_info(company_b)
    if "error" in info_a or "error" in info_b:
        return {"error": "One or both companies not found"}
    val_a = info_a.get(metric)
    val_b = info_b.get(metric)
    return {
        "metric": metric,
        company_a: val_a,
        company_b: val_b,
        "winner": company_a if (val_a or 0) > (val_b or 0) else company_b
    }

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Get basic information about an AI company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": "Company name, e.g. 'OpenAI', 'Anthropic'"}
                },
                "required": ["company_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_news",
            "description": "Get recent news headlines about a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string"},
                    "max_results": {"type": "integer", "default": 3}
                },
                "required": ["company_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_companies",
            "description": "Compare two companies on a specific metric.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_a": {"type": "string"},
                    "company_b": {"type": "string"},
                    "metric": {
                        "type": "string",
                        "enum": ["employees", "valuation_usd_bn", "founded"],
                        "description": "Metric to compare"
                    }
                },
                "required": ["company_a", "company_b", "metric"]
            }
        }
    }
]

FUNCTION_MAP = {
    "get_company_info": get_company_info,
    "get_recent_news": get_recent_news,
    "compare_companies": compare_companies,
}

def research_assistant(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a research assistant with access to company data and news. Use tools to gather information before answering."},
        {"role": "user", "content": question}
    ]

    rounds = 0
    while rounds < 5:
        rounds += 1
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS,
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = FUNCTION_MAP[tc.function.name](**args)
                print(f"  [Tool] {tc.function.name}({args}) → {str(result)[:80]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return "Max rounds reached"

questions = [
    "How does OpenAI compare to Anthropic in terms of employees and valuation?",
    "What are the latest news about Anthropic?",
]
for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {research_assistant(q)}")
```

---

## Exercise 3 — Async parallel extractor (Stretch)

Extract data from multiple documents simultaneously using async tool calls.

```python
import os
import json
import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Optional

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DocumentSummary(BaseModel):
    document_type: str = Field(description="invoice, contract, email, report, or other")
    key_dates: list[str] = Field(default_factory=list, description="Important dates in YYYY-MM-DD format")
    monetary_amounts: list[float] = Field(default_factory=list, description="Dollar amounts mentioned")
    action_items: list[str] = Field(default_factory=list, description="Required actions or next steps")
    urgency: str = Field(description="low, medium, high, or critical")

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "summarize_document",
        "description": "Extract key information from a document.",
        "parameters": DocumentSummary.model_json_schema()
    }
}

async def extract_single(doc_id: str, text: str) -> tuple[str, DocumentSummary | None]:
    try:
        response = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Extract key info from:\n\n{text}"}],
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "function", "function": {"name": "summarize_document"}},
            temperature=0.0,
        )
        raw = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return doc_id, DocumentSummary(**raw)
    except Exception as e:
        print(f"  Failed {doc_id}: {e}")
        return doc_id, None

DOCUMENTS = {
    "invoice_001": "Invoice #4521 from Tech Supplies Co. dated 2025-05-15. Payment of $4,250.00 due by 2025-06-15. URGENT: Late fees apply after due date.",
    "contract_001": "Service Agreement expires 2025-12-31. Monthly fee: $2,500. Renewal must be initiated by 2025-11-30 or contract auto-terminates.",
    "email_001": "Hi team, please review the Q2 report by Friday 2025-05-30. No financial impact but exec team needs it for the board meeting.",
}

async def extract_all(documents: dict[str, str]) -> dict[str, DocumentSummary | None]:
    tasks = [extract_single(doc_id, text) for doc_id, text in documents.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)

results = asyncio.run(extract_all(DOCUMENTS))

print(f"\n{'Document':<15} {'Type':<12} {'Urgency':<10} {'Amounts':<20} {'Actions'}")
print("-" * 80)
for doc_id, result in results.items():
    if result:
        amounts = ", ".join(f"${a}" for a in result.monetary_amounts[:2])
        actions = result.action_items[0][:30] if result.action_items else "None"
        print(f"{doc_id:<15} {result.document_type:<12} {result.urgency:<10} {amounts:<20} {actions}")
```

---

[[05-parallel-tool-calls]] | [[07-interview-questions]]
