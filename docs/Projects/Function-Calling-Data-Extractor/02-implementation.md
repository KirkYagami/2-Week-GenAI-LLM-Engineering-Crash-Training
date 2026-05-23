# Implementation — Function-Calling Data Extractor

## Pydantic schemas

```python
# schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class InvoiceData(BaseModel):
    vendor_name: Optional[str] = Field(None, description="Name of the company issuing the invoice")
    invoice_number: Optional[str] = Field(None, description="Invoice ID or number")
    invoice_date: Optional[str] = Field(None, description="Invoice issue date (ISO 8601 preferred)")
    due_date: Optional[str] = Field(None, description="Payment due date")
    total_amount: Optional[float] = Field(None, description="Total invoice amount as a number")
    currency: Optional[str] = Field(None, description="Currency code (USD, EUR, GBP, etc.)")
    line_items: list[dict] = Field(default_factory=list, description="List of {description, quantity, unit_price, total}")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extraction confidence 0–1")

class SupportTicket(BaseModel):
    customer_email: Optional[str] = Field(None, description="Customer's email address")
    subject: Optional[str] = Field(None, description="Ticket subject line")
    issue_category: Optional[str] = Field(None, description="One of: login, billing, performance, feature_request, bug, other")
    urgency: Optional[str] = Field(None, description="One of: low, medium, high, critical")
    browser: Optional[str] = Field(None, description="Browser name and version if mentioned")
    os: Optional[str] = Field(None, description="Operating system if mentioned")
    account_id: Optional[str] = Field(None, description="Customer account identifier if mentioned")
    confidence: float = Field(0.0, ge=0.0, le=1.0)

SCHEMAS = {
    "invoice": InvoiceData,
    "support_ticket": SupportTicket,
}
```

## Extractor with tool_use

```python
# extractor.py
import os
import json
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from schemas import SCHEMAS, InvoiceData, SupportTicket

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_tool(schema_class: type[BaseModel], tool_name: str) -> dict:
    """Build an OpenAI tool definition from a Pydantic model."""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": f"Extract structured {tool_name.replace('_', ' ')} data from the text",
            "parameters": schema_class.model_json_schema(),
        },
    }

async def extract(text: str, schema_name: str) -> dict:
    if schema_name not in SCHEMAS:
        raise ValueError(f"Unknown schema: {schema_name}. Choose from: {list(SCHEMAS.keys())}")

    schema_class = SCHEMAS[schema_name]
    tool_name = f"extract_{schema_name}"
    tool = build_tool(schema_class, tool_name)

    # Step 1: Call LLM with tool
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Extract structured data from the text using the {tool_name} function. "
                    f"Set confidence based on how complete and unambiguous the extraction is: "
                    f"1.0 = all fields clearly present, 0.5 = some fields missing, 0.0 = could not extract."
                ),
            },
            {"role": "user", "content": text[:8000]},  # truncate to avoid context overflow
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        temperature=0.0,
    )

    # Step 2: Parse tool call arguments
    choice = response.choices[0]
    if not choice.message.tool_calls:
        raise ValueError("LLM did not return a tool call")

    tool_call = choice.message.tool_calls[0]
    raw_args = json.loads(tool_call.function.arguments)

    # Step 3: Validate with Pydantic
    try:
        validated = schema_class(**raw_args)
    except ValidationError as e:
        # Return what we got with confidence=0 and flag the errors
        raw_args["confidence"] = 0.0
        raw_args["validation_errors"] = e.errors()
        return raw_args

    return validated.model_dump()
```

## FastAPI application

```python
# app.py
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from extractor import extract

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Data Extractor API", version="1.0.0", lifespan=lifespan)

class TextExtractRequest(BaseModel):
    text: str
    schema_name: str = "invoice"

@app.post("/extract/text")
async def extract_from_text(request: TextExtractRequest):
    if len(request.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text too short for extraction.")
    try:
        result = await extract(request.text, request.schema_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result

@app.post("/extract/file")
async def extract_from_file(
    file: UploadFile = File(...),
    schema_name: str = Form("invoice"),
):
    if file.size and file.size > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 5MB)")

    content = await file.read()

    if file.filename.endswith(".pdf"):
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n\n".join(page.get_text() for page in doc)
    elif file.filename.endswith((".txt", ".md")):
        text = content.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Supported file types: .pdf, .txt, .md")

    if len(text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Document appears empty or image-only.")

    try:
        result = await extract(text, schema_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result

@app.get("/schemas")
async def list_schemas():
    from schemas import SCHEMAS
    return {"available_schemas": list(SCHEMAS.keys())}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

## Test it

```bash
uvicorn app:app --reload

# Extract from text
curl -X POST http://localhost:8000/extract/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Invoice #INV-001 from Acme Corp, total $1,250 USD, due Jan 15 2025", "schema_name": "invoice"}'

# Extract from file
curl -X POST http://localhost:8000/extract/file \
  -F "file=@test_data/invoices/sample.pdf" \
  -F "schema_name=invoice"
```

Expected output:
```json
{
  "vendor_name": "Acme Corp",
  "invoice_number": "INV-001",
  "invoice_date": null,
  "due_date": "2025-01-15",
  "total_amount": 1250.0,
  "currency": "USD",
  "line_items": [],
  "confidence": 0.7
}
```

---

[[01-setup]] | [[03-advanced-features]]
