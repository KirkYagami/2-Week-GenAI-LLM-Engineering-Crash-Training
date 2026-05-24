# Structured Output

Getting reliable structured output from LLMs is one of the most practical production problems. Interviewers ask about it because it separates people who've built real systems from those who've only run demos.

---

## Q1: What are the three ways to get structured JSON from an LLM? When do you use each?

??? "Show answer"
    | Method | How | When to use |
    |--------|-----|-------------|
    | **Prompt-based** | Tell the model to output JSON; parse with `json.loads()` | Quick prototyping, no API support |
    | **JSON mode** | `response_format={"type": "json_object"}` (OpenAI) | Guaranteed valid JSON, no schema enforcement |
    | **Function calling / structured output** | JSON Schema or Pydantic model; model fills it | Production — enforces schema, field types, required fields |

    In production, always prefer function calling or `.with_structured_output()`. Prompt-based JSON fails silently when the model includes prose before or after the JSON block.

    ```python
    from pydantic import BaseModel
    from langchain_openai import ChatOpenAI

    class SupportTicket(BaseModel):
        category: str
        severity: int
        summary: str
        needs_escalation: bool

    llm = ChatOpenAI(model="gpt-4o")
    structured_llm = llm.with_structured_output(SupportTicket)
    result = structured_llm.invoke("Customer says their account was charged twice last month.")
    print(result.category, result.severity)  # already typed
    ```

---

## Q2: How does OpenAI's `strict=True` function calling work, and what are its limits?

??? "Show answer"
    With `strict=True`, OpenAI constrains the model's output at the token level — it cannot produce a token sequence that doesn't match the schema. The result is guaranteed to parse.

    ```python
    import json, os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    tools = [{
        "type": "function",
        "function": {
            "name": "extract_ticket",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["billing", "technical", "account"]},
                    "severity": {"type": "integer"},
                    "summary": {"type": "string"},
                },
                "required": ["category", "severity", "summary"],
                "additionalProperties": False,
            }
        }
    }]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Customer can't log in — password reset emails not arriving."}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_ticket"}},
    )
    result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    ```

    **Limits**: `strict=True` supports only a subset of JSON Schema — no `anyOf`, no recursive schemas, no `$ref`. For complex schemas, use `strict=False` and validate with Pydantic after parsing.

---

## Q3: How do you use `model_json_schema()` to generate tool definitions from Pydantic?

??? "Show answer"
    `model_json_schema()` generates a correct JSON Schema from a Pydantic model automatically, keeping Python types and schema in sync.

    ```python
    from pydantic import BaseModel, Field

    class InvoiceData(BaseModel):
        invoice_number: str = Field(description="Invoice ID from the document header")
        vendor_name: str = Field(description="Name of the vendor or supplier")
        total_amount: float = Field(description="Total amount due including tax")
        due_date: str = Field(description="Payment due date in YYYY-MM-DD format")

    schema = InvoiceData.model_json_schema()
    # schema["properties"] and schema["required"] are ready to pass to OpenAI tools
    ```

    The `description` fields in `Field()` are passed through to the schema and help the model understand what each field means. Treat them as prompt instructions — be specific.

---

## Q4: How do you validate structured output and handle schema violations?

??? "Show answer"
    Defense-in-depth: validate at multiple stages, retry with error context.

    ```python
    import json, os
    from pydantic import BaseModel, ValidationError
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def extract_with_retry(text: str, schema_class: type[BaseModel], max_retries: int = 2) -> BaseModel:
        messages = [{"role": "user", "content": f"Extract data from: {text}"}]

        for attempt in range(max_retries + 1):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content

            try:
                data = json.loads(raw)
                return schema_class(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_retries:
                    raise
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Validation error: {e}. Fix it and return valid JSON."})
    ```

---

## Q5: When should you use nullable fields vs making a field required?

??? "Show answer"
    Make a field **required** when the model can always infer a value from the input — even if it's a default or "unknown".

    Make a field **nullable** when the information might genuinely be absent from the input and you need to distinguish "not present" from any string value.

    ```python
    from pydantic import BaseModel
    from typing import Optional

    class InvoiceData(BaseModel):
        invoice_number: str            # always present in a real invoice
        due_date: Optional[str] = None # might not appear in informal receipts
        po_number: Optional[str] = None
    ```

    Add a `confidence` field when extraction quality varies:

    ```python
    class InvoiceData(BaseModel):
        invoice_number: str
        total_amount: float
        confidence: float  # 0.0–1.0; model signals its own uncertainty
    ```

    Filter out low-confidence extractions in your application logic rather than passing uncertain values downstream.

---

*Previous: [Chain-of-Thought](02-chain-of-thought.md) | Next: [RAG Fundamentals](../02-RAG-and-Vector-Search/01-rag-fundamentals.md)*
