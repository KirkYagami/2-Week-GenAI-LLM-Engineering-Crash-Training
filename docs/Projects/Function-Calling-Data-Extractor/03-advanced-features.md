# Advanced Features — Function-Calling Data Extractor

## Retry on low confidence

When the extraction returns a low confidence score, retry with a more detailed prompt:

```python
async def extract_with_retry(text: str, schema_name: str, min_confidence: float = 0.6) -> dict:
    result = await extract(text, schema_name)
    if result.get("confidence", 0) < min_confidence:
        # Retry with a more explicit prompt
        from extractor import aclient, build_tool, SCHEMAS
        schema_class = SCHEMAS[schema_name]
        tool = build_tool(schema_class, f"extract_{schema_name}")
        response = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract EVERY piece of structured information you can find. "
                        "Look carefully for amounts, dates, names, IDs, and categories. "
                        "If a value is ambiguous, make your best inference and set confidence accordingly."
                    ),
                },
                {"role": "user", "content": text[:8000]},
            ],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": f"extract_{schema_name}"}},
            temperature=0.0,
        )
        import json
        from pydantic import ValidationError
        raw = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        try:
            validated = schema_class(**raw)
            return validated.model_dump()
        except ValidationError:
            return raw
    return result
```

## Adding a new schema

To add a job description extractor, add to `schemas.py`:

```python
class JobDescription(BaseModel):
    job_title: Optional[str] = Field(None, description="Exact job title")
    company_name: Optional[str] = Field(None, description="Hiring company name")
    location: Optional[str] = Field(None, description="Job location (city, state, remote)")
    employment_type: Optional[str] = Field(None, description="full-time, part-time, contract, freelance")
    salary_min: Optional[float] = Field(None, description="Minimum salary if range given")
    salary_max: Optional[float] = Field(None, description="Maximum salary if range given")
    required_skills: list[str] = Field(default_factory=list, description="Required technical skills")
    years_experience: Optional[int] = Field(None, description="Minimum years of experience required")
    confidence: float = Field(0.0, ge=0.0, le=1.0)

# Register it
SCHEMAS["job_description"] = JobDescription
```

No other changes needed — the tool is built dynamically from the Pydantic schema.

## Batch extraction

For processing many documents, use the OpenAI Batch API or async parallel extraction:

```python
import asyncio

async def batch_extract(texts: list[str], schema_name: str, max_concurrent: int = 5) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    async def extract_one(text: str) -> dict:
        async with semaphore:
            try:
                return await extract(text, schema_name)
            except Exception as e:
                return {"error": str(e), "confidence": 0.0}
    return await asyncio.gather(*[extract_one(t) for t in texts])
```

## Field normalization

Standardize inconsistent date and currency formats post-extraction:

```python
from datetime import datetime
import re

def normalize_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y"]:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str  # Return as-is if we can't parse

def normalize_amount(amount) -> float | None:
    if amount is None:
        return None
    if isinstance(amount, (int, float)):
        return float(amount)
    # Handle strings like "$1,250.00" or "1.250,00"
    cleaned = re.sub(r"[^\d.]", "", str(amount).replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None
```

---

[[02-implementation]] | [[04-evaluation]]
