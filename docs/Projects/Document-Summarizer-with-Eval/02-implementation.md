# Implementation — Document Summarizer

## Map-reduce summarization

Documents longer than the context window can't be summarized in one pass. The map-reduce approach: summarize each chunk independently (map), then summarize the chunk summaries together (reduce).

```python
# summarizer.py
import os
import asyncio
from openai import AsyncOpenAI
from tokens import chunk_by_tokens, count_tokens

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_CONTEXT_TOKENS = 3500
OVERLAP_TOKENS = 200

async def summarize_chunk(chunk: str, style: str = "concise") -> str:
    STYLE_INSTRUCTIONS = {
        "concise": "Summarize in 2–3 sentences. Be factual and specific.",
        "bullets": "Extract 3–5 key bullet points. Each bullet: one specific fact or claim.",
        "detailed": "Write a detailed summary preserving all key facts, numbers, and names.",
    }
    instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["concise"])
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": chunk},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    return response.choices[0].message.content

async def map_summaries(chunks: list[str], style: str = "concise", max_concurrent: int = 5) -> list[str]:
    """Summarize all chunks concurrently, limited to max_concurrent at a time."""
    semaphore = asyncio.Semaphore(max_concurrent)
    async def summarize_with_limit(chunk: str) -> str:
        async with semaphore:
            return await summarize_chunk(chunk, style)
    return await asyncio.gather(*[summarize_with_limit(c) for c in chunks])

async def reduce_summaries(chunk_summaries: list[str], style: str = "concise", format_type: str = "paragraph") -> str:
    """Combine chunk summaries into a final summary."""
    combined = "\n\n---\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )

    FORMAT_INSTRUCTIONS = {
        "paragraph": "Write a coherent paragraph summary integrating all sections.",
        "bullets": "Write 5–8 bullet points covering the most important points across all sections.",
        "executive": "Write an executive summary: 1-sentence TL;DR, then 3–5 key takeaways in bullet form.",
    }
    format_instruction = FORMAT_INSTRUCTIONS.get(format_type, FORMAT_INSTRUCTIONS["paragraph"])

    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Synthesize these section summaries into a final document summary. {format_instruction}"},
            {"role": "user", "content": combined},
        ],
        temperature=0.0,
        max_tokens=600,
    )
    return response.choices[0].message.content

async def summarize_document(text: str, format_type: str = "paragraph", style: str = "concise") -> dict:
    """Full map-reduce pipeline."""
    total_tokens = count_tokens(text)

    if total_tokens <= MAX_CONTEXT_TOKENS:
        # Short document: single-pass
        summary = await summarize_chunk(text, style)
        return {"summary": summary, "chunks": 1, "strategy": "single-pass", "input_tokens": total_tokens}

    # Long document: map-reduce
    chunks = chunk_by_tokens(text, max_tokens=MAX_CONTEXT_TOKENS, overlap_tokens=OVERLAP_TOKENS)
    chunk_summaries = await map_summaries(chunks, style)
    final_summary = await reduce_summaries(chunk_summaries, style, format_type)

    return {
        "summary": final_summary,
        "chunk_summaries": chunk_summaries,
        "chunks": len(chunks),
        "strategy": "map-reduce",
        "input_tokens": total_tokens,
    }
```

## FastAPI application

```python
# app.py
import os
import io
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from summarizer import summarize_document

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Document Summarizer", lifespan=lifespan)

FORMAT_OPTIONS = {"paragraph", "bullets", "executive"}

def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n\n".join(page.get_text() for page in doc)
    elif filename.endswith((".md", ".txt")):
        return file_bytes.decode("utf-8")
    else:
        raise ValueError(f"Unsupported file type: {filename}")

@app.post("/summarize")
async def summarize(
    file: UploadFile = File(...),
    format_type: str = Form("paragraph"),
    style: str = Form("concise"),
):
    if format_type not in FORMAT_OPTIONS:
        raise HTTPException(status_code=400, detail=f"format_type must be one of: {FORMAT_OPTIONS}")

    if file.size and file.size > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 5MB)")

    content = await file.read()
    try:
        text = extract_text(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract text: {e}")

    if len(text.strip()) < 100:
        raise HTTPException(status_code=422, detail="Document is too short or appears empty.")

    result = await summarize_document(text, format_type=format_type, style=style)
    return result

@app.post("/summarize/text")
async def summarize_text(text: str, format_type: str = "paragraph", style: str = "concise"):
    """Summarize raw text directly (for testing)."""
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Text too short.")
    result = await summarize_document(text, format_type=format_type, style=style)
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}
```

## Test

```bash
uvicorn app:app --reload

# Test with a file
curl -X POST http://localhost:8000/summarize \
  -F "file=@test_docs/sample_article.md" \
  -F "format_type=executive" \
  -F "style=concise"

# Test with raw text
curl -X POST "http://localhost:8000/summarize/text?format_type=bullets&style=concise" \
  -H "Content-Type: application/json" \
  -d '"Your long document text here..."'
```

---

[[01-setup]] | [[03-advanced-features]]
