# Implementation — AI Writing Assistant

## Prompt templates

```python
# prompts.py
from langchain_core.prompts import ChatPromptTemplate

OUTLINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a professional content strategist. Create a structured outline."),
    ("user", "Create a {section_count}-section outline for: {topic}\nTarget audience: {audience}\nStyle: {style}"),
])

DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a skilled writer. Write clear, engaging content following the outline exactly."),
    ("user", "Write a complete draft based on this outline:\n{outline}\n\nStyle: {style}\nTarget length: {target_words} words"),
])

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an editor. Improve the draft: fix flow, remove redundancy, strengthen transitions."),
    ("user", "Refine this draft:\n{draft}\n\nFocus on: clarity, conciseness, flow. Keep the same structure."),
])

STYLE_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a style editor. Enforce the requested tone and style consistently throughout the text."),
    ("user", "Rewrite to match this style: {style}\n\nText:\n{text}"),
])
```

## LangChain pipeline

```python
# pipeline.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompts import OUTLINE_PROMPT, DRAFT_PROMPT, REFINE_PROMPT, STYLE_CHECK_PROMPT

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
)

parser = StrOutputParser()

# Individual step chains
outline_chain = OUTLINE_PROMPT | llm | parser
draft_chain = DRAFT_PROMPT | llm | parser
refine_chain = REFINE_PROMPT | llm | parser
style_chain = STYLE_CHECK_PROMPT | llm | parser

async def run_pipeline(
    topic: str,
    style: str = "professional",
    audience: str = "general readers",
    target_words: int = 400,
    section_count: int = 3,
) -> dict:
    """Run the full 4-step writing pipeline and return all intermediate outputs."""

    # Step 1: Outline
    outline = await outline_chain.ainvoke({
        "topic": topic,
        "style": style,
        "audience": audience,
        "section_count": section_count,
    })

    # Step 2: Draft
    draft = await draft_chain.ainvoke({
        "outline": outline,
        "style": style,
        "target_words": target_words,
    })

    # Step 3: Refine
    refined = await refine_chain.ainvoke({"draft": draft})

    # Step 4: Style enforcement
    final = await style_chain.ainvoke({"text": refined, "style": style})

    return {
        "outline": outline,
        "draft": draft,
        "refined": refined,
        "final": final,
        "word_count": len(final.split()),
    }
```

## FastAPI application

```python
# app.py
import os
import json
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompts import OUTLINE_PROMPT, DRAFT_PROMPT, REFINE_PROMPT, STYLE_CHECK_PROMPT

load_dotenv()

STYLE_OPTIONS = {"professional", "casual", "technical", "persuasive", "academic"}

class WriteRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500)
    style: str = Field("professional", description=f"One of: {', '.join(STYLE_OPTIONS)}")
    audience: str = Field("general readers", max_length=200)
    target_words: int = Field(400, ge=100, le=2000)
    stream_final: bool = True

class WriteResponse(BaseModel):
    outline: str
    final: str
    word_count: int
    latency_ms: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = ChatOpenAI(
        model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    yield

app = FastAPI(title="AI Writing Assistant", lifespan=lifespan)

@app.post("/write", response_model=WriteResponse)
async def write(request: WriteRequest):
    if request.style not in STYLE_OPTIONS:
        raise HTTPException(status_code=400, detail=f"style must be one of: {STYLE_OPTIONS}")

    llm = app.state.llm
    parser = StrOutputParser()

    start = time.perf_counter()

    outline = await (OUTLINE_PROMPT | llm | parser).ainvoke({
        "topic": request.topic,
        "style": request.style,
        "audience": request.audience,
        "section_count": 3,
    })
    draft = await (DRAFT_PROMPT | llm | parser).ainvoke({
        "outline": outline,
        "style": request.style,
        "target_words": request.target_words,
    })
    refined = await (REFINE_PROMPT | llm | parser).ainvoke({"draft": draft})
    final = await (STYLE_CHECK_PROMPT | llm | parser).ainvoke({
        "text": refined, "style": request.style,
    })

    return WriteResponse(
        outline=outline,
        final=final,
        word_count=len(final.split()),
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )

@app.post("/write/stream")
async def write_stream(request: WriteRequest):
    if request.style not in STYLE_OPTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid style")

    llm = app.state.llm
    parser = StrOutputParser()

    async def generate():
        # Stream step names as events
        yield f"data: {json.dumps({'event': 'step', 'step': 'outline'})}\n\n"
        outline = await (OUTLINE_PROMPT | llm | parser).ainvoke({
            "topic": request.topic, "style": request.style,
            "audience": request.audience, "section_count": 3,
        })
        yield f"data: {json.dumps({'event': 'outline', 'content': outline})}\n\n"

        yield f"data: {json.dumps({'event': 'step', 'step': 'draft'})}\n\n"
        draft = await (DRAFT_PROMPT | llm | parser).ainvoke({
            "outline": outline, "style": request.style, "target_words": request.target_words,
        })

        yield f"data: {json.dumps({'event': 'step', 'step': 'refine'})}\n\n"
        refined = await (REFINE_PROMPT | llm | parser).ainvoke({"draft": draft})

        yield f"data: {json.dumps({'event': 'step', 'step': 'style'})}\n\n"
        # Stream the final step token-by-token
        async for chunk in (STYLE_CHECK_PROMPT | llm | parser).astream({
            "text": refined, "style": request.style,
        }):
            yield f"data: {json.dumps({'event': 'token', 'token': chunk})}\n\n"

        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/health")
async def health():
    return {"status": "ok"}
```

## Test it

```bash
uvicorn app:app --reload

# Non-streaming
curl -X POST http://localhost:8000/write \
  -H "Content-Type: application/json" \
  -d '{"topic": "Why Python is great for AI development", "style": "technical", "target_words": 300}'

# Streaming
curl -X POST http://localhost:8000/write/stream \
  -H "Content-Type: application/json" \
  -d '{"topic": "Remote work productivity tips", "style": "casual"}'
```

---

[[01-setup]] | [[03-advanced-features]]
