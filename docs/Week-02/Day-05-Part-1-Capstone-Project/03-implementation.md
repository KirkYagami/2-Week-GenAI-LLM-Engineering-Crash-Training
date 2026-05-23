# Implementation

This note gives you a production-quality starter for each capstone option. Pick your option, copy the scaffold, and extend it. The skeleton handles the hard parts — API client lifecycle, error handling, async patterns — so you can focus on the domain logic.

## Learning objectives

- Implement a complete LLM application from scratch in under two hours
- Wire together RAG, agents, or extraction pipelines with FastAPI
- Handle errors, retries, and edge cases without breaking the demo

---

## Option A — RAG Customer Support Bot

```python
# app.py
import os
import time
import hashlib
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ---- Lifecycle ----

aclient: AsyncOpenAI = None
collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global aclient, collection
    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    collection = chroma.get_or_create_collection("docs", embedding_function=embedding_fn)
    print(f"Loaded collection with {collection.count()} documents")
    yield
    await aclient.close()

app = FastAPI(title="RAG Support Bot", lifespan=lifespan)

# ---- Cache ----

_cache: dict[str, dict] = {}
CACHE_TTL = 600

def cache_key(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()

# ---- Models ----

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    n_results: int = Field(5, ge=1, le=10)
    use_cache: bool = True

# ---- Endpoints ----

@app.post("/chat")
async def chat(request: ChatRequest):
    key = cache_key(request.question)
    if request.use_cache:
        entry = _cache.get(key)
        if entry and time.time() - entry["ts"] < CACHE_TTL:
            return {**entry["data"], "cached": True}

    results = collection.query(query_texts=[request.question], n_results=request.n_results)
    chunks = results["documents"][0] if results["documents"] else []
    ids = results["ids"][0] if results["ids"] else []

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    context = "\n\n---\n\n".join(
        f"[Source {i+1}: {doc_id}]\n{chunk}"
        for i, (doc_id, chunk) in enumerate(zip(ids, chunks))
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful support assistant. Answer questions using only the provided context. "
                "If the context doesn't contain the answer, say so. "
                "Always cite the source numbers you used."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {request.question}",
        },
    ]

    start = time.perf_counter()
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_tokens=600,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    data = {
        "answer": response.choices[0].message.content,
        "sources": ids,
        "tokens": response.usage.total_tokens,
        "latency_ms": round(latency_ms, 1),
    }
    _cache[key] = {"data": data, "ts": time.time()}
    return {**data, "cached": False}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    results = collection.query(query_texts=[request.question], n_results=request.n_results)
    chunks = results["documents"][0] if results["documents"] else []
    ids = results["ids"][0] if results["ids"] else []

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    context = "\n\n---\n\n".join(f"[Source {i+1}]\n{c}" for i, c in enumerate(chunks))
    messages = [
        {"role": "system", "content": "Answer using only the provided context. Cite source numbers."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"},
    ]

    async def token_stream():
        yield f"data: {json.dumps({'event': 'sources', 'ids': ids})}\n\n"
        async for chunk in await aclient.chat.completions.create(
            model="gpt-4o-mini", messages=messages, stream=True, max_tokens=600, temperature=0.0,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'event': 'token', 'token': delta})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/health")
async def health():
    return {"status": "ok", "doc_count": collection.count(), "cache_entries": len(_cache)}
```

```python
# ingest.py — run once to populate ChromaDB
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

def ingest_directory(docs_path: str = "./docs") -> None:
    import pathlib
    chroma = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    collection = chroma.get_or_create_collection("docs", embedding_function=embedding_fn)

    documents, ids = [], []
    for path in pathlib.Path(docs_path).rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        # Simple fixed-size chunking; replace with semantic chunking for production
        chunks = [text[i:i+800] for i in range(0, len(text), 600)]
        for j, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{path.stem}_{j}")

    collection.add(documents=documents, ids=ids)
    print(f"Ingested {len(documents)} chunks from {docs_path}")

if __name__ == "__main__":
    ingest_directory()
```

---

## Option B — LangGraph Research Agent

```python
# research_agent.py
import os
import asyncio
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ---- State ----

class ResearchState(TypedDict):
    question: str
    sub_questions: list[str]
    findings: Annotated[list[str], operator.add]
    draft: str
    quality_score: float
    feedback: str
    attempts: int

# ---- LLM clients ----

planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))
critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

# ---- Nodes ----

async def planner(state: ResearchState) -> dict:
    response = await planner_llm.ainvoke([
        SystemMessage(content="Break the research question into 3–5 specific sub-questions. Return as a numbered list."),
        HumanMessage(content=state["question"]),
    ])
    lines = [l.strip("1234567890. ") for l in response.content.strip().splitlines() if l.strip()]
    return {"sub_questions": lines[:5], "attempts": 0}

async def researcher(state: ResearchState) -> dict:
    async def research_one(sub_q: str) -> str:
        r = await planner_llm.ainvoke([
            SystemMessage(content="Provide a detailed factual answer to this research sub-question in 2–3 paragraphs."),
            HumanMessage(content=sub_q),
        ])
        return f"### {sub_q}\n{r.content}"

    findings = await asyncio.gather(*[research_one(q) for q in state["sub_questions"]])
    return {"findings": list(findings)}

async def writer(state: ResearchState) -> dict:
    findings_text = "\n\n".join(state["findings"])
    prompt = (
        f"Research question: {state['question']}\n\n"
        f"Findings:\n{findings_text}\n\n"
        f"{'Previous feedback: ' + state['feedback'] if state.get('feedback') else ''}\n\n"
        f"Write a comprehensive, well-structured report with an executive summary, key findings, and conclusion."
    )
    response = await writer_llm.ainvoke([HumanMessage(content=prompt)])
    return {"draft": response.content, "attempts": state.get("attempts", 0) + 1}

async def critic(state: ResearchState) -> dict:
    prompt = (
        f"Evaluate this research report on a scale of 0.0–1.0 for: accuracy, completeness, clarity, structure.\n\n"
        f"Report:\n{state['draft']}\n\n"
        f"Respond with exactly two lines:\nSCORE: <float>\nFEEDBACK: <one sentence>"
    )
    response = await critic_llm.ainvoke([HumanMessage(content=prompt)])
    lines = {l.split(":")[0].strip(): ":".join(l.split(":")[1:]).strip()
             for l in response.content.strip().splitlines() if ":" in l}
    try:
        score = float(lines.get("SCORE", "0.5"))
    except ValueError:
        score = 0.5
    return {"quality_score": score, "feedback": lines.get("FEEDBACK", "")}

def route_after_critic(state: ResearchState) -> str:
    if state["quality_score"] >= 0.80 or state["attempts"] >= 3:
        return "end"
    return "writer"

# ---- Graph ----

graph = StateGraph(ResearchState)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("critic", critic)

graph.set_entry_point("planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "critic")
graph.add_conditional_edges("critic", route_after_critic, {"end": END, "writer": "writer"})

research_app = graph.compile()

async def run_research(question: str) -> dict:
    result = await research_app.ainvoke({"question": question, "findings": [], "attempts": 0})
    return {
        "question": result["question"],
        "report": result["draft"],
        "quality_score": result["quality_score"],
        "attempts": result["attempts"],
    }

# Test:
# import asyncio
# result = asyncio.run(run_research("What are the main approaches to LLM alignment?"))
# print(result["report"])
```

---

## Option C — Document Intelligence Pipeline

```python
# doc_extractor.py
import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from openai import AsyncOpenAI
import io

try:
    import pymupdf as fitz
except ImportError:
    import fitz

app = FastAPI(title="Document Intelligence API")
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class InvoiceData(BaseModel):
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = Field(None, description="ISO 4217 currency code, e.g. USD")
    line_items: list[dict] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extraction confidence 0–1")

EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_invoice",
        "description": "Extract structured data from an invoice document",
        "parameters": InvoiceData.model_json_schema(),
    },
}

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n\n".join(pages)

@app.post("/extract/invoice", response_model=InvoiceData)
async def extract_invoice(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB).")

    try:
        text = extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to extract text: {e}")

    if len(text.strip()) < 50:
        raise HTTPException(status_code=422, detail="Document appears to be empty or image-only.")

    # Truncate to fit context window
    text = text[:12000]

    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract invoice data from the document text. Set confidence based on how complete and clear the extraction is.",
            },
            {"role": "user", "content": f"Document text:\n\n{text}"},
        ],
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "function", "function": {"name": "extract_invoice"}},
        temperature=0.0,
    )

    tool_call = response.choices[0].message.tool_calls[0]
    extracted = json.loads(tool_call.function.arguments)

    try:
        return InvoiceData(**extracted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

> [!tip] Build the happy path first, then add error handling
> Get one end-to-end request working with a known-good input before handling edge cases. Error handling for malformed PDFs, empty responses, and API failures is important but don't let it block your first working demo.

> [!warning] Never hardcode API keys — use environment variables
> All examples load credentials with `os.getenv("OPENAI_API_KEY")`. In production, use a secrets manager. In local development, use a `.env` file with `python-dotenv`. Never commit a key to git.

---

[[02-architecture-design]] | [[04-evaluation-and-testing]]
