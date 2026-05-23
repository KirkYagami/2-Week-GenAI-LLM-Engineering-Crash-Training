# Implementation — LangGraph Research Agent

## State definition

```python
# state.py
import operator
from typing import Annotated
from typing_extensions import TypedDict

class ResearchState(TypedDict):
    question: str                                    # Original research question
    sub_questions: list[str]                         # Planner output
    findings: Annotated[list[str], operator.add]    # Accumulates across nodes
    draft: str                                       # Writer output
    quality_score: float                             # Critic score (0–1)
    feedback: str                                    # Critic feedback
    attempts: int                                    # Loop count (max 3)
    final_report: str                                # Final approved report
```

The `Annotated[list[str], operator.add]` reducer means any node that returns `{"findings": [...]}` appends to the existing list instead of overwriting it.

## Node functions

```python
# nodes.py
import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import ResearchState

_llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
_llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=os.getenv("OPENAI_API_KEY"))

async def planner(state: ResearchState) -> dict:
    """Decompose research question into sub-questions."""
    response = await _llm_fast.ainvoke([
        SystemMessage(content=(
            "You are a research strategist. Given a research question, produce exactly 4 specific sub-questions "
            "that together would answer the main question. Return them as a numbered list, one per line."
        )),
        HumanMessage(content=f"Research question: {state['question']}"),
    ])
    lines = response.content.strip().splitlines()
    sub_questions = [l.lstrip("1234567890. ").strip() for l in lines if l.strip()][:4]
    return {"sub_questions": sub_questions, "attempts": 0, "findings": []}

async def researcher(state: ResearchState) -> dict:
    """Answer all sub-questions in parallel."""
    async def research_one(sub_q: str) -> str:
        resp = await _llm_fast.ainvoke([
            SystemMessage(content=(
                "You are a research expert. Provide a detailed, factual answer to the question. "
                "Include specific examples, numbers, and citations where possible. "
                "Write 2–3 comprehensive paragraphs."
            )),
            HumanMessage(content=sub_q),
        ])
        return f"### {sub_q}\n\n{resp.content}"

    findings = await asyncio.gather(*[research_one(q) for q in state["sub_questions"]])
    return {"findings": list(findings)}

async def writer(state: ResearchState) -> dict:
    """Synthesize findings into a structured report."""
    findings_text = "\n\n---\n\n".join(state["findings"])
    feedback_prompt = f"\n\nPrevious draft was rejected. Critic feedback: {state['feedback']}" if state.get("feedback") else ""

    response = await _llm_creative.ainvoke([
        SystemMessage(content=(
            "You are a research writer. Synthesize the findings into a professional report with:\n"
            "1. Executive Summary (2-3 sentences)\n"
            "2. Key Findings (organized by theme, not by sub-question)\n"
            "3. Analysis (implications and connections between findings)\n"
            "4. Conclusion\n\n"
            "Write in third person, active voice. Use markdown formatting."
        )),
        HumanMessage(content=(
            f"Research question: {state['question']}\n\n"
            f"Research findings:\n{findings_text}"
            f"{feedback_prompt}"
        )),
    ])
    return {"draft": response.content, "attempts": state.get("attempts", 0) + 1}

async def critic(state: ResearchState) -> dict:
    """Evaluate draft quality and provide improvement feedback."""
    response = await _llm_fast.ainvoke([
        SystemMessage(content=(
            "You are a research editor. Evaluate this report on 4 criteria:\n"
            "- Completeness: covers all aspects of the question\n"
            "- Accuracy: claims are specific and verifiable\n"
            "- Coherence: well-structured and logically flows\n"
            "- Insight: goes beyond summarizing, adds analytical value\n\n"
            "Respond with exactly:\n"
            "SCORE: <float 0.0-1.0 average across criteria>\n"
            "FEEDBACK: <one specific sentence on the main weakness>"
        )),
        HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"Report to evaluate:\n{state['draft']}"
        )),
    ])
    lines = {}
    for line in response.content.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            lines[k.strip()] = v.strip()

    try:
        score = float(lines.get("SCORE", "0.5"))
    except ValueError:
        score = 0.5

    return {"quality_score": score, "feedback": lines.get("FEEDBACK", "")}
```

## Graph assembly

```python
# agent.py
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import ResearchState
from nodes import planner, researcher, writer, critic

load_dotenv()

def route_after_critic(state: ResearchState) -> str:
    """Loop back to writer if quality is low; end if acceptable or max attempts reached."""
    if state["quality_score"] >= 0.80 or state["attempts"] >= 3:
        return "finalize"
    return "writer"

async def finalize(state: ResearchState) -> dict:
    return {"final_report": state["draft"]}

graph = StateGraph(ResearchState)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("critic", critic)
graph.add_node("finalize", finalize)

graph.set_entry_point("planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "critic")
graph.add_conditional_edges(
    "critic",
    route_after_critic,
    {"finalize": "finalize", "writer": "writer"},
)
graph.add_edge("finalize", END)

research_app = graph.compile()

async def run(question: str) -> dict:
    result = await research_app.ainvoke({
        "question": question,
        "sub_questions": [],
        "findings": [],
        "draft": "",
        "quality_score": 0.0,
        "feedback": "",
        "attempts": 0,
        "final_report": "",
    })
    return {
        "report": result["final_report"],
        "quality_score": result["quality_score"],
        "attempts": result["attempts"],
    }
```

## FastAPI endpoint

```python
# app.py
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import run

app = FastAPI(title="Research Agent", version="1.0.0")

class ResearchRequest(BaseModel):
    question: str

@app.post("/research")
async def research(request: ResearchRequest):
    if len(request.question.strip()) < 10:
        raise HTTPException(status_code=400, detail="Question too short.")
    result = await run(request.question)
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}
```

```bash
uvicorn app:app --reload

curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main approaches to reducing hallucination in LLMs?"}'
```

---

[[01-setup]] | [[03-advanced-features]]
