# Multi-Agent Orchestration

A single agent with many tools becomes harder to maintain as the tool list grows. Multi-agent systems split the work: a planner routes tasks to specialist agents, each of which has a focused tool set and system prompt. LangGraph makes this pattern explicit through subgraphs and handoffs.

## Learning objectives

- Build a supervisor pattern: orchestrator routes to specialist agents
- Implement subgraphs for encapsulating specialist logic
- Pass context between agents using shared state
- Add a critic agent that reviews and triggers rewrites
- Know when multi-agent adds value vs single-agent

---

## The supervisor pattern

```
User question
      ↓
 Supervisor (planner)
   ├── Research agent (web search, summarize)
   ├── Data agent (calculations, statistics)
   └── Writer agent (synthesize, format)
         ↓
   Final answer
```

The supervisor classifies the task and routes to the right specialist. Each specialist completes its job and returns control to the supervisor (or directly to the next node).

---

## Supervisor implementation

```python
import os
import json
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

class OrchestratorState(TypedDict):
    user_request: str
    research_output: str
    analysis_output: str
    final_report: str
    next_step: str
    steps_taken: Annotated[list[str], operator.add]

# Simulated tools for each agent
def web_search(query: str) -> str:
    data = {
        "ai market": "The AI market is projected to reach $1.8T by 2030 (McKinsey).",
        "llm adoption": "80% of enterprises are piloting LLMs as of 2024 (Gartner).",
        "top models": "Leading models: GPT-4o (OpenAI), Claude Sonnet (Anthropic), Gemini (Google).",
    }
    for key, val in data.items():
        if key in query.lower():
            return val
    return f"Research data for: {query}"

def analyze_data(data: str, question: str) -> str:
    resp = llm.invoke([
        SystemMessage(content="Analyze this data and extract key insights."),
        HumanMessage(content=f"Data: {data}\nAnalysis question: {question}")
    ])
    return resp.content

# Agent nodes
def supervisor(state: OrchestratorState) -> dict:
    """Determine which specialist to call next."""
    research_done = bool(state.get("research_output"))
    analysis_done = bool(state.get("analysis_output"))

    if not research_done:
        next_step = "research"
    elif not analysis_done:
        next_step = "analyze"
    else:
        next_step = "write"

    return {"next_step": next_step, "steps_taken": [f"supervisor→{next_step}"]}

def research_agent(state: OrchestratorState) -> dict:
    """Search and gather information."""
    # Determine what to search for
    resp = llm.invoke([
        SystemMessage(content="What should I search for to answer this question? Give 2 search queries, one per line."),
        HumanMessage(content=state["user_request"])
    ])
    queries = resp.content.strip().split("\n")[:2]
    results = [web_search(q) for q in queries]
    return {
        "research_output": "\n".join(results),
        "steps_taken": ["research_agent completed"],
    }

def analysis_agent(state: OrchestratorState) -> dict:
    """Analyze the research output."""
    analysis = analyze_data(state["research_output"], state["user_request"])
    return {
        "analysis_output": analysis,
        "steps_taken": ["analysis_agent completed"],
    }

def writer_agent(state: OrchestratorState) -> dict:
    """Synthesize everything into a final report."""
    resp = llm.invoke([
        SystemMessage(content="Write a clear, structured report based on the research and analysis."),
        HumanMessage(content=f"""User question: {state['user_request']}
Research: {state['research_output']}
Analysis: {state['analysis_output']}

Write a 2-3 paragraph report.""")
    ])
    return {"final_report": resp.content, "steps_taken": ["writer_agent completed"]}

def route_supervisor(state: OrchestratorState) -> str:
    return state["next_step"]

# Build the orchestrator graph
graph = StateGraph(OrchestratorState)
graph.add_node("supervisor", supervisor)
graph.add_node("research", research_agent)
graph.add_node("analyze", analysis_agent)
graph.add_node("write", writer_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {"research": "research", "analyze": "analyze", "write": "write"},
)
graph.add_edge("research", "supervisor")   # Return to supervisor after each specialist
graph.add_edge("analyze", "supervisor")
graph.add_edge("write", END)

app = graph.compile()

result = app.invoke({
    "user_request": "What is the current state of AI adoption in enterprises?",
    "research_output": "",
    "analysis_output": "",
    "final_report": "",
    "next_step": "",
    "steps_taken": [],
})

print(f"Steps: {result['steps_taken']}")
print(f"\nReport:\n{result['final_report']}")
```

---

## Critic-rewrite loop

Add a critic agent that reviews the writer's output and triggers a rewrite if quality is low:

```python
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))

class ReportState(TypedDict):
    topic: str
    report: str
    critique: str
    approved: bool
    revision_count: int

def writer(state: ReportState) -> dict:
    prompt = state["topic"]
    if state.get("critique"):
        prompt += f"\n\nRevision feedback: {state['critique']}"
    resp = llm.invoke([
        SystemMessage(content="Write a factual 3-paragraph report."),
        HumanMessage(content=prompt)
    ])
    return {"report": resp.content, "revision_count": state.get("revision_count", 0) + 1}

def critic(state: ReportState) -> dict:
    resp = llm.invoke([
        SystemMessage(content="""Review this report. Check: accuracy, completeness, clarity.
Reply JSON: {"approved": true/false, "critique": "specific improvements needed"}"""),
        HumanMessage(content=f"Topic: {state['topic']}\nReport: {state['report']}")
    ])
    import json, re
    try:
        match = re.search(r'\{.*\}', resp.content, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        return {"approved": bool(data.get("approved", False)), "critique": data.get("critique", "")}
    except Exception:
        return {"approved": True, "critique": ""}  # Fail open

def route_after_critique(state: ReportState) -> Literal["approve", "revise"]:
    if state["approved"] or state["revision_count"] >= 3:
        return "approve"
    return "revise"

graph = StateGraph(ReportState)
graph.add_node("write", writer)
graph.add_node("critique", critic)
graph.add_edge(START, "write")
graph.add_edge("write", "critique")
graph.add_conditional_edges("critique", route_after_critique, {"approve": END, "revise": "write"})
app = graph.compile()

result = app.invoke({
    "topic": "The impact of large language models on software development productivity",
    "report": "", "critique": "", "approved": False, "revision_count": 0
})
print(f"Revisions: {result['revision_count']} | Approved: {result['approved']}")
print(f"\nFinal Report:\n{result['report'][:500]}")
```

> [!success] Two agents reviewing each other catches more errors than one agent reviewing itself
> Self-critique is biased toward the agent's own output. A separate critic with a different system prompt (and potentially a different model) gives more useful, honest feedback.

---

[[04-conditional-routing]] | [[06-practice-exercises]]
