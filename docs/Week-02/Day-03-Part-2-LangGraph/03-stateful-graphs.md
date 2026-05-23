# Stateful Graphs

State is what makes LangGraph graphs more than just function pipelines. The state object flows through every node, accumulates information, and determines routing decisions. Understanding how state is defined, updated, and persisted unlocks the full power of the framework.

## Learning objectives

- Define state with `TypedDict` and understand the merge semantics
- Use `Annotated` with reducers to control how fields accumulate
- Implement checkpointing to persist and resume graph state
- Build a graph that tracks intermediate results across nodes
- Read state at any point in the graph execution

---

## State design principles

```python
from typing import TypedDict, Annotated, Optional
import operator

# ── Simple state: last write wins ─────────────────────────────────
class SimpleState(TypedDict):
    query: str           # If two nodes write "query", the last one wins
    result: str
    step: int

# ── Accumulating state: reducer controls merge ────────────────────
from langchain_core.messages import BaseMessage

class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]  # Appends each new message
    metadata: dict  # Last write wins

# ── Custom reducers ───────────────────────────────────────────────
def keep_unique(existing: list, new: list) -> list:
    """Reducer that deduplicates list items."""
    return list(dict.fromkeys(existing + new))

class ResearchState(TypedDict):
    question: str
    search_results: Annotated[list[str], keep_unique]   # Deduplicates
    sources: Annotated[list[str], operator.add]         # Appends all
    answer: str
    quality_score: float
```

---

## A complete stateful research graph

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

class ResearchState(TypedDict):
    question: str
    search_queries: Annotated[list[str], operator.add]
    findings: Annotated[list[str], operator.add]
    answer: str
    quality: str  # "sufficient" or "insufficient"
    iteration: int

def query_generator(state: ResearchState) -> dict:
    """Generate search queries from the question."""
    resp = llm.invoke([
        SystemMessage(content="Generate 3 specific search queries for this research question. One per line."),
        HumanMessage(content=state["question"])
    ])
    queries = [q.strip() for q in resp.content.strip().split("\n") if q.strip()][:3]
    return {"search_queries": queries, "iteration": state.get("iteration", 0) + 1}

def researcher(state: ResearchState) -> dict:
    """Search for each query and collect findings."""
    # Simulated search
    knowledge = {
        "python": "Python was created by Guido van Rossum in 1991. It emphasizes readability.",
        "performance": "Python is slower than C/C++ for CPU-bound tasks but fast enough for I/O tasks.",
        "use cases": "Python is used for data science, web dev, automation, and ML.",
        "popularity": "Python is the #1 language on TIOBE as of 2024.",
    }
    findings = []
    for query in state["search_queries"]:
        for key, val in knowledge.items():
            if key in query.lower():
                findings.append(f"[{query}]: {val}")
                break
        else:
            findings.append(f"[{query}]: No specific data found.")
    return {"findings": findings}

def synthesizer(state: ResearchState) -> dict:
    """Synthesize findings into an answer."""
    findings_text = "\n".join(state["findings"])
    resp = llm.invoke([
        SystemMessage(content="Synthesize research findings into a comprehensive answer."),
        HumanMessage(content=f"Question: {state['question']}\n\nFindings:\n{findings_text}")
    ])
    return {"answer": resp.content}

def quality_checker(state: ResearchState) -> dict:
    """Evaluate if the answer is sufficient."""
    resp = llm.invoke([
        SystemMessage(content="Evaluate if this answer adequately addresses the question. Reply 'sufficient' or 'insufficient'."),
        HumanMessage(content=f"Q: {state['question']}\nA: {state['answer']}")
    ])
    quality = "sufficient" if "sufficient" in resp.content.lower() else "insufficient"
    return {"quality": quality}

def route_quality(state: ResearchState) -> str:
    """Route based on quality and iteration count."""
    if state["quality"] == "sufficient" or state.get("iteration", 0) >= 3:
        return "done"
    return "retry"

# Build graph
graph = StateGraph(ResearchState)
graph.add_node("query_gen", query_generator)
graph.add_node("research", researcher)
graph.add_node("synthesize", synthesizer)
graph.add_node("quality_check", quality_checker)

graph.add_edge(START, "query_gen")
graph.add_edge("query_gen", "research")
graph.add_edge("research", "synthesize")
graph.add_edge("synthesize", "quality_check")
graph.add_conditional_edges(
    "quality_check",
    route_quality,
    {"done": END, "retry": "query_gen"},  # Retry loops back
)

app = graph.compile()

result = app.invoke({
    "question": "What are Python's strengths and use cases?",
    "search_queries": [],
    "findings": [],
    "answer": "",
    "quality": "",
    "iteration": 0,
})
print(f"Iterations: {result['iteration']}")
print(f"Queries used: {len(result['search_queries'])}")
print(f"Quality: {result['quality']}")
print(f"\nAnswer:\n{result['answer']}")
```

---

## Checkpointing: persist and resume

LangGraph's checkpointing saves state after each node, enabling pause/resume and human-in-the-loop:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class SimpleState(TypedDict):
    value: int

def increment(state: SimpleState) -> dict:
    return {"value": state["value"] + 1}

graph = StateGraph(SimpleState)
graph.add_node("increment", increment)
graph.add_edge(START, "increment")
graph.add_edge("increment", END)

# Compile with a checkpointer
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Each thread_id is an independent conversation/session
config = {"configurable": {"thread_id": "session-001"}}

# First invocation
result1 = app.invoke({"value": 0}, config=config)
print(f"After first run: {result1['value']}")  # → 1

# Resume same session — state is restored from checkpoint
result2 = app.invoke({"value": result1["value"]}, config=config)
print(f"After second run: {result2['value']}")  # → 2

# Get state at any point
state_snapshot = app.get_state(config)
print(f"Current state: {state_snapshot.values}")
```

---

## Streaming state updates

```python
# Stream shows each node's output as it completes
for chunk in app.stream({"value": 0}, config=config):
    for node_name, state_update in chunk.items():
        print(f"Node '{node_name}' updated state: {state_update}")
```

> [!success] Use checkpointing for any agent that might need to pause
> Even if you don't use human-in-the-loop today, adding a `MemorySaver` checkpointer costs nothing and gives you the ability to inspect intermediate state, resume failed runs, and add human review later without restructuring your graph.

---

[[02-nodes-and-edges]] | [[04-conditional-routing]]
