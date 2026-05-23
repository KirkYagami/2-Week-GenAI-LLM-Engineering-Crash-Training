# Conditional Routing

Conditional edges are the branching mechanism in LangGraph. They let the graph choose where to go next based on the current state — routing to different nodes, looping back, or terminating. This is what enables retry loops, quality gates, and classifier-driven workflows.

## Learning objectives

- Write router functions that return node names
- Use `add_conditional_edges` with explicit edge maps
- Build a retry loop with a termination condition
- Implement a classifier that routes to specialized nodes
- Combine conditional and unconditional edges in one graph

---

## The router function pattern

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

class State(TypedDict):
    score: float
    answer: str
    attempts: int

def route_by_score(state: State) -> Literal["pass", "retry", "give_up"]:
    """Router function: reads state, returns a string key."""
    if state["score"] >= 0.8:
        return "pass"
    elif state["attempts"] < 3:
        return "retry"
    else:
        return "give_up"

graph = StateGraph(State)
graph.add_node("generate", lambda state: {"answer": "...", "score": 0.7, "attempts": state["attempts"] + 1})
graph.add_node("final", lambda state: {"answer": f"Final: {state['answer']}"})
graph.add_node("fallback", lambda state: {"answer": "Unable to generate a high-quality answer."})

graph.add_edge(START, "generate")
graph.add_conditional_edges(
    "generate",           # Source node
    route_by_score,       # Router function
    {                     # Map: router output → node name
        "pass": "final",
        "retry": "generate",  # Loop back
        "give_up": "fallback",
    },
)
graph.add_edge("final", END)
graph.add_edge("fallback", END)
```

---

## Quality gate with retry loop

```python
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))

class WritingState(TypedDict):
    prompt: str
    draft: str
    feedback: str
    score: float
    attempts: int

def writer(state: WritingState) -> dict:
    """Generate a draft, incorporating feedback from previous attempts."""
    messages = [SystemMessage(content="Write a concise, accurate 2-sentence answer.")]
    if state.get("feedback"):
        messages.append(SystemMessage(content=f"Previous feedback: {state['feedback']}. Incorporate this."))
    messages.append(HumanMessage(content=state["prompt"]))

    resp = llm.invoke(messages)
    return {"draft": resp.content, "attempts": state.get("attempts", 0) + 1}

def critic(state: WritingState) -> dict:
    """Score the draft and provide feedback."""
    resp = llm.invoke([
        SystemMessage(content="""Evaluate this answer on a 0.0-1.0 scale.
Return JSON: {"score": 0.X, "feedback": "..."}"""),
        HumanMessage(content=f"Prompt: {state['prompt']}\nAnswer: {state['draft']}")
    ])
    import json, re
    try:
        match = re.search(r'\{.*\}', resp.content, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        return {"score": float(data.get("score", 0.5)), "feedback": data.get("feedback", "")}
    except Exception:
        return {"score": 0.5, "feedback": "Unable to parse feedback."}

def route_after_critic(state: WritingState) -> Literal["accept", "revise", "give_up"]:
    if state["score"] >= 0.85:
        return "accept"
    if state["attempts"] < 3:
        return "revise"
    return "give_up"

# Build the graph
graph = StateGraph(WritingState)
graph.add_node("write", writer)
graph.add_node("critique", critic)
graph.add_edge(START, "write")
graph.add_edge("write", "critique")
graph.add_conditional_edges(
    "critique",
    route_after_critic,
    {"accept": END, "revise": "write", "give_up": END},
)

app = graph.compile()

result = app.invoke({
    "prompt": "Explain the difference between supervised and unsupervised learning.",
    "draft": "", "feedback": "", "score": 0.0, "attempts": 0
})
print(f"Attempts: {result['attempts']} | Score: {result['score']:.2f}")
print(f"Draft: {result['draft']}")
```

---

## Classifier-driven routing

Route incoming requests to specialized nodes based on classification:

```python
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

class RequestState(TypedDict):
    user_input: str
    category: str
    response: str

def classifier(state: RequestState) -> dict:
    """Classify the user's request."""
    resp = llm.invoke([
        SystemMessage(content="Classify as: math, coding, general. Reply with one word only."),
        HumanMessage(content=state["user_input"])
    ])
    return {"category": resp.content.strip().lower()}

def route_by_category(state: RequestState) -> str:
    cat = state["category"]
    return cat if cat in ["math", "coding"] else "general"

def math_expert(state: RequestState) -> dict:
    resp = llm.invoke([
        SystemMessage(content="You are a mathematics expert. Solve step by step."),
        HumanMessage(content=state["user_input"])
    ])
    return {"response": resp.content}

def coding_expert(state: RequestState) -> dict:
    resp = llm.invoke([
        SystemMessage(content="You are a coding expert. Provide working code examples."),
        HumanMessage(content=state["user_input"])
    ])
    return {"response": resp.content}

def general_assistant(state: RequestState) -> dict:
    resp = llm.invoke([HumanMessage(content=state["user_input"])])
    return {"response": resp.content}

graph = StateGraph(RequestState)
graph.add_node("classify", classifier)
graph.add_node("math", math_expert)
graph.add_node("coding", coding_expert)
graph.add_node("general", general_assistant)

graph.add_edge(START, "classify")
graph.add_conditional_edges(
    "classify",
    route_by_category,
    {"math": "math", "coding": "coding", "general": "general"},
)
graph.add_edge("math", END)
graph.add_edge("coding", END)
graph.add_edge("general", END)

app = graph.compile()

queries = [
    "What is the derivative of x^3 + 2x?",
    "Write a Python function to reverse a string.",
    "What's the capital of Australia?",
]
for q in queries:
    result = app.invoke({"user_input": q, "category": "", "response": ""})
    print(f"Q: {q[:50]}")
    print(f"Category: {result['category']} | Response: {result['response'][:100]}...\n")
```

> [!warning] Always provide a catch-all route in conditional edges
> If your router function returns a value not in the edge map, LangGraph raises a `KeyError`. Add a default: `{"known_cat": "node_a", ..., "unknown": "fallback_node"}` — or handle the default case in the router function itself.

---

[[03-stateful-graphs]] | [[05-multi-agent-orchestration]]
