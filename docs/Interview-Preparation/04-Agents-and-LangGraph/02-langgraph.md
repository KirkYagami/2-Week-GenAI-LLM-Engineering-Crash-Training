# LangGraph

LangGraph questions appear in senior LLM engineering interviews. Interviewers want to know whether you understand stateful graph execution — not just that you've imported StateGraph and run an example.

---

## Q1: What is LangGraph and why does it exist?

??? "Show answer"
    LangGraph is a library for building stateful, graph-structured LLM applications. It exists to solve three problems that raw agent loops can't handle reliably:

    1. **Explicit state** — instead of passing state implicitly through a prompt, you define a `TypedDict` that flows through every node. Any node can read and write state.
    2. **Conditional routing** — edges can be functions that inspect state and route to different nodes. This makes branching logic explicit and testable.
    3. **Persistence and human-in-the-loop** — checkpointing via `MemorySaver` lets you pause a graph mid-execution, inspect or modify state, and resume. This is essential for approval workflows.

    ```python
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class AgentState(TypedDict):
        query: str
        answer: str
        attempts: int

    def generate(state: AgentState) -> AgentState:
        # call LLM, update state
        return {"answer": "...", "attempts": state["attempts"] + 1}

    graph = StateGraph(AgentState)
    graph.add_node("generate", generate)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)
    app = graph.compile()
    ```

---

## Q2: What is an `Annotated` reducer and why do you need one?

??? "Show answer"
    By default, each node in a LangGraph graph **replaces** the current state value for any key it returns. If two nodes both write to `messages`, the second one overwrites the first.

    A reducer changes this: instead of replacing, it defines how to **merge** the new value with the existing one. The most common reducer is `operator.add` for list fields — it appends rather than replaces.

    ```python
    from typing import Annotated
    from typing_extensions import TypedDict
    import operator

    class ResearchState(TypedDict):
        query: str
        # Each node that writes findings APPENDS to the list, not replaces
        findings: Annotated[list[str], operator.add]
        quality_score: float

    # Node 1 writes ["finding about topic A"]
    # Node 2 writes ["finding about topic B"]
    # Result: findings = ["finding about topic A", "finding about topic B"]
    ```

    Without the reducer, Node 2 would overwrite Node 1's findings. This is critical for any multi-node pipeline that accumulates results.

---

## Q3: How does conditional routing work in LangGraph?

??? "Show answer"
    `add_conditional_edges` takes a source node and a routing function that inspects state and returns the name of the next node.

    ```python
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class State(TypedDict):
        draft: str
        quality_score: float
        attempts: int

    def route_after_review(state: State) -> str:
        if state["quality_score"] >= 0.80 or state["attempts"] >= 3:
            return "finalize"
        return "writer"   # loop back

    graph = StateGraph(State)
    graph.add_node("writer", write_draft)
    graph.add_node("reviewer", review_draft)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"finalize": "finalize", "writer": "writer"},
    )
    graph.add_edge("finalize", END)
    ```

    The mapping dict `{"finalize": "finalize", "writer": "writer"}` maps return values from the routing function to node names. This makes routing logic explicit and testable — call `route_after_review(state)` in a unit test without running the whole graph.

---

## Q4: What is `MemorySaver` and how does checkpointing work?

??? "Show answer"
    `MemorySaver` is an in-memory checkpointer that saves graph state after each node execution. This enables:
    - **Resume after failure** — if a node fails, resume from the last checkpoint
    - **Human-in-the-loop** — pause before a node, let a human inspect/modify state, then continue
    - **Multi-turn conversation** — maintain state across multiple `.invoke()` calls using a `thread_id`

    ```python
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "user-123"}}

    # First turn
    result1 = app.invoke({"query": "What is RAG?"}, config=config)

    # Second turn — state from first turn is preserved
    result2 = app.invoke({"query": "How does it handle failures?"}, config=config)
    ```

    For production, replace `MemorySaver` with `SqliteSaver` or `PostgresSaver` so state survives process restarts.

---

## Q5: What is `interrupt_before` and when would you use it?

??? "Show answer"
    `interrupt_before` pauses graph execution before the specified node runs. The graph returns control to the caller, who can inspect the state, optionally modify it, and then resume by calling `.invoke()` again with `None` as input.

    ```python
    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    # Pause before "execute" so a human can approve the plan
    app = graph.compile(checkpointer=memory, interrupt_before=["execute"])

    config = {"configurable": {"thread_id": "approval-flow-1"}}

    # Run until the interrupt
    result = app.invoke({"task": "Send email to all users"}, config=config)
    # result contains the state BEFORE execute runs — human reviews here

    # Human approves — resume from the interrupt
    final = app.invoke(None, config=config)
    ```

    Use `interrupt_before` for:
    - **High-stakes actions** — sending emails, writing to databases, making purchases
    - **QA checkpoints** — let a reviewer approve generated content before it's published
    - **Debugging** — step through a graph node by node

---

*Previous: [Agent Patterns](01-agent-patterns.md) | Next: [Tool Use](03-tool-use.md)*
