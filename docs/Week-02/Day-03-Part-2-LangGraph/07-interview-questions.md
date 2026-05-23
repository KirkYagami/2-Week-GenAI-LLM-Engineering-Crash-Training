# Interview Questions — LangGraph

---

**Q1: What problem does LangGraph solve that a plain tool-calling loop can't?**

??? "Show answer"
    A plain tool-calling loop handles one linear flow: model → tool → model → tool → answer. It can't express:

    - **Branching:** route to node A if score is high, node B if low
    - **Cycles:** retry a node based on quality output
    - **Parallelism:** execute independent nodes simultaneously
    - **Human-in-the-loop:** pause the graph and wait for human input
    - **Persistent state:** resume a graph from a checkpoint after a crash or user pause
    - **Multi-agent coordination:** orchestrator routes to specialized sub-agents

    LangGraph represents these control-flow patterns explicitly as graph structure (nodes + edges), making complex agent behavior inspectable, debuggable, and testable rather than buried in loop logic.

---

**Q2: What is the difference between a direct edge and a conditional edge in LangGraph?**

??? "Show answer"
    **Direct edge:** `graph.add_edge("node_a", "node_b")` — execution always goes from A to B. Deterministic, no branching.

    **Conditional edge:** `graph.add_conditional_edges("node_a", router_fn, {"key": "node_b", ...})` — execution goes to the node named by whatever string `router_fn(state)` returns. The router reads the current state and returns a key; the key maps to a target node.

    The router function has the signature `(state: StateType) -> str`. It must return a value that exists as a key in the edge map — otherwise LangGraph raises a `KeyError` at runtime.

    Always include a catch-all path or validate router outputs to prevent unexpected failures.

---

**Q3: Explain the `Annotated[list[X], operator.add]` pattern in LangGraph state. What does it change?**

??? "Show answer"
    Without annotation, when a node returns `{"messages": [new_msg]}`, LangGraph replaces the entire `messages` field with `[new_msg]` — losing all previous messages.

    `Annotated[list[BaseMessage], operator.add]` attaches a *reducer* to the field. The reducer defines how to merge new values with existing ones. `operator.add` for lists is list concatenation — so returning `{"messages": [new_msg]}` appends `new_msg` to the existing list instead of replacing it.

    This is essential for message history in conversational agents: every node adds its messages and all previous messages are preserved.

    You can write custom reducers too:
    ```python
    def keep_unique(existing: list, new: list) -> list:
        return list(dict.fromkeys(existing + new))

    class State(TypedDict):
        tags: Annotated[list[str], keep_unique]  # Deduplicates on merge
    ```

---

**Q4: What does `graph.compile(checkpointer=MemorySaver())` give you that compilation without a checkpointer doesn't?**

??? "Show answer"
    Without a checkpointer, the graph executes and returns. If execution fails mid-way, state is lost. There's no way to pause, inspect intermediate state, or implement human-in-the-loop.

    With `MemorySaver()` (or a persistent backend like `SqliteSaver`):
    1. **State is saved after every node** — if execution fails, you can resume from the last successful node
    2. **`app.get_state(config)`** returns the current state at any point, for inspection and debugging
    3. **Human-in-the-loop:** call `app.invoke(..., interrupt_before=["node_name"])` to pause before a specific node and wait for human approval before continuing
    4. **Separate sessions:** different `thread_id` values in config maintain completely independent state, enabling multiple concurrent users

    `MemorySaver` stores state in Python memory (process-local). For production, use `SqliteSaver` or a Redis-based checkpointer for durability across restarts.

---

**Q5: Describe the supervisor pattern in multi-agent LangGraph. What does the supervisor do, and how does it return control from specialists?**

??? "Show answer"
    In the supervisor pattern:
    - The **supervisor** is a node that reads the current state and decides which specialist to invoke next. It returns `{"next_step": "researcher"}` or similar.
    - A **conditional edge** from the supervisor routes to the correct specialist based on `next_step`.
    - **Specialist nodes** do their work and route back to the supervisor via a direct edge: `graph.add_edge("research", "supervisor")`.
    - The supervisor runs again, checks what's been completed, and routes to the next specialist or to `END`.

    This creates a hub-and-spoke topology: supervisor → specialist → supervisor → specialist → ... → END.

    The supervisor's state-reading logic determines when to stop. Example: if research is done and analysis is done and writing is done, return `"done"` which routes to END.

---

**Q6: When should you use LangGraph instead of a simpler tool-calling loop or LCEL chain?**

??? "Show answer"
    **Use LCEL chain when:** linear pipeline, no branching, no loops, fixed number of steps. Most RAG pipelines fit here.

    **Use tool-calling loop when:** unknown number of steps, dynamic tool selection, single goal. Most straightforward agents fit here.

    **Use LangGraph when any of these apply:**
    - **Retry loop:** "generate → validate → retry if bad" — needs a cycle
    - **Quality-gated workflow:** route based on score/classification
    - **Multiple specialized agents:** researcher + writer + critic with explicit handoffs
    - **Human-in-the-loop:** must pause and wait for review
    - **Persistent state:** conversation that resumes across sessions
    - **Parallel execution:** independent subtasks that can run concurrently

    The key question: is the control flow expressible as a graph? If yes, LangGraph. If it's a simple sequence, LCEL is lighter. If it's a basic tool loop, use the raw API directly.

---

[[06-practice-exercises]]
