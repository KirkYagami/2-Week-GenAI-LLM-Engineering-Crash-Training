# Agent System Design

Agent system design questions test whether you can reason about reliability and observability in non-deterministic systems — the hardest part of production agent engineering.

---

## Q1: Design a LangGraph research agent for a news summarization product.

??? "Show answer"
    **Clarifying questions**:
    - Real-time or batch? (determines latency vs throughput tradeoff)
    - What sources? (web search, internal RSS feeds, specific APIs)
    - What's the output format? (short bullets, full article, structured JSON)
    - How do you handle breaking news where sources conflict?

    **Graph design**:

    ```python
    from langgraph.graph import StateGraph, START, END
    from typing import Annotated, TypedDict
    import operator

    class NewsState(TypedDict):
        topic: str
        queries: list[str]
        articles: Annotated[list[str], operator.add]  # accumulates across researcher nodes
        summary: str
        quality_score: float
        attempts: int

    # Nodes:
    # planner: decomposes topic into 3-5 search queries
    # researcher: searches one query, extracts relevant snippets (run in parallel)
    # writer: synthesizes articles into a structured summary
    # editor: scores quality (0-1), returns to writer if < 0.80

    def route_after_edit(state: NewsState) -> str:
        if state["quality_score"] >= 0.80 or state["attempts"] >= 3:
            return END
        return "writer"

    graph = StateGraph(NewsState)
    graph.add_node("planner", plan_queries)
    graph.add_node("researcher", research_parallel)
    graph.add_node("writer", write_summary)
    graph.add_node("editor", edit_and_score)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "editor")
    graph.add_conditional_edges("editor", route_after_edit, {END: END, "writer": "writer"})
    ```

    **Parallel researcher nodes** using `asyncio.gather` inside a single node:

    ```python
    import asyncio, os
    from openai import AsyncOpenAI

    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def research_parallel(state: NewsState) -> dict:
        results = await asyncio.gather(*[search_and_extract(q) for q in state["queries"]])
        return {"articles": list(results)}  # operator.add appends to existing
    ```

---

## Q2: How do you make agents reliable enough for production?

??? "Show answer"
    Agents are non-deterministic — the same input can produce different paths. Reliability requires:

    **1. Narrow the tool set** — each additional tool is another surface for the model to misuse. Remove tools that overlap in purpose or are rarely called.

    **2. Validate tool calls before executing** — check that arguments are valid before running any side effects.

    ```python
    def safe_execute(tool_name: str, args: dict) -> str:
        if tool_name not in REGISTERED_TOOLS:
            return f"ERROR: Unknown tool '{tool_name}'"
        tool = REGISTERED_TOOLS[tool_name]
        try:
            validated_args = tool.input_schema(**args)  # Pydantic validation
            return tool.execute(**validated_args.dict())
        except Exception as e:
            return f"ERROR: {e}"
    ```

    **3. Step limits and timeouts** — hard caps prevent infinite loops.

    **4. Structured output for key decisions** — force routing nodes to output a typed decision (not free text).

    **5. Regression tests with fixed seeds** — run your agent on 50 representative inputs before every deployment, compare outputs to expected outcomes.

    **6. Human-in-the-loop for irreversible actions** — use `interrupt_before` for any node that sends emails, modifies databases, or calls external services.

---

## Q3: How would you implement human-in-the-loop for a high-stakes agent?

??? "Show answer"
    LangGraph's `interrupt_before` pauses execution before a specified node. A human reviews the state, optionally modifies it, and resumes.

    ```python
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph

    memory = MemorySaver()
    # Pause before any node that takes irreversible action
    app = graph.compile(
        checkpointer=memory,
        interrupt_before=["send_email", "create_ticket", "update_database"],
    )

    config = {"configurable": {"thread_id": "approval-123"}}

    # Agent runs until interrupt
    result = app.invoke({"task": "Send onboarding email to 500 new users"}, config=config)
    pending_state = result  # human reviews this

    # API endpoint for human approval
    @app.post("/approve/{thread_id}")
    async def approve(thread_id: str, modifications: dict = None):
        config = {"configurable": {"thread_id": thread_id}}
        if modifications:
            # Optionally update state before resuming
            app.update_state(config, modifications)
        # Resume from interrupt
        return app.invoke(None, config=config)

    @app.post("/reject/{thread_id}")
    async def reject(thread_id: str, reason: str):
        # Log and discard — don't resume
        log_rejection(thread_id, reason)
        return {"status": "rejected"}
    ```

    For asynchronous approval workflows (reviewer not online), store the thread_id in a database and notify via email/Slack. The agent state persists in the checkpointer until resumed.

---

## Q4: How do you scale an agent that runs many tool calls per request?

??? "Show answer"
    Each tool call adds latency. With 5 tool calls per request at 500ms each = 2.5s of tool latency alone. Strategies:

    **Parallel tool execution** — if tool calls are independent, run them concurrently:

    ```python
    import asyncio, json, os
    from openai import AsyncOpenAI

    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def execute_parallel_tools(tool_calls: list) -> list[str]:
        return await asyncio.gather(*[
            execute_tool(tc.function.name, json.loads(tc.function.arguments))
            for tc in tool_calls
        ])
    ```

    **Async task queue for long-running agents** — if an agent run takes > 30 seconds, don't block the HTTP request. Use a job queue:

    ```python
    from fastapi import FastAPI, BackgroundTasks

    app = FastAPI()
    jobs: dict[str, dict] = {}  # in-memory; use Redis in production

    @app.post("/agent/start")
    async def start_agent(query: str, background_tasks: BackgroundTasks):
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "running", "result": None}
        background_tasks.add_task(run_agent, job_id, query)
        return {"job_id": job_id}

    @app.get("/agent/status/{job_id}")
    async def get_status(job_id: str):
        return jobs.get(job_id, {"status": "not_found"})
    ```

    **Tool result caching** — cache deterministic tool calls (web search for the same query, database lookups by ID) with a short TTL.

---

## Q5: How do you monitor and debug agents in production?

??? "Show answer"
    Agents are harder to debug than chains because the execution path is non-deterministic. Structured observability is essential.

    **What to capture per agent run**:

    ```python
    from langsmith import traceable

    @traceable(name="agent_run", tags=["production"])
    async def run_agent(query: str, user_id: str) -> str:
        # LangSmith automatically captures:
        # - All nested LLM calls with inputs/outputs
        # - Tool calls and their results
        # - Total tokens and cost
        # - Latency per step
        result = await agent_graph.ainvoke({"query": query})
        return result["answer"]
    ```

    **Metrics to alert on**:
    - Average steps per run (spike = agent looping or task complexity increase)
    - Tool call error rate (spike = external service degradation)
    - Runs that hit step limit (increase = agent failing to complete tasks)
    - Cost per run (spike = prompt injection or runaway loops)

    **Debugging a specific failure**:
    1. Retrieve the LangSmith trace for the run ID
    2. Inspect the state at each step — what did the model see, what did it decide?
    3. Look for: hallucinated tool calls, correct tool call with wrong arguments, loop patterns, premature final answers

    LangSmith's trace viewer makes this dramatically faster than reading raw logs — every node's input/output is visible with latency and token counts at each step.

---

*Previous: [RAG Architecture](02-rag-architecture.md) | Next: [Mock Interview Simulator](../08-mock-interview-simulator.md)*
