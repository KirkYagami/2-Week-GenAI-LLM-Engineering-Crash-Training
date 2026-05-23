# Interview Questions — LangGraph Research Agent

---

**Q1: Why does the `findings` field use `Annotated[list[str], operator.add]` instead of `list[str]`?**

??? "Show answer"
    LangGraph merges the state returned by each node with the existing state. Without a reducer, returning `{"findings": [...]}` from the researcher node would overwrite the existing `findings` list.
    
    `Annotated[list[str], operator.add]` tells LangGraph to use `operator.add` as the reducer — which for lists means concatenation (not integer addition). So if the current state has `findings = ["finding 1"]` and a node returns `{"findings": ["finding 2"]}`, the merged state is `findings = ["finding 1", "finding 2"]`.
    
    This matters here because the planner and researcher run at different times, and if multiple nodes contribute findings, they all accumulate rather than overwriting each other.

---

**Q2: What happens if the critic loop runs 3 times and the quality score never reaches 0.80?**

??? "Show answer"
    The `route_after_critic` function checks `state["attempts"] >= 3` before checking the quality score. If attempts have reached 3, it routes to `"finalize"` regardless of the quality score. The `finalize` node copies `draft` to `final_report` and routes to `END`.
    
    This is an important safety guard: without a max attempts limit, a low-quality writer prompt could loop indefinitely, burning tokens and hanging the request. The tradeoff: you might ship a 0.65-quality report after 3 loops. In production, you could flag low-quality reports for human review rather than returning them directly to the user.

---

**Q3: How does parallel execution in the researcher node work, and what limits its concurrency?**

??? "Show answer"
    `asyncio.gather(*[research_one(q) for q in state["sub_questions"]])` launches all sub-question research coroutines concurrently on the same event loop. They yield control at each `await aclient.chat.completions.create(...)`, allowing others to make progress.
    
    Limits: (1) OpenAI rate limits — sending 4 simultaneous requests may hit the requests-per-minute limit for low-tier accounts. Add a `asyncio.Semaphore(max_concurrent=2)` to limit to 2 at a time if you hit 429s. (2) The event loop is single-threaded — if any researcher coroutine does CPU-bound work (not awaiting), it blocks others. All LLM calls are network I/O, so this isn't an issue here.

---

**Q4: How would you add a human-in-the-loop step where a user can edit the draft before the critic evaluates it?**

??? "Show answer"
    LangGraph supports `interrupt_before` and `interrupt_after` to pause graph execution:
    
    ```python
    research_app = graph.compile(
        checkpointer=memory,
        interrupt_before=["critic"],  # Pause before critic runs
    )
    
    # First run: pauses before critic
    config = {"configurable": {"thread_id": "session-1"}}
    result = await research_app.ainvoke(initial_state, config=config)
    
    # Human reviews and edits draft
    current_state = research_app.get_state(config)
    research_app.update_state(config, {"draft": edited_draft})
    
    # Resume: critic evaluates the edited draft
    result = await research_app.ainvoke(None, config=config)
    ```
    
    This pattern lets you build a "human approval" workflow where AI-generated drafts are reviewed before quality evaluation, and poor drafts can be manually corrected rather than having the agent loop.

---

[[05-deployment]]
