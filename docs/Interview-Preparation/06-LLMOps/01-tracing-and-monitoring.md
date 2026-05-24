# Tracing and Monitoring

LLMOps questions reveal whether you've shipped LLM systems or just built demos. The difference between the two is mostly observability.

---

## Q1: What should you log for every LLM request in production?

??? "Show answer"
    Minimum viable logging per request:

    ```python
    import time, uuid, os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def call_with_logging(messages: list, model: str = "gpt-4o") -> str:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        try:
            response = client.chat.completions.create(model=model, messages=messages)
            latency_ms = (time.perf_counter() - start) * 1000
            usage = response.usage

            log_event({
                "request_id": request_id,
                "model": model,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "cost_usd": estimate_cost(model, usage.prompt_tokens, usage.completion_tokens),
                "status": "success",
            })

            return response.choices[0].message.content

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            log_event({"request_id": request_id, "latency_ms": latency_ms, "status": "error", "error": str(e)})
            raise
    ```

    Optional but valuable: input/output content (with PII redaction), user ID, session ID, prompt version, model temperature.

    > [!warning] Privacy
    > Don't log raw user input without considering your data retention policy and any PII obligations. At minimum, strip or hash personal identifiers before logging.

---

## Q2: How does LangSmith tracing work and what does `@traceable` capture?

??? "Show answer"
    LangSmith is Anthropic and LangChain's observability platform. It captures the full execution tree of LLM calls, including intermediate steps in chains and agents.

    The `@traceable` decorator wraps a function and records its inputs, outputs, and any nested LLM calls as a "run" in LangSmith.

    ```python
    import os
    from langsmith import traceable
    from openai import OpenAI

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = "my-rag-app"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @traceable(name="retrieve_context")
    def retrieve(query: str) -> list[str]:
        # ChromaDB call — appears as a child run in LangSmith
        return collection.query(...)["documents"][0]

    @traceable(name="rag_answer")
    def answer(query: str) -> str:
        context = retrieve(query)  # nested — shows as child trace
        response = client.chat.completions.create(...)
        return response.choices[0].message.content
    ```

    In LangSmith you can see: each function's inputs/outputs, latency per step, total token counts, and the full prompt sent to the model. This makes debugging RAG pipelines dramatically faster than reading logs.

---

## Q3: How do you detect prompt regressions in production?

??? "Show answer"
    A **prompt regression** is when a change to your system prompt, model version, or retrieval logic degrades output quality — often silently.

    Detection approach: maintain a **golden test set** — a fixed set of input/output pairs that represent correct behaviour. Run the evaluation on every deployment.

    ```python
    import os
    from langsmith import Client
    from langsmith.evaluation import evaluate

    ls_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

    def evaluator(run, example) -> dict:
        # LLM-as-judge: score the output against the expected output
        score = llm_judge(
            question=example.inputs["query"],
            answer=run.outputs["answer"],
            expected=example.outputs["expected_answer"],
        )
        return {"key": "quality", "score": score["score"]}

    results = evaluate(
        lambda inputs: {"answer": rag_answer(inputs["query"])},
        data="golden-test-set",
        evaluators=[evaluator],
        experiment_prefix="v2.1-deployment",
    )

    print(results.to_pandas()[["quality"]].mean())
    ```

    Alert if the mean quality score drops by more than 5% compared to the baseline experiment. Fail the deployment if it drops by more than 15%.

---

## Q4: What metrics should you monitor in production for an LLM API?

??? "Show answer"
    | Metric | Target | Alert when |
    |--------|--------|------------|
    | **p50 latency** | < 1s for cached, < 3s for streaming | p50 > 5s |
    | **p95 latency** | < 5s | p95 > 10s |
    | **Error rate** | < 1% | > 5% in 5-minute window |
    | **Rate limit rate** | < 0.1% | > 1% |
    | **Cost per request** | Baseline ± 20% | Sudden spike (prompt injection or runaway loops) |
    | **Faithfulness score** | > 0.80 (RAG) | < 0.70 over 100 requests |
    | **Cache hit rate** | > 20% for stable workloads | Sudden drop (cache invalidation bug) |

    Track these with your existing observability stack (Datadog, Prometheus, CloudWatch). LangSmith handles LLM-specific metrics; your infra stack handles infrastructure metrics.

---

## Q5: How do you trace an agent's execution across multiple LLM calls?

??? "Show answer"
    Agents make multiple LLM calls with branching and looping. Tracing requires correlating all calls in a session under a single parent run.

    With LangSmith, nesting `@traceable` functions creates a parent-child trace tree automatically.

    For custom tracing, use a `run_id` that flows through all calls:

    ```python
    import uuid, time

    class AgentTracer:
        def __init__(self):
            self.session_id = str(uuid.uuid4())
            self.steps: list[dict] = []
            self.start_time = time.perf_counter()

        def log_step(self, step_type: str, input_data: dict, output_data: dict, tokens: int):
            self.steps.append({
                "session_id": self.session_id,
                "step": len(self.steps) + 1,
                "type": step_type,        # "llm_call", "tool_call", "routing"
                "input": input_data,
                "output": output_data,
                "tokens": tokens,
                "elapsed_ms": (time.perf_counter() - self.start_time) * 1000,
            })

        def summary(self) -> dict:
            return {
                "session_id": self.session_id,
                "total_steps": len(self.steps),
                "total_tokens": sum(s["tokens"] for s in self.steps),
                "total_ms": (time.perf_counter() - self.start_time) * 1000,
            }
    ```

---

*Previous: [Fine-Tuning Evaluation](../05-Fine-Tuning/03-evaluation.md) | Next: [Deployment](02-deployment.md)*
