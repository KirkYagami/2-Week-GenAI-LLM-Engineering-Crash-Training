# LangSmith

LangSmith is LangChain's observability platform for LLM applications. It captures traces automatically when you use LangChain components, lets you inspect every prompt and response, build evaluation datasets from production traces, and run automated evaluations. You can also use it with non-LangChain code via its Python SDK.

## Learning objectives

- Enable LangSmith tracing with environment variables
- Use the `@traceable` decorator for custom functions
- Build evaluation datasets from production traces
- Run LLM-as-judge evaluations on a dataset
- Interpret the LangSmith dashboard

---

## Setup

```python
import os

# Set these before importing langchain/langgraph
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "my-llm-app"  # Groups traces in the UI

# All LangChain/LangGraph calls are now automatically traced
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
response = llm.invoke([HumanMessage(content="What is RAG?")])
# → Visible in LangSmith UI immediately
```

---

## Tracing custom (non-LangChain) code

```python
import os
from langsmith import traceable
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@traceable(name="classify_support_ticket", tags=["support", "classification"])
def classify_ticket(ticket_text: str) -> str:
    """Classify a support ticket. Traced automatically by LangSmith."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify as: billing, technical, account, or shipping."},
            {"role": "user", "content": ticket_text}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

@traceable(name="full_support_pipeline")
def handle_ticket(ticket: str) -> dict:
    """Multi-step pipeline — each step traced as a child span."""
    category = classify_ticket(ticket)
    response = generate_response(ticket, category)
    return {"category": category, "response": response}

@traceable(name="generate_response")
def generate_response(ticket: str, category: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You handle {category} support issues. Be concise."},
            {"role": "user", "content": ticket}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

# Test — all calls appear in LangSmith as a nested trace
result = handle_ticket("I was charged twice for my subscription this month!")
print(f"Category: {result['category']}")
print(f"Response: {result['response'][:200]}")
```

---

## Building evaluation datasets

```python
import os
from langsmith import Client

ls_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# Create a dataset from hand-labeled examples
dataset = ls_client.create_dataset(
    "support-ticket-classification-v1",
    description="Labeled support tickets for classifier evaluation"
)

examples = [
    {
        "inputs": {"ticket": "My credit card was charged twice this month."},
        "outputs": {"category": "billing"}
    },
    {
        "inputs": {"ticket": "The app crashes every time I try to upload a file."},
        "outputs": {"category": "technical"}
    },
    {
        "inputs": {"ticket": "I can't log into my account after the password reset."},
        "outputs": {"category": "account"}
    },
    {
        "inputs": {"ticket": "My order was supposed to arrive yesterday but hasn't shown up."},
        "outputs": {"category": "shipping"}
    },
]

ls_client.create_examples(
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id,
)
print(f"Dataset created: {dataset.id} with {len(examples)} examples")
```

---

## Running evaluations

```python
import os
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator

ls_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# The function to evaluate
@traceable
def classify_ticket_v2(inputs: dict) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reply with exactly one word: billing, technical, account, or shipping."},
            {"role": "user", "content": inputs["ticket"]}
        ],
        temperature=0.0,
    )
    return {"category": response.choices[0].message.content.strip().lower()}

# Exact match evaluator
def exact_match(run, example) -> dict:
    predicted = run.outputs.get("category", "").strip().lower()
    expected = example.outputs.get("category", "").strip().lower()
    return {"key": "exact_match", "score": 1 if predicted == expected else 0}

# Run evaluation
results = evaluate(
    classify_ticket_v2,
    data="support-ticket-classification-v1",  # Dataset name
    evaluators=[exact_match],
    experiment_prefix="classifier-v2",
)
print(f"Evaluation complete. Results visible in LangSmith UI.")
```

---

## Key LangSmith concepts

| Concept | Description |
|---------|-------------|
| **Run** | A single traced operation (one LLM call, one chain invocation) |
| **Trace** | A tree of runs — the full execution of a request |
| **Dataset** | Collection of input/output examples for evaluation |
| **Experiment** | Running a function against a dataset; stores results for comparison |
| **Evaluator** | Function that scores an experiment result |
| **Feedback** | Human or automated rating attached to a run |

> [!tip] Use the LangSmith trace explorer to debug production failures
> When a user reports a bad output, search by session_id or trace_id in LangSmith to see the exact prompt, model response, token counts, and all child spans. No more asking users "what did you type exactly?"

> [!warning] LangSmith stores your prompts and responses
> This is a data privacy consideration. Review your organization's data handling policy before enabling LangSmith tracing in production. Use `filter_inputs` and `filter_outputs` to redact PII from traces.

---

[[01-tracing-and-logging]] | [[03-cost-tracking]]
