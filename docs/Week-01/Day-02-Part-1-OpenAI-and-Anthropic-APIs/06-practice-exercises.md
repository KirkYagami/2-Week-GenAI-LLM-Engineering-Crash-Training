# Practice Exercises — OpenAI and Anthropic APIs

Three levels: warm-up modifies a single parameter, main builds a mini-pipeline, stretch handles failure cases.

---

## Exercise 1 — Streaming vs. non-streaming latency (Warm-up)

Measure the time-to-first-token difference between streaming and non-streaming for the same prompt.

```python
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = "Write a 200-word explanation of how neural networks learn."

def measure_non_streaming() -> tuple[float, str]:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=300
    )
    elapsed = time.perf_counter() - start
    return elapsed, response.choices[0].message.content

def measure_streaming() -> tuple[float, str]:
    start = time.perf_counter()
    first_token_time = None
    full_text = ""

    with client.chat.completions.stream(
        model="gpt-4o",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=300
    ) as stream:
        for text in stream.text_stream:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            full_text += text

    total_time = time.perf_counter() - start
    return first_token_time, total_time, full_text

non_stream_time, _ = measure_non_streaming()
first_token, total_time, _ = measure_streaming()

print(f"Non-streaming total time:  {non_stream_time:.2f}s")
print(f"Streaming time-to-first-token: {first_token:.2f}s")
print(f"Streaming total time:          {total_time:.2f}s")
```

**Expected result:** Streaming time-to-first-token should be 0.5–1.5s while non-streaming total is 2–5s. This is why chat interfaces feel faster even though total generation time is similar.

---

## Exercise 2 — Multi-provider comparison (Main)

Send the same prompt to GPT-4o and Claude Sonnet 4.6, compare responses and cost.

```python
import os
from openai import OpenAI
from anthropic import Anthropic
from dataclasses import dataclass

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@dataclass
class ComparisonResult:
    prompt: str
    openai_response: str
    openai_cost_usd: float
    anthropic_response: str
    anthropic_cost_usd: float

def compare_providers(prompt: str, system: str = "") -> ComparisonResult:
    messages = [{"role": "user", "content": prompt}]

    # OpenAI
    oai_kwargs = {"model": "gpt-4o", "messages": messages, "max_tokens": 300}
    if system:
        oai_kwargs["messages"] = [{"role": "system", "content": system}] + messages
    oai_resp = openai_client.chat.completions.create(**oai_kwargs)
    oai_text = oai_resp.choices[0].message.content
    oai_cost = (oai_resp.usage.prompt_tokens * 2.50 + oai_resp.usage.completion_tokens * 10.00) / 1_000_000

    # Anthropic
    ant_kwargs = {"model": "claude-sonnet-4-6", "max_tokens": 300, "messages": messages}
    if system:
        ant_kwargs["system"] = system
    ant_resp = anthropic_client.messages.create(**ant_kwargs)
    ant_text = ant_resp.content[0].text
    ant_cost = (ant_resp.usage.input_tokens * 3.00 + ant_resp.usage.output_tokens * 15.00) / 1_000_000

    return ComparisonResult(
        prompt=prompt,
        openai_response=oai_text,
        openai_cost_usd=oai_cost,
        anthropic_response=ant_text,
        anthropic_cost_usd=ant_cost
    )

result = compare_providers(
    prompt="What are three non-obvious risks of using LLMs in production systems?",
    system="You are a senior ML engineer with production deployment experience."
)

print("=== GPT-4o ===")
print(result.openai_response)
print(f"Cost: ${result.openai_cost_usd:.6f}\n")

print("=== Claude Sonnet 4.6 ===")
print(result.anthropic_response)
print(f"Cost: ${result.anthropic_cost_usd:.6f}\n")

print(f"OpenAI vs Anthropic cost ratio: {result.openai_cost_usd / result.anthropic_cost_usd:.2f}x")
```

**Extension:** Run the same comparison on 10 different prompts and calculate average cost per response. Which model is cheaper per useful output token?

---

## Exercise 3 — Tool-using research assistant (Main)

Build an assistant that uses tools to look up information before answering.

```python
import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simulate a knowledge base and calculator
KNOWLEDGE_BASE = {
    "transformer": "A transformer is a neural network architecture based on self-attention, introduced in 'Attention Is All You Need' (2017).",
    "rag": "RAG (Retrieval-Augmented Generation) combines a retrieval system with a language model to answer questions using external documents.",
    "fine-tuning": "Fine-tuning adapts a pretrained model to a specific task using labeled examples, updating model weights.",
    "langchain": "LangChain is a framework for building LLM applications with composable chains, agents, and memory."
}

def search_docs(query: str) -> dict:
    query_lower = query.lower()
    results = []
    for key, content in KNOWLEDGE_BASE.items():
        if key in query_lower or any(word in content.lower() for word in query_lower.split()):
            results.append({"term": key, "definition": content})
    return {"query": query, "results": results or [{"term": "not found", "definition": "No matching documentation found."}]}

def calculate(expression: str) -> dict:
    try:
        # Only allow safe math expressions
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid expression — only arithmetic allowed"}
        result = eval(expression)  # safe because we validated chars above
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the technical documentation for information about AI/ML concepts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

REGISTRY = {"search_docs": search_docs, "calculate": calculate}

def research_assistant(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful AI research assistant. Always search the docs before answering technical questions."},
        {"role": "user", "content": question}
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        if response.choices[0].finish_reason != "tool_calls":
            return response.choices[0].message.content

        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)

        for tc in assistant_msg.tool_calls:
            fn = REGISTRY[tc.function.name]
            args = json.loads(tc.function.arguments)
            result = fn(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })

# Test it
print(research_assistant("What is RAG and how does it differ from fine-tuning?"))
print("\n---\n")
print(research_assistant("If I have 1000 documents and each costs $0.003 to embed, what's the total cost?"))
```

---

## Exercise 4 — Prompt caching cost analysis (Stretch)

Measure the actual cost savings from Anthropic prompt caching with a long system prompt.

```python
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Generate a ~2000-token system prompt
LONG_SYSTEM = """
You are an expert data scientist specializing in machine learning and LLM systems.
Your expertise includes:

1. MACHINE LEARNING FUNDAMENTALS
   - Supervised, unsupervised, and reinforcement learning
   - Gradient descent, backpropagation, and optimization algorithms
   - Regularization: L1/L2, dropout, early stopping
   - Ensemble methods: bagging, boosting, stacking
   - Model selection: cross-validation, hyperparameter tuning

2. DEEP LEARNING ARCHITECTURES
   - Convolutional Neural Networks (CNNs) for computer vision
   - Recurrent Neural Networks (RNNs) and LSTMs for sequences
   - Transformers and attention mechanisms
   - Diffusion models for generation
   - Graph Neural Networks for relational data

3. LLM SYSTEMS AND APPLICATIONS
   - Prompt engineering and optimization
   - Retrieval-Augmented Generation (RAG)
   - Fine-tuning with LoRA and QLoRA
   - Evaluation frameworks: RAGAS, LLM-as-judge
   - Agent architectures and tool use
   - LangChain, LangGraph, and agent frameworks

4. PRODUCTION DEPLOYMENT
   - API design for ML services
   - Model serving: FastAPI, Triton, vLLM
   - Monitoring: data drift, concept drift, performance metrics
   - Cost optimization: caching, quantization, batching
   - A/B testing for model evaluation

Always provide precise, actionable answers with code examples when relevant.
Be direct about tradeoffs and uncertainty. Cite specific techniques by name.
""" * 3  # Triple it to ensure > 1024 tokens

QUESTIONS = [
    "Explain the key difference between LoRA and full fine-tuning.",
    "What metrics should I track in a RAG evaluation pipeline?",
    "How do I implement sliding window attention for long sequences?",
    "When should I use ensemble methods vs. a single large model?"
]

def run_with_caching(question: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=[
            {
                "type": "text",
                "text": LONG_SYSTEM,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[{"role": "user", "content": question}]
    )
    return {
        "question": question[:50] + "...",
        "cache_write": response.usage.cache_creation_input_tokens,
        "cache_read": response.usage.cache_read_input_tokens,
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens
    }

# First call — cache miss (cache_write > 0)
print("Running 4 questions with prompt caching...\n")
total_cost_without_cache = 0
total_cost_with_cache = 0

for i, q in enumerate(QUESTIONS):
    stats = run_with_caching(q)
    # Cost without cache
    cost_no_cache = (stats["input"] * 3.00 + stats["output"] * 15.00) / 1_000_000
    # Cost with cache (write at 1.25x, read at 0.1x)
    cost_cache = (stats["cache_write"] * 3.75 + stats["cache_read"] * 0.30 + stats["output"] * 15.00) / 1_000_000
    total_cost_without_cache += cost_no_cache
    total_cost_with_cache += cost_cache

    print(f"Q{i+1}: {stats['question']}")
    print(f"  Cache write: {stats['cache_write']:,} | Cache read: {stats['cache_read']:,}")
    print(f"  Cost without cache: ${cost_no_cache:.6f}")
    print(f"  Cost with cache:    ${cost_cache:.6f}")
    print()

savings = (total_cost_without_cache - total_cost_with_cache) / total_cost_without_cache * 100
print(f"Total without cache: ${total_cost_without_cache:.6f}")
print(f"Total with cache:    ${total_cost_with_cache:.6f}")
print(f"Savings: {savings:.1f}%")
```

**Expected result:** After the first call (cache write), subsequent calls should show ~80–90% cost reduction on the system prompt tokens.

---

## Exercise 5 — Error handling under rate limits (Stretch)

Simulate rate limit errors and verify your retry logic handles them correctly.

```python
import os
import time
import random
from unittest.mock import patch, MagicMock
from openai import RateLimitError, OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

call_count = 0

def flaky_completion(prompt: str, fail_first_n: int = 2) -> str:
    """Simulate a function that fails with rate limits for the first N calls."""
    global call_count
    call_count += 1

    if call_count <= fail_first_n:
        # Simulate a 429 response
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "1"}
        raise RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response.choices[0].message.content

def with_retry(fn, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt + random.uniform(0, 0.5)
            print(f"Rate limit on attempt {attempt + 1}, waiting {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded")

# Test: should succeed after 2 failures
call_count = 0
result = with_retry(lambda: flaky_completion("Say 'success' and nothing else.", fail_first_n=2))
print(f"Final result: {result}")
print(f"Total API calls made: {call_count}")
assert call_count == 3, f"Expected 3 calls (2 failures + 1 success), got {call_count}"
print("Test passed!")
```

---

[[05-cost-and-rate-limits]] | [[07-interview-questions]]
