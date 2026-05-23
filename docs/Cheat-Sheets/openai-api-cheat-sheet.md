# OpenAI API Cheat Sheet

Quick reference for the OpenAI Python SDK. Every snippet is copy-paste ready.

---

## Chat completions

```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Basic call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in one sentence."},
    ],
    temperature=0.0,
    max_tokens=200,
)
text = response.choices[0].message.content
tokens = response.usage.total_tokens
```

## Async chat

```python
from openai import AsyncOpenAI
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = await aclient.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 5."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

## Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Hello world", "Goodbye world"],
)
vectors = [d.embedding for d in response.data]  # list of list[float]
# text-embedding-3-small: 1536 dimensions, ~$0.02/1M tokens
```

## Function calling / tool use

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)

choice = response.choices[0]
if choice.finish_reason == "tool_calls":
    tool_call = choice.message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)  # {"city": "Paris"}
```

## Structured output (with Pydantic)

```python
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float
    sentiment: str

structured_client = client.beta.chat.completions  # or use with_structured_output

# With LangChain:
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
structured = llm.with_structured_output(MovieReview)
result = structured.invoke("Review: Inception is amazing!")
# result.title, result.rating, result.sentiment
```

## Key parameters

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `temperature` | float | 1.0 | 0=deterministic, 1=creative, >1=chaotic |
| `max_tokens` | int | model max | Hard limit on output length |
| `top_p` | float | 1.0 | Nucleus sampling; use instead of temperature |
| `n` | int | 1 | Number of completions to generate |
| `stop` | str/list | None | Stop sequences; model stops before generating these |
| `presence_penalty` | float | 0 | Penalizes repeated topics (-2 to 2) |
| `frequency_penalty` | float | 0 | Penalizes repeated tokens (-2 to 2) |
| `seed` | int | None | Deterministic output (best-effort) |

## Models quick reference

| Model | Context | Input $/1M | Output $/1M | Best for |
|-------|---------|------------|-------------|----------|
| `gpt-4o` | 128k | $2.50 | $10.00 | Complex reasoning |
| `gpt-4o-mini` | 128k | $0.15 | $0.60 | Default workhorse |
| `text-embedding-3-small` | 8k | $0.02 | — | Retrieval, similarity |
| `text-embedding-3-large` | 8k | $0.13 | — | High-accuracy retrieval |

## Error handling

```python
from openai import RateLimitError, APIError, APIConnectionError

try:
    response = client.chat.completions.create(...)
except RateLimitError:
    # Retry with backoff (SDK does this automatically)
    raise
except APIConnectionError:
    # Network issue
    raise
except APIError as e:
    # Upstream OpenAI error (5xx)
    print(f"API error {e.status_code}: {e.message}")
    raise
```
