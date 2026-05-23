# Tracing and Logging

An LLM call that fails silently is the worst kind of production bug. You see a bad output in your dashboard but you have no record of the prompt that caused it, the model version, the latency, or the token count. Tracing captures this context for every call. Logging captures it in a queryable, persistent format.

## Learning objectives

- Implement structured logging for LLM requests and responses
- Build a trace context that captures all inputs and outputs
- Wrap the OpenAI client with a transparent logging layer
- Measure and log per-call latency and token usage
- Design a log schema that supports debugging and cost analysis

---

## What to capture in every LLM trace

```python
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json

@dataclass
class LLMTrace:
    # Identity
    trace_id: str
    session_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Request
    model: str = ""
    messages: list[dict] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    tools: Optional[list] = None

    # Response
    response: str = ""
    finish_reason: str = ""

    # Metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    # Context
    function_name: str = ""    # Which part of your code called the LLM
    tags: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)
```

---

## Logging wrapper for the OpenAI client

```python
import os
import time
import uuid
import json
import logging
from openai import OpenAI
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # JSON lines
)
logger = logging.getLogger("llm_tracer")

# Pricing per 1M tokens (update as prices change)
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = PRICING.get(model, {"input": 0.15, "output": 0.60})
    return (prompt_tokens / 1e6 * price["input"]) + (completion_tokens / 1e6 * price["output"])

class TracedOpenAI:
    """OpenAI client wrapper that logs every call as a structured trace."""

    def __init__(self, session_id: str = None):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.traces: list[dict] = []

    def chat(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        function_name: str = "unknown",
        tags: list[str] = None,
        **kwargs,
    ) -> str:
        trace_id = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()
        error = None

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            reply = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            usage = response.usage
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            trace = {
                "trace_id": trace_id,
                "session_id": self.session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "model": model,
                "function_name": function_name,
                "tags": tags or [],
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if not error else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if not error else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if not error else 0,
                "cost_usd": compute_cost(
                    model,
                    getattr(usage, "prompt_tokens", 0) if not error else 0,
                    getattr(usage, "completion_tokens", 0) if not error else 0,
                ),
                "latency_ms": round(latency_ms, 1),
                "finish_reason": finish_reason if not error else "error",
                "error": error,
                "input_preview": messages[-1]["content"][:100] if messages else "",
                "output_preview": reply[:100] if not error else "",
            }
            self.traces.append(trace)
            logger.info(json.dumps(trace))

        return reply

    def summary(self) -> dict:
        if not self.traces:
            return {}
        return {
            "total_calls": len(self.traces),
            "total_tokens": sum(t["total_tokens"] for t in self.traces),
            "total_cost_usd": round(sum(t["cost_usd"] for t in self.traces), 6),
            "avg_latency_ms": round(sum(t["latency_ms"] for t in self.traces) / len(self.traces), 1),
            "errors": sum(1 for t in self.traces if t["error"]),
        }

# Usage
client = TracedOpenAI(session_id="demo")

client.chat(
    messages=[{"role": "user", "content": "What is Python?"}],
    model="gpt-4o-mini",
    function_name="explain_concept",
    tags=["education", "python"],
)

client.chat(
    messages=[{"role": "user", "content": "Give me a code example."}],
    model="gpt-4o-mini",
    function_name="code_example",
    tags=["education", "python", "code"],
)

print("\nSession summary:")
print(json.dumps(client.summary(), indent=2))
```

---

## Log schema best practices

```python
# Good log entry — queryable, complete
GOOD_LOG = {
    "trace_id": "a3f8c2",
    "session_id": "user-XYZ-session-001",
    "timestamp": "2025-05-24T10:23:45.123Z",
    "model": "gpt-4o-mini",
    "function_name": "classify_ticket",       # Where in your code this was called
    "tags": ["support", "classification"],
    "prompt_tokens": 245,
    "completion_tokens": 3,
    "cost_usd": 0.0000037,
    "latency_ms": 412,
    "finish_reason": "stop",
    "error": None,
}

# Bad log entry — not actionable when things go wrong
BAD_LOG = {
    "time": "10:23",
    "response": "negative",     # No cost, no tokens, no request context
}
```

> [!success] Log input previews and output previews, not full prompts
> Full prompts can be thousands of tokens. Store previews (first 100–200 chars) in your fast log. Store full payloads in cheap object storage (S3, GCS) keyed by `trace_id` — retrieve on demand when debugging.

---

[[00-agenda]] | [[02-langsmith]]
