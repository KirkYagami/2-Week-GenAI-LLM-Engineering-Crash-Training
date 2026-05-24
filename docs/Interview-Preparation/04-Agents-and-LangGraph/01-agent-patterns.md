# Agent Patterns

Agent questions test whether you understand the tradeoffs of autonomous systems — not just how to set one up, but when the complexity is justified and how to make them reliable enough to ship.

---

## Q1: What's the difference between a chain and an agent? When do you use each?

??? "Show answer"
    A **chain** executes a fixed, predetermined sequence of steps. The path is hardcoded at build time. Fast, predictable, cheap to debug.

    An **agent** decides its own sequence of actions at runtime based on intermediate results. It loops, branches, and calls tools until it determines the task is complete.

    | | Chain | Agent |
    |-|-------|-------|
    | Control flow | Fixed | Dynamic — model decides |
    | Cost | Predictable | Variable (unknown number of LLM calls) |
    | Reliability | High | Lower — depends on model quality |
    | Use when | Task steps are known in advance | Task requires planning or adapting to intermediate results |

    Use an agent when:
    - The number of steps isn't known in advance (research tasks, multi-step problem solving)
    - Intermediate results determine what to do next
    - You need tool selection based on context

    Use a chain when the task is well-defined, the sequence is fixed, and you need reliability and predictable cost.

---

## Q2: Explain the ReAct loop. What are Thought, Action, and Observation?

??? "Show answer"
    ReAct (Reasoning + Acting) interleaves chain-of-thought reasoning with tool actions:

    - **Thought**: the model reasons about what it knows and what it needs to do next
    - **Action**: the model specifies a tool call (e.g., `search("current gold price")`)
    - **Observation**: the result of executing the tool is appended to the context

    This cycle repeats until the model produces a final answer.

    ```python
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    REACT_PROMPT = """You have access to these tools:
    - search(query): searches the web
    - calculate(expression): evaluates a math expression

    Use this format:
    Thought: what you need to do
    Action: tool_name(args)
    Observation: [tool result will appear here]
    ... (repeat as needed)
    Final Answer: your answer to the original question

    Question: {question}"""

    def react_agent(question: str, tools: dict, max_steps: int = 5) -> str:
        prompt = REACT_PROMPT.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        for _ in range(max_steps):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stop=["Observation:"],
            )
            output = response.choices[0].message.content

            if "Final Answer:" in output:
                return output.split("Final Answer:")[-1].strip()

            # Parse and execute the action
            # ... execute tool, append observation, continue loop
        return "Max steps reached without final answer"
    ```

---

## Q3: What memory strategies do agents use? When do you use each?

??? "Show answer"
    | Strategy | How | Best for |
    |----------|-----|----------|
    | **In-context (full history)** | Keep all messages in the context window | Short conversations (< 20 turns) where every message matters |
    | **Sliding window** | Keep only the last N messages | Ongoing conversations where recent context matters most |
    | **Summarized** | Compress old messages with an LLM summary | Long conversations where broad context matters but not exact wording |
    | **Vector-backed** | Store messages as embeddings; retrieve relevant ones | Very long sessions, user preference recall across sessions |

    In practice:
    - Start with full in-context history
    - Switch to sliding window when you hit context limits
    - Add vector memory only when users explicitly report "the agent forgot" something from earlier in the conversation

    ```python
    from collections import deque

    class SlidingWindowMemory:
        def __init__(self, max_messages: int = 20):
            self.messages: deque = deque(maxlen=max_messages)

        def add(self, role: str, content: str):
            self.messages.append({"role": role, "content": content})

        def get(self) -> list[dict]:
            return list(self.messages)
    ```

---

## Q4: How do you prevent an agent from running indefinitely?

??? "Show answer"
    Four safeguards:

    1. **Step limit** — hard cap on the number of LLM calls in a loop
    2. **Token budget** — track cumulative tokens; stop if you exceed a threshold
    3. **Tool call validation** — detect repeated identical tool calls (loop detection)
    4. **Timeout** — wall-clock timeout on the entire agent run

    ```python
    import asyncio, os
    from openai import AsyncOpenAI

    aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def agent_with_guards(query: str, max_steps: int = 10, timeout_seconds: float = 60.0):
        steps = 0
        seen_actions = set()
        messages = [{"role": "user", "content": query}]

        async def _run():
            nonlocal steps
            while steps < max_steps:
                response = await aclient.chat.completions.create(model="gpt-4o", messages=messages)
                output = response.choices[0].message.content
                steps += 1

                if "Final Answer:" in output:
                    return output.split("Final Answer:")[-1].strip()

                # Detect repeated actions
                if output in seen_actions:
                    return "Agent stuck in loop — stopping."
                seen_actions.add(output)

        try:
            return await asyncio.wait_for(_run(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return "Agent timed out."
    ```

---

## Q5: What are the main failure modes of LLM agents?

??? "Show answer"
    | Failure | Description | Mitigation |
    |---------|-------------|------------|
    | **Hallucinated tool calls** | Model invents tool names or arguments that don't exist | Validate every tool call before executing; return errors to the model |
    | **Infinite loop** | Model keeps calling the same tool with the same args | Loop detection (track seen actions); step limit |
    | **Lost context** | Model forgets the original goal after many steps | Re-inject the original query at each step |
    | **Tool error cascades** | One tool fails, model interprets error as data and acts on it | Distinguish error strings from data; handle errors explicitly in your loop |
    | **Overconfident final answer** | Model produces a final answer from a failed tool call | Require tool calls to succeed before allowing "Final Answer" |
    | **Prompt injection via tools** | Tool output contains adversarial instructions | Sanitize tool outputs before adding to context; use a separate "tool processing" prompt |

    The most reliable agents are those with the narrowest possible tool set. Every additional tool is another surface for hallucination and misuse.

---

*Previous: [Embeddings](../03-LLM-APIs/03-embeddings.md) | Next: [LangGraph](02-langgraph.md)*
