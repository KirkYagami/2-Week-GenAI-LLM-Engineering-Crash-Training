# Practice Exercises — AI Agents

---

## Exercise 1 — ReAct agent with step tracing (Warm-up)

Build a ReAct agent and instrument it to output a clean step trace that shows Thought, Action, Observation for each step.

```python
import os
import re
from openai import OpenAI
from dataclasses import dataclass, field

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class Step:
    thought: str
    action_tool: str
    action_input: str
    observation: str

@dataclass
class AgentTrace:
    question: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    error: str = ""

# Tools
def search(query: str) -> str:
    """Search for factual information."""
    data = {
        "python release year": "Python was first released in 1991.",
        "python creator": "Python was created by Guido van Rossum.",
        "python version": "The latest stable Python version is 3.13 (released 2024).",
        "python popularity": "Python ranks #1 on the TIOBE index as of 2024.",
    }
    for key, val in data.items():
        if any(word in query.lower() for word in key.split()):
            return val
    return f"No specific data found for: {query}"

def calculator(expression: str) -> str:
    """Compute a mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

TOOLS = {"search": search, "calculator": calculator}

SYSTEM = """Solve problems step-by-step using:

Thought: [reasoning]
Action: tool_name[input]
Observation: [result]
... repeat ...
Final Answer: [answer]

Tools: search[query], calculator[expression]
Always start with Thought: Always end with Final Answer:"""

def react_agent_traced(question: str, max_steps: int = 8) -> AgentTrace:
    trace = AgentTrace(question=question)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stop=["Observation:"],
            temperature=0.0,
            max_tokens=200,
        )
        output = response.choices[0].message.content

        # Parse final answer
        final_match = re.search(r"Final Answer:\s*(.+)", output, re.DOTALL)
        if final_match:
            trace.final_answer = final_match.group(1).strip()
            return trace

        # Parse thought and action
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", output, re.DOTALL)
        action_match = re.search(r"Action:\s*(\w+)\[(.+?)\]", output)

        if not action_match:
            trace.error = f"Failed to parse action from: {output}"
            return trace

        thought = thought_match.group(1).strip() if thought_match else ""
        tool_name = action_match.group(1)
        tool_input = action_match.group(2)

        observation = TOOLS.get(tool_name, lambda x: f"Unknown tool: {tool_name}")(tool_input)

        trace.steps.append(Step(
            thought=thought,
            action_tool=tool_name,
            action_input=tool_input,
            observation=observation,
        ))

        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    trace.error = "Max steps reached"
    return trace

def print_trace(trace: AgentTrace) -> None:
    print(f"\nQ: {trace.question}")
    print("=" * 60)
    for i, step in enumerate(trace.steps, 1):
        print(f"\nStep {i}:")
        print(f"  Thought: {step.thought}")
        print(f"  Action:  {step.action_tool}[{step.action_input}]")
        print(f"  Obs:     {step.observation}")
    if trace.final_answer:
        print(f"\nFinal Answer: {trace.final_answer}")
    if trace.error:
        print(f"\nError: {trace.error}")
    print(f"\nTotal steps: {len(trace.steps)}")

# Run
for q in [
    "Who created Python and in what year?",
    "If Python was released in 1991, how many years ago is that from 2025?",
]:
    trace = react_agent_traced(q)
    print_trace(trace)
```

---

## Exercise 2 — Customer support agent (Main)

Build a customer support agent that classifies tickets, retrieves account data, and resolves issues — with step-level logging.

```python
import os
import json
from openai import OpenAI
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simulated databases
ACCOUNTS = {
    "ACC-1001": {"name": "Alice Chen", "plan": "Pro", "status": "active", "balance": 0.0},
    "ACC-1002": {"name": "Bob Smith", "plan": "Basic", "status": "suspended", "balance": -29.99},
    "ACC-1003": {"name": "Carol Jones", "plan": "Enterprise", "status": "active", "balance": 150.00},
}

ORDERS = {
    "ORD-5001": {"account": "ACC-1001", "item": "Standing Desk", "status": "shipped", "tracking": "TRK-123"},
    "ORD-5002": {"account": "ACC-1002", "item": "Chair", "status": "delivered", "date": "2025-05-10"},
}

# Tool implementations
def lookup_account(account_id: str) -> dict:
    """Look up customer account details by account ID."""
    return ACCOUNTS.get(account_id, {"error": f"Account {account_id} not found"})

def lookup_order(order_id: str) -> dict:
    """Get order status and tracking information."""
    return ORDERS.get(order_id, {"error": f"Order {order_id} not found"})

def update_account_status(account_id: str, new_status: str) -> dict:
    """Update an account's status (active, suspended, cancelled)."""
    if account_id not in ACCOUNTS:
        return {"error": f"Account {account_id} not found"}
    if new_status not in ["active", "suspended", "cancelled"]:
        return {"error": f"Invalid status: {new_status}"}
    ACCOUNTS[account_id]["status"] = new_status
    return {"success": True, "account_id": account_id, "new_status": new_status}

def apply_credit(account_id: str, amount: float, reason: str) -> dict:
    """Apply a credit to a customer's account balance."""
    if account_id not in ACCOUNTS:
        return {"error": f"Account {account_id} not found"}
    ACCOUNTS[account_id]["balance"] += amount
    return {"success": True, "account_id": account_id, "credit": amount,
            "new_balance": ACCOUNTS[account_id]["balance"], "reason": reason}

SUPPORT_TOOLS = [
    {"type": "function", "function": {"name": "lookup_account", "description": "Look up customer account details.",
        "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}}, "required": ["account_id"]}}},
    {"type": "function", "function": {"name": "lookup_order", "description": "Get order status and tracking.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "update_account_status", "description": "Update account status.",
        "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}, "new_status": {"type": "string", "enum": ["active", "suspended", "cancelled"]}}, "required": ["account_id", "new_status"]}}},
    {"type": "function", "function": {"name": "apply_credit", "description": "Apply account credit.",
        "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}, "amount": {"type": "number"}, "reason": {"type": "string"}}, "required": ["account_id", "amount", "reason"]}}},
]
TOOL_MAP = {"lookup_account": lookup_account, "lookup_order": lookup_order,
            "update_account_status": update_account_status, "apply_credit": apply_credit}

SUPPORT_SYSTEM = """You are a customer support agent with access to account and order data.
Resolve customer issues by looking up their information, then taking appropriate action.
Always verify account details before making changes. Be concise and helpful."""

def support_agent(ticket: str, max_steps: int = 8) -> str:
    messages = [{"role": "system", "content": SUPPORT_SYSTEM}, {"role": "user", "content": ticket}]
    steps = []

    for _ in range(max_steps):
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=SUPPORT_TOOLS)
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            print(f"  Completed in {len(steps)} steps: {[s['tool'] for s in steps]}")
            return choice.message.content

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = TOOL_MAP[tc.function.name](**args)
                steps.append({"tool": tc.function.name, "args": args})
                print(f"  → {tc.function.name}({args}) = {str(result)[:80]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return "Unable to resolve in the given steps."

# Test tickets
TICKETS = [
    "My account ACC-1002 was suspended. Can you reactivate it and apply a $30 credit for the inconvenience?",
    "I need the status of order ORD-5001.",
    "Can you check account ACC-1001 and tell me my current plan?",
]
for t in TICKETS:
    print(f"\nTicket: {t}")
    response = support_agent(t)
    print(f"Response: {response}")
```

---

[[05-memory-strategies]] | [[07-interview-questions]]
