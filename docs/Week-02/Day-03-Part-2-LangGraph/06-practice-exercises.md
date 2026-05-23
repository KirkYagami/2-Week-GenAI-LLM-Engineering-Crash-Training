# Practice Exercises — LangGraph

---

## Exercise 1 — Sequential pipeline with streaming (Warm-up)

Build a three-node sequential graph that streams its state updates so you can watch each node complete in real time.

```python
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))

class PipelineState(TypedDict):
    topic: str
    keywords: list[str]
    outline: str
    article: str

def extract_keywords(state: PipelineState) -> dict:
    resp = llm.invoke([
        SystemMessage(content="Extract 5 key terms. Return as comma-separated list only."),
        HumanMessage(content=f"Topic: {state['topic']}")
    ])
    keywords = [k.strip() for k in resp.content.split(",")][:5]
    return {"keywords": keywords}

def generate_outline(state: PipelineState) -> dict:
    kw_str = ", ".join(state["keywords"])
    resp = llm.invoke([
        SystemMessage(content="Create a 3-section outline. Use: '1. Title\\n2. Title\\n3. Title' format."),
        HumanMessage(content=f"Topic: {state['topic']} | Keywords: {kw_str}")
    ])
    return {"outline": resp.content}

def write_article(state: PipelineState) -> dict:
    resp = llm.invoke([
        SystemMessage(content="Write a concise 2-paragraph article following this outline."),
        HumanMessage(content=f"Outline:\n{state['outline']}\n\nKeywords to include: {', '.join(state['keywords'])}")
    ])
    return {"article": resp.content}

graph = StateGraph(PipelineState)
graph.add_node("keywords", extract_keywords)
graph.add_node("outline", generate_outline)
graph.add_node("article", write_article)

graph.add_edge(START, "keywords")
graph.add_edge("keywords", "outline")
graph.add_edge("outline", "article")
graph.add_edge("article", END)

app = graph.compile()

topic = "The rise of open-source large language models"
print(f"Running pipeline for: '{topic}'\n")

# Stream to see each node complete in real time
for chunk in app.stream({"topic": topic, "keywords": [], "outline": "", "article": ""}):
    for node_name, update in chunk.items():
        print(f"✓ Node '{node_name}' completed:")
        for key, val in update.items():
            if isinstance(val, list):
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: {str(val)[:100]}...")
        print()
```

---

## Exercise 2 — Retry loop with quality gate (Main)

Build a code-generation graph that generates code, tests it (syntactically), and retries if there are errors.

```python
import os
import ast
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))

class CodeState(TypedDict):
    requirement: str
    code: str
    test_output: str
    passed: bool
    error_message: str
    attempts: int

def code_generator(state: CodeState) -> dict:
    """Generate Python code for the requirement."""
    messages = [
        SystemMessage(content="Write clean, working Python code. Return ONLY the code, no explanation."),
    ]
    if state.get("error_message"):
        messages.append(SystemMessage(content=f"Previous attempt had this error: {state['error_message']}. Fix it."))
        messages.append(SystemMessage(content=f"Previous code:\n{state['code']}"))

    messages.append(HumanMessage(content=state["requirement"]))
    resp = llm.invoke(messages)

    # Strip markdown code blocks if present
    code = resp.content
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    return {"code": code.strip(), "attempts": state.get("attempts", 0) + 1}

def syntax_tester(state: CodeState) -> dict:
    """Test code for syntax errors using ast.parse."""
    try:
        ast.parse(state["code"])
        return {"passed": True, "error_message": "", "test_output": "Syntax OK"}
    except SyntaxError as e:
        return {"passed": False, "error_message": f"SyntaxError at line {e.lineno}: {e.msg}", "test_output": str(e)}

def logic_tester(state: CodeState) -> dict:
    """Ask the LLM to review the code logic."""
    resp = llm.invoke([
        SystemMessage(content="""Review this code for logical errors.
Reply JSON: {"logic_ok": true/false, "issue": "description or null"}"""),
        HumanMessage(content=f"Requirement: {state['requirement']}\nCode:\n{state['code']}")
    ])
    import json, re
    try:
        match = re.search(r'\{.*\}', resp.content, re.DOTALL)
        data = json.loads(match.group()) if match else {"logic_ok": True}
        logic_ok = bool(data.get("logic_ok", True))
        issue = data.get("issue") or ""
        return {
            "passed": logic_ok,
            "error_message": issue if not logic_ok else "",
            "test_output": f"Logic {'OK' if logic_ok else 'FAILED: ' + issue}"
        }
    except Exception:
        return {"passed": True, "error_message": "", "test_output": "Logic check skipped"}

def route_after_syntax(state: CodeState) -> Literal["logic_test", "fix"]:
    return "logic_test" if state["passed"] else "fix"

def route_after_logic(state: CodeState) -> Literal["done", "fix", "give_up"]:
    if state["passed"]:
        return "done"
    if state["attempts"] < 3:
        return "fix"
    return "give_up"

graph = StateGraph(CodeState)
graph.add_node("generate", code_generator)
graph.add_node("syntax_test", syntax_tester)
graph.add_node("logic_test", logic_tester)

graph.add_edge(START, "generate")
graph.add_edge("generate", "syntax_test")
graph.add_conditional_edges("syntax_test", route_after_syntax, {"logic_test": "logic_test", "fix": "generate"})
graph.add_conditional_edges("logic_test", route_after_logic, {"done": END, "fix": "generate", "give_up": END})

app = graph.compile()

REQUIREMENTS = [
    "Write a function called 'fibonacci' that returns the nth Fibonacci number.",
    "Write a function called 'flatten_list' that flattens a nested list.",
]

for req in REQUIREMENTS:
    print(f"\nRequirement: {req}")
    result = app.invoke({
        "requirement": req, "code": "", "test_output": "",
        "passed": False, "error_message": "", "attempts": 0
    })
    print(f"Attempts: {result['attempts']} | Passed: {result['passed']}")
    print(f"Test output: {result['test_output']}")
    print(f"Code:\n{result['code']}\n")
```

---

[[05-multi-agent-orchestration]] | [[07-interview-questions]]
