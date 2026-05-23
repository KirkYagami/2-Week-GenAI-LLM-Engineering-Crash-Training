# Chains and Prompts

Chains are the core pattern in LangChain: compose a prompt template, an LLM, and an output parser into a reusable unit. Prompts are the inputs to that unit. Getting both right determines whether your application works as intended.

## Learning objectives

- Build few-shot prompt templates with dynamic examples
- Use output parsers for structured responses
- Compose sequential chains where one output feeds the next
- Debug chains by inspecting intermediate outputs

---

## ChatPromptTemplate patterns

```python
import os
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

# Basic template with variables
basic_template = ChatPromptTemplate.from_messages([
    ("system", "You are a customer support agent for {company}. Be {tone}."),
    ("human", "{user_message}")
])

chain = basic_template | llm | StrOutputParser()

response = chain.invoke({
    "company": "Acme Corp",
    "tone": "professional but friendly",
    "user_message": "My order hasn't arrived after 2 weeks."
})
print(response)
```

---

## Few-shot prompt templates

Few-shot examples dramatically improve model consistency for structured tasks.

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

# Define the example format
example_template = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# Define the examples
examples = [
    {
        "input": "The delivery took 3 weeks and the package was damaged.",
        "output": '{"sentiment": "negative", "category": "shipping", "priority": "high", "action": "refund_or_replace"}'
    },
    {
        "input": "The product is exactly as described, very happy!",
        "output": '{"sentiment": "positive", "category": "product_quality", "priority": "low", "action": "thank_customer"}'
    },
    {
        "input": "I can\'t figure out how to set up the app.",
        "output": '{"sentiment": "neutral", "category": "technical_support", "priority": "medium", "action": "send_documentation"}'
    }
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_template
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify customer feedback as JSON. Match the format of the examples exactly."),
    few_shot_prompt,
    ("human", "{input}")
])

from langchain_core.output_parsers import JsonOutputParser

classify_chain = final_prompt | llm | JsonOutputParser()

result = classify_chain.invoke({"input": "The website keeps timing out when I try to checkout."})
print(result)
# {'sentiment': 'negative', 'category': 'technical_support', 'priority': 'high', 'action': '...'}
```

---

## Structured output with Pydantic

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

class ContentPlan(BaseModel):
    title: str = Field(description="Blog post title")
    outline: list[str] = Field(description="3-5 section headings")
    target_audience: str = Field(description="Who this post is for")
    estimated_words: int = Field(description="Approximate word count")
    seo_keywords: list[str] = Field(description="Top 3 SEO keywords")

# Use with_structured_output for Pydantic models
structured_llm = llm.with_structured_output(ContentPlan)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a content strategist. Create detailed content plans."),
    ("human", "Create a content plan for a blog post about: {topic}")
])

plan_chain = prompt | structured_llm

plan = plan_chain.invoke({"topic": "Getting started with LLM evaluation"})
print(f"Title: {plan.title}")
print(f"Audience: {plan.target_audience}")
print(f"Sections: {plan.outline}")
print(f"Keywords: {plan.seo_keywords}")
```

---

## Sequential chains (pipeline pattern)

Chain the output of one step as input to the next.

```python
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))
parser = StrOutputParser()

# Step 1: Generate a blog outline
outline_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical writer. Create a concise outline only."),
    ("human", "Create a 5-section outline for a blog post about: {topic}")
])

# Step 2: Write the introduction based on the outline
intro_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical writer. Write engaging introductions."),
    ("human", "Given this outline:\n\n{outline}\n\nWrite a compelling 3-paragraph introduction.")
])

# Compose: topic → outline → introduction
outline_chain = outline_prompt | llm | parser
intro_chain = intro_prompt | llm | parser

# Sequential pipeline using a lambda to pass the outline
full_chain = outline_chain | (lambda outline: {"outline": outline}) | intro_chain

result = full_chain.invoke({"topic": "RAG vs fine-tuning for enterprise AI"})
print(result)
```

---

## Parallel chains with RunnableParallel

Run multiple chains at the same time and combine their outputs.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
parser = StrOutputParser()

# Two chains run in parallel
pros_prompt = ChatPromptTemplate.from_messages([
    ("human", "List 3 key advantages of {technology} in bullet points. Be concise.")
])
cons_prompt = ChatPromptTemplate.from_messages([
    ("human", "List 3 key disadvantages of {technology} in bullet points. Be concise.")
])

parallel_chain = RunnableParallel(
    pros=pros_prompt | llm | parser,
    cons=cons_prompt | llm | parser,
    technology=RunnablePassthrough()   # Pass input through unchanged
)

result = parallel_chain.invoke({"technology": "vector databases"})
print(f"Technology: {result['technology']}")
print(f"\nPros:\n{result['pros']}")
print(f"\nCons:\n{result['cons']}")
```

> [!tip] Use RunnableParallel for independent steps
> Any two chain steps that don't depend on each other should run in parallel. `RunnableParallel` handles async execution automatically — both LLM calls happen concurrently, cutting wall-clock time roughly in half.

---

## Debugging chains

```python
from langchain_core.runnables import RunnableLambda

def debug_step(name: str):
    """Middleware that prints the value at a pipeline stage."""
    def _print(x):
        print(f"\n[DEBUG {name}] Type: {type(x).__name__}")
        content = str(x)[:200]
        print(f"[DEBUG {name}] Value: {content}")
        return x
    return RunnableLambda(_print)

# Insert debug steps between chain components
debug_chain = (
    outline_prompt
    | debug_step("after_prompt")
    | llm
    | debug_step("after_llm")
    | parser
    | debug_step("final_output")
)

# result = debug_chain.invoke({"topic": "transformer architecture"})
```

---

[[01-langchain-overview]] | [[03-memory]]
