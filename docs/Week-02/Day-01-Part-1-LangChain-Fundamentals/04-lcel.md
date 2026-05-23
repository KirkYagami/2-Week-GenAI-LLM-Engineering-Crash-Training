# LangChain Expression Language (LCEL)

LCEL is LangChain's composition syntax. It uses Python's `|` operator to chain components into pipelines — the same operator used for Unix pipes. The result is readable, composable, and supports streaming, async, and batch execution without any extra code.

## Learning objectives

- Build LCEL pipelines using `|` composition
- Understand the `Runnable` interface and its key methods
- Use `RunnableParallel`, `RunnableLambda`, and `RunnablePassthrough`
- Stream, batch, and run pipelines asynchronously

---

## The Runnable interface

Everything in LCEL is a `Runnable`. A `Runnable` is anything with these methods:

```python
runnable.invoke(input)          # Single call — returns output
runnable.stream(input)          # Generator of output chunks
runnable.batch([input1, ...])   # Multiple inputs in parallel
await runnable.ainvoke(input)   # Async single call
await runnable.astream(input)   # Async streaming
await runnable.abatch([...])    # Async batch
```

When you use `|`, LangChain wraps the result in a `RunnableSequence` that implements all these methods automatically.

---

## Basic composition

```python
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

# The | operator creates a RunnableSequence
chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {domain}."),
        ("human", "{question}")
    ])
    | llm
    | StrOutputParser()
)

# invoke
result = chain.invoke({"domain": "machine learning", "question": "What is overfitting?"})
print(result)

# batch — runs multiple inputs concurrently
results = chain.batch([
    {"domain": "machine learning", "question": "What is overfitting?"},
    {"domain": "databases", "question": "What is an index?"},
    {"domain": "networking", "question": "What is TCP?"},
])
for r in results:
    print(r[:100])
```

---

## Streaming

```python
# Stream tokens as they arrive
chain = (
    ChatPromptTemplate.from_messages([("human", "Explain {topic} in 3 paragraphs.")])
    | llm
    | StrOutputParser()
)

print("Streaming: ", end="")
for chunk in chain.stream({"topic": "attention mechanisms"}):
    print(chunk, end="", flush=True)
print()
```

---

## RunnablePassthrough — passing input through

Use `RunnablePassthrough` to pass the original input alongside a transformed version.

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

# Pattern: retrieval chain where we pass both context and question
def fake_retriever(query: str) -> str:
    """Simulates a retriever that returns context."""
    return f"Context: Python was created by Guido van Rossum in 1991."

# Pass-through the question while also retrieving context
rag_chain = (
    RunnableParallel(
        context=RunnablePassthrough() | (lambda q: fake_retriever(q["question"])),
        question=RunnablePassthrough() | (lambda q: q["question"])
    )
    | ChatPromptTemplate.from_template(
        "Answer based only on context.\n\nContext: {context}\n\nQuestion: {question}"
    )
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke({"question": "Who created Python?"})
print(result)
```

---

## RunnableLambda — wrapping any function

Any Python function can become a `Runnable` with `RunnableLambda`.

```python
from langchain_core.runnables import RunnableLambda

def clean_text(text: str) -> str:
    """Preprocessing step — strip whitespace and normalize."""
    return " ".join(text.split())

def add_metadata(result: str) -> dict:
    """Post-processing — wrap result in structured output."""
    return {"answer": result, "word_count": len(result.split()), "source": "gpt-4o-mini"}

chain_with_pre_post = (
    RunnableLambda(clean_text)
    | ChatPromptTemplate.from_messages([("human", "{question}")])  # This won't work directly
    # ^ RunnableLambda integrates into pipelines but the prompt expects a dict
)

# Better pattern: use lambda to transform between steps
chain = (
    ChatPromptTemplate.from_messages([("human", "Summarize: {text}")])
    | llm
    | StrOutputParser()
    | RunnableLambda(add_metadata)
)

result = chain.invoke({"text": "  Python is a high-level programming language.   "})
print(result)
# {'answer': 'Python is a high-level programming language.', 'word_count': 7, 'source': 'gpt-4o-mini'}
```

---

## Async pipelines

```python
import os
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))

chain = (
    ChatPromptTemplate.from_messages([("human", "Define {term} in one sentence.")])
    | llm
    | StrOutputParser()
)

async def run_many(terms: list[str]) -> list[str]:
    """Run many chain calls concurrently."""
    tasks = [chain.ainvoke({"term": term}) for term in terms]
    return await asyncio.gather(*tasks)

# Run concurrently — much faster than sequential
terms = ["embedding", "tokenization", "attention", "quantization", "RLHF"]
# results = asyncio.run(run_many(terms))
# All 5 calls happen simultaneously
```

---

## LCEL with a retriever (RAG pattern)

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

# Build a small vector store
texts = [
    "Python was created by Guido van Rossum and first released in 1991.",
    "RAG combines retrieval with language model generation.",
    "LangChain uses LCEL for composing pipelines with the | operator.",
]
vectorstore = Chroma.from_texts(texts, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | ChatPromptTemplate.from_template(
        "Answer based only on the context.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("What is LCEL?")
print(result)
```

> [!success] LCEL is the right way
> In LangChain 0.3+, all new code should use LCEL. The old `LLMChain`, `SequentialChain`, and `StuffDocumentsChain` classes still exist for backward compatibility but are no longer actively developed. LCEL chains support streaming, async, and tracing with LangSmith automatically.

---

[[03-memory]] | [[05-practice-exercises]]
