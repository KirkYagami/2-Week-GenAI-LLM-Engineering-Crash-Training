# LangChain Cheat Sheet

Quick reference for LangChain LCEL (LangChain Expression Language) patterns.

---

## Basic chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"input": "What is Python?"})
# Async: result = await chain.ainvoke({"input": "..."})
# Stream: for chunk in chain.stream({"input": "..."}): print(chunk, end="")
```

## Structured output

```python
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str
    confidence: float

structured_chain = prompt | llm.with_structured_output(Sentiment)
result = structured_chain.invoke({"input": "I love this product!"})
# result.label, result.confidence
```

## RAG chain

```python
from langchain_core.runnables import RunnablePassthrough

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only this context:\n{context}"),
    ("user", "{question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the refund policy?")
```

## Memory (chat history)

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_with_history.invoke(
    {"input": "Hello!"},
    config={"configurable": {"session_id": "user-1"}},
)
```

## Parallel execution

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    sentiment=sentiment_chain,
    summary=summary_chain,
    topics=topics_chain,
)
results = parallel.invoke({"input": "Long document text..."})
# results["sentiment"], results["summary"], results["topics"]
```

## Common output parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,        # Raw string
    JsonOutputParser,       # JSON dict
    PydanticOutputParser,   # Pydantic model
    CommaSeparatedListOutputParser,  # ["item1", "item2"]
)
```

## ChatPromptTemplate patterns

```python
# From messages list
prompt = ChatPromptTemplate.from_messages([
    ("system", "Context: {context}"),
    ("human", "{question}"),
    ("ai", "{answer}"),     # For few-shot examples
    ("human", "{new_question}"),
])

# From template string
prompt = ChatPromptTemplate.from_template("Summarize: {text}")

# Partial (pre-fill some variables)
partial_prompt = prompt.partial(context="You are a Python expert.")
```
