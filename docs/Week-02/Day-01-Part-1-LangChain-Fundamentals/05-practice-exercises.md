# Practice Exercises — LangChain Fundamentals

---

## Exercise 1 — Multi-provider chatbot (Warm-up)

Build a chatbot that can switch between GPT-4o-mini and Claude Haiku mid-conversation based on a user command, without losing conversation history.

```python
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Initialize both models
gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
claude = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.7, api_key=os.getenv("ANTHROPIC_API_KEY"))

MODELS = {"gpt": gpt, "claude": claude}

def run_multimodel_chat():
    history = [SystemMessage(content="You are a helpful assistant. Be concise.")]
    current_model = "gpt"

    print("Multi-provider chatbot (type '/switch' to change model, 'quit' to exit)")
    print(f"Current model: {current_model}\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "/switch":
            current_model = "claude" if current_model == "gpt" else "gpt"
            print(f"[Switched to {current_model}]\n")
            continue

        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        response = MODELS[current_model].invoke(history)
        history.append(response)

        print(f"[{current_model.upper()}]: {response.content}\n")

# run_multimodel_chat()
print("Multi-provider chatbot ready. Uncomment run_multimodel_chat() to start.")
```

**Extension:** Add a `/summary` command that summarizes the conversation so far using whichever model is currently active.

---

## Exercise 2 — Document Q&A pipeline with LCEL (Main)

Build a complete document Q&A pipeline using LCEL. It should: ingest a list of text chunks, build a Chroma vector store, retrieve relevant chunks, and answer questions with citations.

```python
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# Sample corpus
DOCUMENTS = [
    {"id": "1", "text": "LangChain uses LCEL (LangChain Expression Language) for composing pipelines. The | operator chains Runnables together.", "source": "langchain_docs"},
    {"id": "2", "text": "LCEL supports streaming, async, and batch execution automatically for any chain built with the | operator.", "source": "langchain_docs"},
    {"id": "3", "text": "ChatPromptTemplate allows you to define reusable prompt templates with variables. Use from_messages() for chat models.", "source": "langchain_docs"},
    {"id": "4", "text": "Memory in LangChain allows conversation history to persist across multiple turns. Use InMemoryChatMessageHistory for development.", "source": "langchain_docs"},
    {"id": "5", "text": "RunnableParallel runs multiple chains concurrently and combines their outputs into a dictionary.", "source": "langchain_docs"},
    {"id": "6", "text": "LangSmith provides tracing and debugging for LangChain pipelines. Set LANGCHAIN_TRACING_V2=true to enable.", "source": "langchain_docs"},
]

# Build vector store
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

docs = [Document(page_content=d["text"], metadata={"id": d["id"], "source": d["source"]})
        for d in DOCUMENTS]
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Structured output with citations
class AnswerWithCitations(BaseModel):
    answer: str = Field(description="The answer to the question")
    citations: list[str] = Field(description="Source document IDs used")
    confidence: str = Field(description="high / medium / low")

structured_llm = llm.with_structured_output(AnswerWithCitations)

# Build the QA chain
def format_docs_with_ids(docs) -> str:
    return "\n\n".join(
        f"[{doc.metadata['id']}] {doc.page_content}"
        for doc in docs
    )

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer questions using ONLY the provided documents.
Cite the document IDs that support your answer.
If the answer is not in the documents, say so."""),
    ("human", "Documents:\n{context}\n\nQuestion: {question}")
])

qa_chain = (
    {
        "context": retriever | format_docs_with_ids,
        "question": RunnablePassthrough()
    }
    | qa_prompt
    | structured_llm
)

# Test
questions = [
    "What is LCEL and what operator does it use?",
    "How do I enable LangSmith tracing?",
    "Does LangChain support databases?",  # Should say not in documents
]

for q in questions:
    result = qa_chain.invoke(q)
    print(f"Q: {q}")
    print(f"A: {result.answer}")
    print(f"Citations: {result.citations}")
    print(f"Confidence: {result.confidence}\n")
```

---

## Exercise 3 — Multi-step content pipeline with parallel branches (Stretch)

Build a content pipeline that: (1) takes a blog topic, (2) in parallel generates a draft outline AND three Twitter thread ideas, (3) expands the outline into a full post, (4) outputs a final package.

```python
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
parser = StrOutputParser()

# Step 1: Parallel content generation
outline_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a technical content strategist."),
        ("human", "Create a 5-section blog post outline for: {topic}. Return numbered sections only.")
    ])
    | llm
    | parser
)

twitter_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a technical Twitter influencer."),
        ("human", "Create 3 different Twitter thread opening lines for the topic: {topic}. Number them 1-3.")
    ])
    | llm
    | parser
)

# Run outline and Twitter ideas in parallel
parallel_step = RunnableParallel(
    outline=outline_chain,
    twitter_ideas=twitter_chain,
    topic=RunnablePassthrough() | (lambda x: x["topic"])
)

# Step 2: Expand outline into post
expand_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a technical writer. Write engaging blog posts."),
        ("human", "Write a 400-word blog post following this outline:\n\n{outline}\n\nTopic: {topic}")
    ])
    | llm
    | parser
)

# Step 3: Combine everything
def build_content_package(parallel_result: dict) -> dict:
    return {
        "topic": parallel_result["topic"],
        "outline": parallel_result["outline"],
        "twitter_ideas": parallel_result["twitter_ideas"],
        "expand_input": {
            "outline": parallel_result["outline"],
            "topic": parallel_result["topic"]
        }
    }

def add_blog_post(package: dict) -> dict:
    blog_post = expand_chain.invoke(package["expand_input"])
    package["blog_post"] = blog_post
    return package

from langchain_core.runnables import RunnableLambda

full_pipeline = (
    parallel_step
    | RunnableLambda(build_content_package)
    | RunnableLambda(add_blog_post)
)

result = full_pipeline.invoke({"topic": "Why every ML engineer should learn LangChain"})

print(f"Topic: {result['topic']}\n")
print(f"Outline:\n{result['outline']}\n")
print(f"Twitter Ideas:\n{result['twitter_ideas']}\n")
print(f"Blog Post (first 300 chars):\n{result['blog_post'][:300]}...")
```

---

[[04-lcel]] | [[06-interview-questions]]
