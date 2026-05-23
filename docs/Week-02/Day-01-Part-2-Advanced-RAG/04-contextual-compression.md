# Contextual Compression

Retrieved chunks often contain partially relevant information — the document has the answer, but it's buried in surrounding context that's not relevant to this specific question. Contextual compression extracts only the relevant portion from each chunk before passing it to the LLM, reducing noise and token cost.

## Learning objectives

- Implement LLM-based contextual compression
- Use LangChain's `ContextualCompressionRetriever`
- Apply embeddings-based filtering as a cheap compression alternative
- Measure context quality before and after compression

---

## The problem: noisy context

```
Retrieved chunk (from a 500-word document):
"Our company was founded in 2010 with a focus on enterprise software.
We have offices in New York, London, and Tokyo. Our team includes 500 employees.
Refunds are accepted within 30 days for unused products in original packaging.
You can request a refund by contacting support@company.com or calling 1-800-HELP.
Our products include CRM, ERP, and workflow automation tools."

Question: "What is your refund policy?"

Relevant portion:
"Refunds are accepted within 30 days for unused products in original packaging.
Request via support@company.com or 1-800-HELP."

The irrelevant parts (founding, offices, employees, products) waste context window space
and can distract the LLM.
```

---

## LLM-based contextual compression

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COMPRESSION_PROMPT = """Extract the relevant portion of the following document for answering the given question.
- Extract only text that directly helps answer the question.
- Keep the extracted text verbatim (do not paraphrase).
- If nothing in the document is relevant, respond with: "NOT RELEVANT"
- Return only the extracted text, no explanation.

Question: {question}

Document:
{document}

Relevant extract:"""

def compress_chunk(question: str, document: str) -> str | None:
    """
    Extract only the relevant portion of a document for a given question.
    Returns None if the document is not relevant.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": COMPRESSION_PROMPT.format(question=question, document=document)
        }],
        temperature=0.0,
        max_tokens=300
    )
    result = response.choices[0].message.content.strip()

    if result == "NOT RELEVANT" or len(result) < 10:
        return None
    return result

def compress_retrieved_chunks(
    question: str,
    chunks: list[str],
    min_length: int = 20
) -> list[str]:
    """Compress all retrieved chunks, removing irrelevant ones."""
    compressed = []
    for chunk in chunks:
        result = compress_chunk(question, chunk)
        if result and len(result) >= min_length:
            compressed.append(result)

    return compressed

# Test
question = "What is your refund policy?"

CHUNKS = [
    """Our company was founded in 2010 with a focus on enterprise software.
We have offices in New York, London, and Tokyo. Our team includes 500 employees.
Refunds are accepted within 30 days for unused products in original packaging.
You can request a refund by contacting support@company.com or calling 1-800-HELP.
Our products include CRM, ERP, and workflow automation tools.""",

    """The integration API supports REST and GraphQL endpoints.
Authentication requires a Bearer token in the Authorization header.
Rate limits apply: 1000 requests per minute for standard tier.""",

    """For subscription cancellations, go to Settings > Billing > Cancel Subscription.
Your subscription will remain active until the end of the billing period.
Prorated refunds are available for annual plans canceled within the first 30 days.""",
]

print(f"Question: {question}\n")
print("=" * 60)

original_tokens = sum(len(c.split()) for c in CHUNKS)
compressed_chunks = compress_retrieved_chunks(question, CHUNKS)
compressed_tokens = sum(len(c.split()) for c in compressed_chunks)

print(f"Original: {len(CHUNKS)} chunks, ~{original_tokens} tokens")
print(f"Compressed: {len(compressed_chunks)} chunks, ~{compressed_tokens} tokens")
print(f"Reduction: {(1 - compressed_tokens/original_tokens)*100:.0f}%\n")

for i, chunk in enumerate(compressed_chunks, 1):
    print(f"Compressed chunk {i}: {chunk}")
```

---

## LangChain ContextualCompressionRetriever

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

# Build vector store
texts = [
    "Our company was founded in 2010. We have 500 employees. Refunds accepted within 30 days.",
    "The API supports REST and GraphQL. Auth uses Bearer tokens. Rate limit: 1000 req/min.",
    "For subscription cancellations: Settings > Billing > Cancel. Prorated refunds available.",
]

vectorstore = Chroma.from_texts(texts, embedding=embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Wrap with contextual compression
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Retrieve with compression
docs = compression_retriever.invoke("What is your refund policy?")
print(f"Retrieved {len(docs)} compressed documents:")
for doc in docs:
    print(f"  → {doc.page_content}")
```

---

## Embeddings-based filtering (cheaper alternative)

Instead of using an LLM to compress, filter chunks that are semantically similar to the question.

```python
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([item.embedding for item in resp.data])

def filter_by_similarity(
    question: str,
    chunks: list[str],
    min_similarity: float = 0.45
) -> list[tuple[str, float]]:
    """
    Filter chunks below similarity threshold.
    Fast alternative to LLM-based compression — no additional LLM call.
    """
    all_texts = [question] + chunks
    embs = embed(all_texts)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    q_emb = embs[0]
    chunk_embs = embs[1:]
    scores = chunk_embs @ q_emb

    return [
        (chunk, float(score))
        for chunk, score in zip(chunks, scores)
        if float(score) >= min_similarity
    ]

# Test
relevant = filter_by_similarity(question, CHUNKS, min_similarity=0.45)
print(f"Kept {len(relevant)}/{len(CHUNKS)} chunks above similarity threshold:")
for chunk, score in relevant:
    print(f"  [{score:.3f}] {chunk[:80]}...")
```

| Compression type | Quality | Latency | Cost | Use when |
|-----------------|---------|---------|------|----------|
| LLM extraction | Highest | +500ms | +$0.002/chunk | Context quality is critical |
| Similarity filter | Good | +5ms | Negligible | Fast pre-filter |
| No compression | N/A | 0ms | 0 | Short, focused documents |

> [!success] Use compression when chunks are large
> Contextual compression gives the biggest benefit when your chunks are large (500–1000 words) and contain diverse content. For small, focused chunks (100–200 words), the overhead of an additional LLM call isn't justified. Benchmark on your corpus before adding it to your pipeline.

---

[[03-multi-query]] | [[05-advanced-rag-patterns]]
