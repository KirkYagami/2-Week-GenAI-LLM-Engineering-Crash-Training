# Agenda — Advanced RAG

Basic RAG — chunk, embed, retrieve, generate — gets you to 70% quality. The remaining 30% comes from addressing specific failure modes: the retrieved chunks are irrelevant, the query doesn't match the embedding space of the documents, or the context window is filled with partially-relevant text. Advanced RAG techniques fix each of these.

## Learning objectives

By the end of this session you will be able to:

- Apply cross-encoder reranking to improve retrieval precision
- Use HyDE to bridge the query-document embedding gap
- Implement multi-query retrieval for better recall on complex questions
- Apply contextual compression to reduce noise in retrieved context

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:30 | Reranking — cross-encoders and Cohere Rerank | [[01-reranking]] |
| 0:30 – 1:00 | HyDE — hypothetical document embeddings | [[02-hyde]] |
| 1:00 – 1:30 | Multi-query retrieval | [[03-multi-query]] |
| 1:30 – 2:00 | Contextual compression | [[04-contextual-compression]] |
| 2:00 – 2:30 | Advanced RAG patterns combined | [[05-advanced-rag-patterns]] |
| 2:30 – 3:00 | Practice exercises | [[06-practice-exercises]] |

## Setup

```bash
pip install langchain langchain-openai langchain-cohere sentence-transformers
```

> [!tip] When to apply advanced RAG
> Each technique adds latency and complexity. Apply them when you have evidence of the specific failure they fix: low faithfulness → better grounding prompt; low context precision → reranking; low context recall → multi-query or HyDE. Don't add advanced techniques preemptively.

[[../Day-01-Part-1-LangChain-Fundamentals/06-interview-questions|← LangChain Fundamentals]] | [[01-reranking|Start →]]
