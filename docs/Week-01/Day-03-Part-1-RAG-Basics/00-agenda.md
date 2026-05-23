# Agenda — RAG Basics

Retrieval-Augmented Generation solves the most common failure mode of LLMs in production: hallucination about facts they weren't trained on. Today you build a working RAG system from scratch and understand every design decision.

## Learning objectives

By the end of this session you will be able to:

- Explain the RAG architecture and why it reduces hallucination
- Implement five different chunking strategies and choose between them
- Build a complete retrieval + augmentation + generation pipeline
- Decide when RAG is better than fine-tuning (and when it isn't)

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:20 | What is RAG? — architecture, benefits, limitations | [[01-what-is-rag]] |
| 0:20 – 0:55 | Chunking strategies — fixed, semantic, recursive, hierarchical | [[02-chunking-strategies]] |
| 0:55 – 1:25 | Retrieval and augmentation — context assembly, prompt design | [[03-retrieval-and-augmentation]] |
| 1:25 – 2:00 | RAG pipeline end-to-end — PDF ingestion to grounded answers | [[04-rag-pipeline-end-to-end]] |
| 2:00 – 2:30 | RAG vs. fine-tuning — decision framework | [[05-rag-vs-fine-tuning]] |
| 2:30 – 3:00 | Practice exercises | [[06-practice-exercises]] |

## Setup

```bash
pip install openai anthropic chromadb sentence-transformers pypdf2 tiktoken
```

```python
import os
from openai import OpenAI
from anthropic import Anthropic
import chromadb

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
chroma_client = chromadb.Client()
print("Ready!")
```

[[../Day-02-Part-2-Embeddings-and-Semantic-Search/07-interview-questions|← Day 02 Part 2]] | [[01-what-is-rag|Start →]]
