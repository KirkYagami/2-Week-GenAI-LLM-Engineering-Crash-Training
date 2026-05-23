# Chunking Strategies

Chunking is the most underrated part of RAG. The right chunk size and strategy determines whether your retrieval system finds the right information. The wrong strategy makes even perfect embeddings useless.

## Learning objectives

- Implement fixed-size, sentence, recursive, semantic, and hierarchical chunking
- Measure chunk quality with a token counter
- Choose a strategy based on document type and query pattern
- Handle edge cases: very short documents, tables, code blocks

---

## Fixed-size chunking

The simplest approach: split by character or token count with overlap.

```python
import os
import re
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def fixed_size_chunk(
    text: str,
    chunk_size: int = 512,    # tokens
    overlap: int = 64,         # tokens
    source: str = "doc"
) -> list[dict]:
    tokens = enc.encode(text)
    chunks = []

    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append({
            "id": f"{source}-{len(chunks)}",
            "text": chunk_text.strip(),
            "tokens": len(chunk_tokens),
            "source": source,
            "strategy": "fixed"
        })
        i += chunk_size - overlap

    return chunks

sample = """
Artificial intelligence is transforming the way businesses operate. Machine learning,
a core branch of AI, enables systems to learn and improve from experience without being
explicitly programmed. Deep learning, a subset of machine learning, uses neural networks
with many layers to model complex patterns in data.

Natural language processing (NLP) allows computers to understand, interpret, and generate
human language. Applications include sentiment analysis, machine translation, and chatbots.
Large language models like GPT-4o and Claude have pushed NLP capabilities dramatically forward.

Computer vision enables machines to interpret visual information from the world. Convolutional
neural networks (CNNs) excel at image classification, object detection, and segmentation.
Diffusion models now power state-of-the-art image generation.
""".strip()

chunks = fixed_size_chunk(sample, chunk_size=80, overlap=10)
for c in chunks:
    print(f"Chunk {c['id']}: {c['tokens']} tokens — {c['text'][:60]}...")
```

**Best for:** Generic documents, quick prototyping, when you don't know your document structure.
**Avoid for:** Documents with clear semantic boundaries (chapters, functions, paragraphs).

---

## Sentence-level chunking

Respects sentence boundaries — no chunk ends mid-sentence.

```python
import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

def sentence_chunk(
    text: str,
    max_tokens: int = 256,
    overlap_sentences: int = 1,
    source: str = "doc"
) -> list[dict]:
    sentences = sent_tokenize(text)
    chunks = []
    current_sentences = []
    current_tokens = 0

    for i, sent in enumerate(sentences):
        sent_tokens = count_tokens(sent)

        if current_tokens + sent_tokens > max_tokens and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "id": f"{source}-{len(chunks)}",
                "text": chunk_text.strip(),
                "tokens": count_tokens(chunk_text),
                "source": source,
                "strategy": "sentence"
            })
            # Keep overlap sentences
            current_sentences = current_sentences[-overlap_sentences:]
            current_tokens = sum(count_tokens(s) for s in current_sentences)

        current_sentences.append(sent)
        current_tokens += sent_tokens

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append({
            "id": f"{source}-{len(chunks)}",
            "text": chunk_text.strip(),
            "tokens": count_tokens(chunk_text),
            "source": source,
            "strategy": "sentence"
        })

    return chunks

chunks = sentence_chunk(sample, max_tokens=80, overlap_sentences=1)
for c in chunks:
    print(f"Chunk {c['id']}: {c['tokens']} tokens — {c['text'][:70]}...")
```

**Best for:** Articles, documentation, prose documents.
**Avoid for:** Code, structured data, documents with very long sentences.

---

## Recursive character chunking

Tries multiple separators in order, falling back to character splitting only when needed. This is what LangChain's `RecursiveCharacterTextSplitter` implements.

```python
def recursive_chunk(
    text: str,
    chunk_size: int = 500,   # characters
    overlap: int = 50,
    separators: list[str] | None = None,
    source: str = "doc"
) -> list[dict]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, sep_index: int) -> list[str]:
        if sep_index >= len(separators):
            return [text]

        separator = separators[sep_index]
        parts = text.split(separator) if separator else list(text)
        parts = [p for p in parts if p.strip()]

        chunks = []
        current = ""

        for part in parts:
            if len(current) + len(separator) + len(part) <= chunk_size:
                current = (current + separator + part).strip() if current else part
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    # Part itself is too long — recurse with next separator
                    chunks.extend(_split(part, sep_index + 1))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    raw_chunks = _split(text, 0)

    result = []
    for i, chunk_text in enumerate(raw_chunks):
        if chunk_text.strip():
            result.append({
                "id": f"{source}-{i}",
                "text": chunk_text.strip(),
                "tokens": count_tokens(chunk_text),
                "source": source,
                "strategy": "recursive"
            })
    return result

chunks = recursive_chunk(sample, chunk_size=300, overlap=30)
for c in chunks:
    print(f"Chunk {c['id']}: {c['tokens']} tokens — {c['text'][:60]}...")
```

**Best for:** Mixed documents, markdown, code + prose.
**Avoid for:** Documents where you need full semantic preservation.

---

## Semantic chunking

Groups sentences together based on embedding similarity — a new paragraph starts when the topic shifts significantly.

```python
from openai import OpenAI
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def semantic_chunk(
    text: str,
    breakpoint_threshold: float = 0.3,  # drop in similarity triggers new chunk
    min_chunk_tokens: int = 50,
    source: str = "doc"
) -> list[dict]:
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return [{"id": f"{source}-0", "text": text, "tokens": count_tokens(text), "source": source, "strategy": "semantic"}]

    # Embed each sentence
    resp = client.embeddings.create(model="text-embedding-3-small", input=sentences)
    embeddings = [item.embedding for item in resp.data]

    # Compute similarity between consecutive sentences
    similarities = [cosine_sim(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]

    # Find breakpoints where similarity drops sharply
    avg_sim = np.mean(similarities)
    breakpoints = {i+1 for i, sim in enumerate(similarities) if sim < avg_sim - breakpoint_threshold}

    # Build chunks
    chunks = []
    current_sentences = []
    current_tokens = 0

    for i, sent in enumerate(sentences):
        if i in breakpoints and current_tokens >= min_chunk_tokens and current_sentences:
            chunks.append({
                "id": f"{source}-{len(chunks)}",
                "text": " ".join(current_sentences),
                "tokens": current_tokens,
                "source": source,
                "strategy": "semantic"
            })
            current_sentences = []
            current_tokens = 0

        current_sentences.append(sent)
        current_tokens += count_tokens(sent)

    if current_sentences:
        chunks.append({
            "id": f"{source}-{len(chunks)}",
            "text": " ".join(current_sentences),
            "tokens": current_tokens,
            "source": source,
            "strategy": "semantic"
        })

    return chunks

# chunks = semantic_chunk(sample)
# for c in chunks:
#     print(f"Chunk {c['id']}: {c['tokens']} tokens — {c['text'][:70]}...")
```

**Best for:** Long documents with clear topic transitions (research papers, book chapters).
**Cost:** Requires embedding every sentence (more API calls). Use only when chunk quality is critical.

---

## Hierarchical (parent-child) chunking

Store large parent chunks for context and small child chunks for retrieval. Retrieve by child, return the parent.

```python
def hierarchical_chunk(
    text: str,
    parent_size: int = 1000,
    child_size: int = 200,
    child_overlap: int = 20,
    source: str = "doc"
) -> tuple[list[dict], list[dict]]:
    parents = fixed_size_chunk(text, chunk_size=parent_size, overlap=0, source=source)
    children = []

    for parent in parents:
        child_chunks = fixed_size_chunk(
            parent["text"],
            chunk_size=child_size,
            overlap=child_overlap,
            source=f"{parent['id']}"
        )
        for child in child_chunks:
            child["parent_id"] = parent["id"]
            child["parent_text"] = parent["text"]
        children.extend(child_chunks)

    return parents, children

parents, children = hierarchical_chunk(sample, parent_size=150, child_size=50, child_overlap=5)
print(f"Parents: {len(parents)}, Children: {len(children)}")
for child in children[:3]:
    print(f"  Child {child['id']}: '{child['text'][:50]}...' → Parent: {child['parent_id']}")

# At search time: embed + index children, retrieve children, return parent_text in prompt
```

**Best for:** When you want fine-grained retrieval but need enough context for generation.
**How it works:** Embed and search the small child chunks (high precision). When a child is retrieved, include the full parent chunk in the generation context (more context for the LLM).

---

## Chunking strategy comparison

| Strategy | Chunk boundaries | Semantic coherence | Cost | Best for |
|----------|-----------------|-------------------|------|----------|
| Fixed-size | Token count | Low | Very low | Prototyping |
| Sentence | Sentence end | Medium | Low | Prose documents |
| Recursive | Hierarchical separators | Medium-high | Low | Mixed content, markdown |
| Semantic | Topic shifts | High | High (API calls) | Research papers, books |
| Hierarchical | Multi-level | Highest | Medium | Long documents needing full-context generation |

> [!tip] Start with recursive chunking
> For most documents, recursive chunking with `chunk_size=512` and `overlap=64` tokens is the right starting point. It's cheap, fast, and respects natural document structure. Only switch to semantic chunking when you measure poor retrieval quality on recursive chunks.

---

## Handling special content

```python
def chunk_with_metadata(
    text: str,
    source: str,
    doc_type: str = "text"
) -> list[dict]:
    # Code blocks: chunk by function/class, not by token count
    if doc_type == "code":
        # Split on function/class boundaries
        blocks = re.split(r'\n(?=def |class )', text)
        return [
            {"id": f"{source}-{i}", "text": block.strip(), "tokens": count_tokens(block),
             "source": source, "doc_type": "code", "strategy": "code_boundary"}
            for i, block in enumerate(blocks) if block.strip()
        ]

    # Markdown: respect heading boundaries
    if doc_type == "markdown":
        sections = re.split(r'\n(?=#{1,3} )', text)
        return [
            {"id": f"{source}-{i}", "text": section.strip(), "tokens": count_tokens(section),
             "source": source, "doc_type": "markdown", "strategy": "heading"}
            for i, section in enumerate(sections) if section.strip()
        ]

    # Default: recursive character splitting
    return recursive_chunk(text, source=source)
```

---

[[01-what-is-rag]] | [[03-retrieval-and-augmentation]]
