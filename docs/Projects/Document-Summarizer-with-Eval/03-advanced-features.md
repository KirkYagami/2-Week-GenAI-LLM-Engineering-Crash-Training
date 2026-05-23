# Advanced Features — Document Summarizer

## LangSmith experiment tracking

Track summarization quality across different prompt versions and compare experiments:

```python
# langsmith_eval.py
import os
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

ls_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# Create a dataset of documents + reference summaries
def create_eval_dataset(examples: list[dict]) -> str:
    """examples: list of {input: str, reference: str}"""
    dataset = ls_client.create_dataset("summarizer-eval-v1")
    ls_client.create_examples(
        inputs=[{"text": e["input"]} for e in examples],
        outputs=[{"summary": e["reference"]} for e in examples],
        dataset_id=dataset.id,
    )
    return dataset.name

# Define an evaluator
def coverage_evaluator(run, example) -> dict:
    """Check that key facts from the reference appear in the generated summary."""
    generated = run.outputs.get("summary", "")
    reference = example.outputs.get("summary", "")
    reference_words = set(reference.lower().split())
    generated_words = set(generated.lower().split())
    # Simple word overlap as coverage proxy
    overlap = len(reference_words & generated_words) / len(reference_words) if reference_words else 0
    return {"key": "coverage", "score": overlap}

# Run evaluation
def run_langsmith_eval(dataset_name: str, prompt_version: str = "v1") -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following document in 2–3 concise sentences."),
        ("user", "{text}"),
    ])
    chain = prompt | llm | parser

    def target_fn(inputs: dict) -> dict:
        summary = chain.invoke(inputs)
        return {"summary": summary}

    results = evaluate(
        target_fn,
        data=dataset_name,
        evaluators=[coverage_evaluator],
        experiment_prefix=f"summarizer-{prompt_version}",
        client=ls_client,
    )
    return results
```

## Streaming chunk progress

For large documents, show users which chunk is being processed:

```python
@app.post("/summarize/stream")
async def summarize_stream(file: UploadFile = File(...), format_type: str = Form("paragraph")):
    content = await file.read()
    text = extract_text(content, file.filename)
    chunks = chunk_by_tokens(text, max_tokens=3500)

    async def progress_stream():
        import json
        yield f"data: {json.dumps({'event': 'start', 'total_chunks': len(chunks)})}\n\n"
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await summarize_chunk(chunk)
            chunk_summaries.append(summary)
            yield f"data: {json.dumps({'event': 'chunk_done', 'chunk': i+1, 'total': len(chunks)})}\n\n"

        yield f"data: {json.dumps({'event': 'reducing'})}\n\n"
        final = await reduce_summaries(chunk_summaries, format_type=format_type)
        yield f"data: {json.dumps({'event': 'done', 'summary': final})}\n\n"

    return StreamingResponse(
        progress_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

## Multi-level summarization (recursive)

For very long documents (100+ pages), summarize recursively:

```python
async def recursive_summarize(text: str, target_tokens: int = 500) -> str:
    """Summarize until the summary fits in target_tokens."""
    from tokens import count_tokens, chunk_by_tokens

    current_text = text
    iteration = 0
    while count_tokens(current_text) > target_tokens and iteration < 5:
        chunks = chunk_by_tokens(current_text, max_tokens=3500)
        summaries = await map_summaries(chunks)
        current_text = "\n\n".join(summaries)
        iteration += 1

    return current_text
```

---

[[02-implementation]] | [[04-evaluation]]
