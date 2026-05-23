# Setup and Environment — AI Writing Assistant

## Project structure

```
ai-writing-assistant/
├── app.py              ← FastAPI application
├── pipeline.py         ← LangChain writing pipeline
├── prompts.py          ← Prompt templates
├── eval.py             ← Output quality evaluation
├── requirements.txt
├── .env.example
└── Dockerfile
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```
# requirements.txt
openai==1.51.0
langchain==0.3.7
langchain-openai==0.2.3
langchain-core==0.3.15
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.0
python-dotenv==1.0.1
httpx==0.27.2
```

```bash
# .env.example
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=gpt-4o-mini
MAX_TOKENS_PER_STEP=800
```

## Quick smoke test

```python
# smoke_test.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write one sentence about Python."}],
    max_tokens=50,
)
print(resp.choices[0].message.content)
print("Setup OK")
```

```bash
python smoke_test.py
# Output: Python is a versatile, beginner-friendly programming language...
# Setup OK
```

---

[[README]] | [[02-implementation]]
