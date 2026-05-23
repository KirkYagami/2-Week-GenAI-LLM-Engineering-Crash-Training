# Hugging Face Spaces

Spaces is Hugging Face's platform for hosting ML demo apps. You write a Gradio or Streamlit app, push it to a Space repository, and it runs on free CPU hardware (or a GPU for $0.40–$2/hour). No Docker, no Kubernetes, no infrastructure — the app is live in minutes.

## Learning objectives

- Build a Gradio app that calls a model via the Inference API
- Deploy to a Space using Git or the SDK
- Use the Spaces API to call any running demo programmatically
- Understand hardware tiers and when to upgrade

---

## Your first Gradio app

A Gradio app needs one file — `app.py`. The `gr.Interface` takes a Python function, defines input/output widgets, and generates the full UI automatically.

```python
# app.py — deploy this to a Space
import os
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(token=os.getenv("HF_TOKEN"))

def classify_customer_query(query: str) -> dict:
    """Classify a customer support query into a department."""
    result = client.zero_shot_classification(
        text=query,
        labels=["billing", "technical support", "account management", "general inquiry"],
        model="facebook/bart-large-mnli"
    )
    return {label: float(score) for label, score in zip(result["labels"], result["scores"])}

demo = gr.Interface(
    fn=classify_customer_query,
    inputs=gr.Textbox(
        label="Customer query",
        placeholder="Type a support question...",
        lines=3
    ),
    outputs=gr.Label(
        label="Department routing",
        num_top_classes=4
    ),
    title="Customer Query Router",
    description="Routes customer support queries to the right department using zero-shot classification.",
    examples=[
        ["I was charged twice for my subscription."],
        ["My app crashes when I try to log in."],
        ["How do I change my password?"],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
```

Run locally with `python app.py` — opens at `http://localhost:7860`.

---

## Multi-component chat interface

Gradio's `gr.Blocks` API gives full layout control. Use `gr.ChatInterface` for conversational apps.

```python
# app.py — streaming chat with history
import os
import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=os.getenv("HF_TOKEN")
)

def chat(message: str, history: list[list[str]]) -> str:
    """Generate a response given a message and conversation history."""
    # Build messages list from Gradio history format
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    # Format with Mistral chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = ""
    for token in client.text_generation(
        prompt=prompt,
        max_new_tokens=300,
        temperature=0.7,
        stream=True
    ):
        response += token
        yield response

demo = gr.ChatInterface(
    fn=chat,
    title="Open-Source Chat",
    description="Powered by Mistral 7B via Hugging Face Inference API",
    examples=["Explain RAG in simple terms.", "Write a Python function to reverse a string."],
    type="messages"
)

if __name__ == "__main__":
    demo.launch()
```

---

## Deploying to a Space

**Option 1 — Git push (recommended)**

```bash
# Create the Space on the Hub (or via web UI)
huggingface-cli repo create my-demo --type space --space_sdk gradio

# Clone, add files, push
git clone https://huggingface.co/spaces/your-username/my-demo
cd my-demo

# Add your app.py and requirements.txt
cat > requirements.txt << EOF
gradio>=4.0
huggingface_hub>=0.20
transformers>=4.40
EOF

git add app.py requirements.txt
git commit -m "add initial demo"
git push
# Space builds automatically; live at https://huggingface.co/spaces/your-username/my-demo
```

**Option 2 — Python SDK**

```python
from huggingface_hub import HfApi, create_repo

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create the Space
create_repo(
    repo_id="your-username/my-demo",
    repo_type="space",
    space_sdk="gradio",
    private=False,
    exist_ok=True
)

# Upload files
api.upload_file(path_or_fileobj="app.py", path_in_repo="app.py",
                repo_id="your-username/my-demo", repo_type="space")
api.upload_file(path_or_fileobj="requirements.txt", path_in_repo="requirements.txt",
                repo_id="your-username/my-demo", repo_type="space")
```

---

## Setting Space secrets

API keys should never be in your app code. Use Space secrets.

```python
from huggingface_hub import add_space_secret

add_space_secret(
    repo_id="your-username/my-demo",
    key="HF_TOKEN",
    value=os.getenv("HF_TOKEN")
)
# The secret is now available as os.environ["HF_TOKEN"] in the Space runtime
```

Via the web UI: Space settings → Variables and secrets → New secret.

---

## Calling any Space via the API

Every Gradio Space exposes a REST API. You can call it from other applications.

```python
from gradio_client import Client

# Connect to any public Space
space_client = Client("your-username/my-demo")

# Call the classify function
result = space_client.predict(
    "My subscription was charged twice",
    api_name="/predict"
)
print(result)

# List available API endpoints
space_client.view_api()
```

---

## Space hardware tiers

| Hardware | VRAM | Use case | Cost/hour |
|----------|------|----------|-----------|
| CPU Basic | — | Lightweight models, demo UI | Free |
| CPU Upgrade | — | Faster CPU, more RAM | $0.03 |
| T4 Small | 16 GB | 7B models in 4-bit | $0.40 |
| T4 Medium | 16 GB | 7B models in float16 | $0.60 |
| A10G Small | 24 GB | 13B models | $1.05 |
| A100 Large | 80 GB | 70B models in float16 | $4.13 |
| ZeroGPU | Shared A100 | Free, 60s time limit per call | Free |

> [!tip] ZeroGPU for prototyping
> ZeroGPU allocates a shared A100 slice for each request (up to 60 seconds). It's free for Hugging Face Pro subscribers. For model demos that take < 30 seconds per inference, ZeroGPU is the best free option for GPU-requiring models.

---

[[03-inference-api]] | [[05-datasets-and-models]]
