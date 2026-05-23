# Hugging Face Hub

The Hugging Face Hub is the GitHub of machine learning — a central repository for models, datasets, Spaces demos, and documentation. Unlike GitHub, every model and dataset is versioned, downloadable, and runnable with a single line of Python.

## Learning objectives

- Search and filter models by task, license, and hardware requirements
- Download model weights using `huggingface_hub`
- Programmatically list and inspect model metadata
- Upload your own model or dataset to the Hub

---

## Hub structure

```
huggingface.co/
├── models/          ← 500K+ model repositories
│   ├── meta-llama/Llama-3.1-8B-Instruct
│   ├── mistralai/Mistral-7B-Instruct-v0.3
│   ├── BAAI/bge-large-en-v1.5
│   └── ...
├── datasets/        ← 200K+ dataset repositories
│   ├── squad
│   ├── openai/gsm8k
│   └── ...
└── spaces/          ← Hosted demo apps (Gradio / Streamlit)
    └── your-username/your-demo
```

Every repository uses **Git LFS** for large files. Model weights, tokenizer files, and configs are versioned alongside code.

---

## Searching for models programmatically

```python
import os
from huggingface_hub import HfApi, list_models, ModelFilter

api = HfApi(token=os.getenv("HF_TOKEN"))

# List the top text-generation models by downloads, filtered by license
models = list_models(
    task="text-generation",
    library="transformers",
    sort="downloads",
    direction=-1,
    limit=10
)

for model in models:
    print(f"{model.id:<50} downloads={model.downloads:>10,}")
```

```python
# Search with specific filters
from huggingface_hub import list_models

# Find all instruction-tuned Llama models under Apache 2.0 license
llama_instruct_models = list(list_models(
    filter=ModelFilter(
        task="text-generation",
        library="transformers",
        language="en",
    ),
    search="Llama-3",
    sort="downloads",
    direction=-1,
    limit=20
))

for m in llama_instruct_models[:5]:
    print(f"{m.id}")
    print(f"  downloads: {m.downloads:,}")
    print(f"  likes: {m.likes}")
```

---

## Inspecting model metadata

```python
from huggingface_hub import HfApi, ModelInfo

api = HfApi(token=os.getenv("HF_TOKEN"))

def inspect_model(model_id: str) -> dict:
    """Get key metadata for a Hub model."""
    info: ModelInfo = api.model_info(model_id)

    return {
        "id": info.id,
        "author": info.author,
        "pipeline_tag": info.pipeline_tag,
        "tags": info.tags[:10] if info.tags else [],
        "downloads_last_month": info.downloads,
        "likes": info.likes,
        "created_at": info.created_at.isoformat() if info.created_at else None,
        "model_size_gb": _estimate_size_gb(info),
    }

def _estimate_size_gb(info: ModelInfo) -> str:
    """Rough size estimate from safetensors metadata if available."""
    safetensors = getattr(info, "safetensors", None)
    if safetensors and hasattr(safetensors, "total"):
        total_bytes = safetensors.total
        return f"{total_bytes / 1e9:.1f} GB"
    return "unknown"

# Compare two embedding models
for model_id in ["BAAI/bge-large-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]:
    meta = inspect_model(model_id)
    print(f"\n{meta['id']}")
    print(f"  Task: {meta['pipeline_tag']}")
    print(f"  Downloads: {meta['downloads_last_month']:,}")
    print(f"  Size: {meta['model_size_gb']}")
    print(f"  Tags: {meta['tags']}")
```

---

## Downloading model files

```python
from huggingface_hub import snapshot_download, hf_hub_download
import os

# Download a single file (e.g., tokenizer config)
tokenizer_config = hf_hub_download(
    repo_id="microsoft/DialoGPT-medium",
    filename="tokenizer_config.json",
    token=os.getenv("HF_TOKEN"),
    cache_dir="./hf_cache"
)
print(f"Downloaded to: {tokenizer_config}")

# Download all model files (will cache locally)
# snapshot_download() handles retries and partial downloads
local_path = snapshot_download(
    repo_id="BAAI/bge-small-en-v1.5",
    cache_dir="./hf_cache",
    token=os.getenv("HF_TOKEN"),
    ignore_patterns=["*.msgpack", "*.h5"]  # Skip non-safetensors formats
)
print(f"Model downloaded to: {local_path}")
```

> [!info] Hugging Face cache
> By default, `snapshot_download` caches to `~/.cache/huggingface/hub/`. Set `HF_HOME` or `cache_dir` to change this. Files are content-addressed by hash — the same model version downloaded twice takes only one disk slot.

---

## Uploading a model to the Hub

```python
from huggingface_hub import HfApi, create_repo, upload_folder

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create a new repository
repo_url = create_repo(
    repo_id="your-username/my-fine-tuned-model",
    repo_type="model",
    private=True,
    token=os.getenv("HF_TOKEN"),
    exist_ok=True
)
print(f"Repo: {repo_url}")

# Upload a local directory (model weights, config, tokenizer)
api.upload_folder(
    folder_path="./my_model_output",
    repo_id="your-username/my-fine-tuned-model",
    repo_type="model",
    commit_message="add fine-tuned llama3 for customer support"
)

# Or upload a single file
api.upload_file(
    path_or_fileobj="./README.md",
    path_in_repo="README.md",
    repo_id="your-username/my-fine-tuned-model",
    repo_type="model"
)
```

---

## Hub model card format

Every Hub repository should have a `README.md` with YAML frontmatter:

```markdown
---
license: apache-2.0
language:
  - en
tags:
  - text-generation
  - llama
  - fine-tuned
datasets:
  - squad
pipeline_tag: text-generation
base_model: meta-llama/Llama-3.1-8B-Instruct
---

# My Fine-Tuned Model

Brief description of what this model does and when to use it.

## Training details
- Base model: Llama 3.1 8B Instruct
- Training data: SQuAD + internal dataset
- Training method: QLoRA (r=16, alpha=32)
- Hardware: 1x A100 40GB, 4 hours

## Usage

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="your-username/my-fine-tuned-model")
```

## Limitations
- Only trained on English text
- May hallucinate on out-of-distribution topics
```

> [!tip] Model cards improve discoverability
> Models with complete model cards (license, tags, base_model, datasets) rank higher in Hub search. The `pipeline_tag` field is especially important — it determines which task section the model appears in.

---

[[00-agenda]] | [[02-transformers-library]]
