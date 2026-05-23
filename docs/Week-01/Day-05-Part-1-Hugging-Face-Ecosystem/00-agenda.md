# Agenda — Hugging Face Ecosystem

Hugging Face is where the open-source ML community lives. Over 500,000 models, 200,000 datasets, and a growing collection of hosted demo environments — all accessible through a consistent Python SDK. For LLM engineers, the Hub is where you find fine-tuned models for your domain, test them without infrastructure, and publish your own work.

## Learning objectives

By the end of this session you will be able to:

- Navigate the Hub to find, evaluate, and download models
- Load and run inference with the `transformers` library
- Call hosted models via the Inference API without GPU hardware
- Deploy a demo app to Spaces with Gradio

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00 – 0:25 | Hugging Face Hub — models, datasets, orgs | [[01-hugging-face-hub]] |
| 0:25 – 1:05 | Transformers library — pipelines, tokenizers, model loading | [[02-transformers-library]] |
| 1:05 – 1:35 | Inference API — serverless hosted inference | [[03-inference-api]] |
| 1:35 – 2:00 | Spaces — hosting demos with Gradio | [[04-spaces]] |
| 2:00 – 2:30 | Datasets and model cards | [[05-datasets-and-models]] |
| 2:30 – 3:00 | Practice exercises | [[06-practice-exercises]] |

## Setup

```bash
pip install transformers datasets huggingface_hub accelerate torch gradio
```

```python
import os
from huggingface_hub import login

# Authenticate — get your token at https://huggingface.co/settings/tokens
login(token=os.getenv("HF_TOKEN"))
```

> [!tip] Free tier is enough for this session
> The Hugging Face Inference API has a free tier that supports most small models. For larger models (70B+), you need a Pro subscription or self-hosted inference. The exercises in this session use models that run on the free tier.

[[../Day-04-Part-2-Responsible-AI-and-Safety/06-interview-questions|← Day 04 Part 2]] | [[01-hugging-face-hub|Start →]]
