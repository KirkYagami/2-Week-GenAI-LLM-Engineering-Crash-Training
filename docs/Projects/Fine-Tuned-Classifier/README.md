# Project 6 — Fine-Tuned Classifier

Fine-tune a small language model with LoRA for a text classification task and compare it against a zero-shot prompted baseline. This project shows when fine-tuning beats prompting — and when it doesn't.

## What you'll build

A complete fine-tuning pipeline that:
- Prepares a labeled classification dataset in chat format
- Fine-tunes `Qwen/Qwen2-0.5B-Instruct` (a small, fast model) with QLoRA
- Evaluates the fine-tuned model vs. zero-shot gpt-4o-mini on held-out test data
- Merges and saves the adapter for inference
- Serves predictions via a FastAPI endpoint

## Skills covered

| Skill | Where |
|-------|-------|
| Dataset preparation in chat format | [[01-setup]] |
| QLoRA fine-tuning with SFTTrainer | [[02-implementation]] |
| Label masking and gradient flow | [[02-implementation]] |
| Merge and save LoRA adapter | [[02-implementation]] |
| F1 comparison: fine-tuned vs zero-shot | [[04-evaluation]] |

## Prerequisites

- Week 02 Day 02 Part 1 — Fine-Tuning
- Week 02 Day 04 Part 2 — Deployment

## Task: customer review sentiment classification

Labels: `positive`, `negative`, `neutral`  
Input: product review text (< 200 words)

This is deliberately simple so you can focus on the fine-tuning pipeline, not the task.

## Tech stack (requires GPU or Google Colab)

```
transformers==4.45.0
peft==0.13.0
trl==0.11.4
bitsandbytes==0.44.0
datasets==3.0.1
accelerate==1.0.1
torch==2.4.1
openai==1.51.0
fastapi==0.115.0
uvicorn==0.30.6
```

> [!info] This project requires a GPU
> QLoRA fine-tuning requires at least 8GB VRAM. Use Google Colab (free T4 GPU) or a cloud instance (Lambda Labs, RunPod). The inference endpoint can run on CPU.

---

[[01-setup]]
