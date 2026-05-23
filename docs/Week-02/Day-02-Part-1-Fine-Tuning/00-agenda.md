# Agenda — Fine-Tuning

**Session length:** 3 hours | **Difficulty:** Advanced | **Prerequisites:** [[Week-01/Day-05-Part-2-Local-LLMs/01-overview|Local LLMs]], Python dataclasses, basic PyTorch tensor operations

---

## What you will build today

A LoRA fine-tuning pipeline for a classification task using PEFT and a Hugging Face base model, compared against a prompt-only baseline.

---

## Schedule

| Time | Topic | File |
|------|-------|------|
| 0:00–0:20 | Overview: full fine-tuning vs PEFT vs RAG vs prompting | [[01-fine-tuning-overview]] |
| 0:20–0:55 | LoRA and QLoRA: math, parameters, practical configs | [[02-lora-and-qlora]] |
| 0:55–1:25 | PEFT library: `get_peft_model`, training loop | [[03-peft]] |
| 1:25–1:45 | Training data: formats, quality, quantity rules | [[04-training-data]] |
| 1:45–2:00 | Decision framework: when to fine-tune | [[05-when-to-fine-tune]] |
| 2:00–2:40 | Practice exercises | [[06-practice-exercises]] |
| 2:40–3:00 | Interview questions review | [[07-interview-questions]] |

---

## Setup

```bash
pip install transformers peft trl datasets accelerate bitsandbytes torch
```

> [!warning] GPU required for full training
> QLoRA fine-tuning requires a CUDA GPU with at least 12 GB VRAM for a 7B model. For CPU-only environments, use the smallest models (GPT-2, DistilBERT) or run in Google Colab with a T4 GPU.

---

[[Week-02/Day-01-Part-2-Advanced-RAG/07-interview-questions|← Day 1 Part 2]] | [[01-fine-tuning-overview|Fine-Tuning Overview →]]
