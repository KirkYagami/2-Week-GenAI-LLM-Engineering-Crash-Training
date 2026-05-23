# PEFT and the Training Loop

The PEFT library (Parameter-Efficient Fine-Tuning) is Hugging Face's abstraction layer for LoRA, QLoRA, prefix tuning, and other adapter methods. TRL (Transformer Reinforcement Learning) adds `SFTTrainer` — a wrapper around the Hugging Face `Trainer` that handles chat formatting, label masking, and packing automatically.

## Learning objectives

- Set up a complete SFT training pipeline with `SFTTrainer`
- Configure `TrainingArguments` for gradient checkpointing and mixed precision
- Load and format datasets in the JSONL chat format
- Monitor loss curves and detect overfitting
- Push trained adapters to the Hugging Face Hub

---

## Dataset format for instruction fine-tuning

SFT datasets typically use one of two formats:

### Format 1: Pre-formatted strings

```python
dataset = [
    {
        "text": "<|system|>You are a helpful assistant.<|endoftext|><|user|>What is 2+2?<|assistant|>4<|endoftext|>"
    },
    ...
]
```

### Format 2: Messages list (preferred)

```python
dataset = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
    },
    ...
]
```

`SFTTrainer` with `dataset_text_field` or a formatting function handles the chat template conversion automatically.

---

## Full SFT training pipeline

```python
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── 1. Load model ────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ── 2. LoRA config ───────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── 3. Dataset ───────────────────────────────────────────────
TRAIN_DATA = [
    {"text": "### Instruction: Classify sentiment.\n### Input: I love this product!\n### Response: positive"},
    {"text": "### Instruction: Classify sentiment.\n### Input: Terrible experience.\n### Response: negative"},
    {"text": "### Instruction: Classify sentiment.\n### Input: It was okay, nothing special.\n### Response: neutral"},
    # Add at least 100–500 examples in practice
]
dataset = Dataset.from_list(TRAIN_DATA)

# ── 4. Training arguments ────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./phi2-sentiment-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # Effective batch = 4 × 4 = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,                          # Use bf16=True on Ampere+ GPUs
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit",          # Memory-efficient optimizer for QLoRA
    gradient_checkpointing=True,        # Trade compute for memory
    report_to="none",                   # Set to "wandb" to enable tracking
)

# ── 5. Trainer ───────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
trainer.save_model("./phi2-sentiment-lora/final")
print("Training complete!")
```

---

## Key training arguments explained

```python
TrainingArguments(
    # Batch size management
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # Simulate larger batches without OOM

    # Learning rate schedule
    learning_rate=2e-4,                 # LoRA uses higher LR than full FT (1e-5–1e-4 for full FT)
    lr_scheduler_type="cosine",         # Cosine decay; "linear" is also common
    warmup_ratio=0.03,                  # Warm up for 3% of steps

    # Memory optimization
    gradient_checkpointing=True,        # Recompute activations during backward pass
    optim="paged_adamw_32bit",          # Keeps optimizer state on CPU, pages to GPU as needed
    fp16=True,                          # 16-bit training (use bf16=True on A100/H100)

    # Checkpointing
    save_strategy="epoch",
    save_total_limit=2,                 # Keep only last 2 checkpoints

    # Sequence length
    # Set max_seq_length in SFTTrainer, not here
)
```

> [!tip] Effective batch size = per_device_batch × gradient_accumulation × num_gpus
> For single-GPU QLoRA: `per_device_train_batch_size=4, gradient_accumulation_steps=4` gives effective batch 16 — enough for stable training. Don't use batch size 1; loss will be noisy.

---

## Monitoring training

```python
# If you log to WandB, plot training/eval loss
# Otherwise, check logs manually

import json

log_file = "./phi2-sentiment-lora/trainer_state.json"
with open(log_file) as f:
    state = json.load(f)

for entry in state["log_history"]:
    if "loss" in entry:
        print(f"step {entry['step']:4d} | loss {entry['loss']:.4f}")
```

### Reading the loss curve

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Loss decreasing steadily | Good | Continue |
| Loss plateau after epoch 1 | Overfitting or LR too low | Add eval set; try warmup |
| Loss oscillating | LR too high | Halve learning rate |
| Loss suddenly spikes | Gradient explosion | Add `max_grad_norm=1.0` |
| Train loss low, eval loss high | Overfitting | Reduce epochs, add dropout |

---

## Evaluating the fine-tuned model

```python
from peft import PeftModel
import torch

# Load fine-tuned model for inference
base = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "./phi2-sentiment-lora/final")
model.eval()

def classify_sentiment(text: str) -> str:
    prompt = f"### Instruction: Classify sentiment.\n### Input: {text}\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

# Test
examples = [
    "This product exceeded my expectations!",
    "Worst purchase I've ever made.",
    "It arrived on time. Nothing to complain about.",
]
for ex in examples:
    print(f"{ex[:50]:<50} → {classify_sentiment(ex)}")
```

---

## Pushing to Hugging Face Hub

```python
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

# Push adapter only (base model stays on Hub)
trainer.model.push_to_hub("your-username/phi2-sentiment-lora")
tokenizer.push_to_hub("your-username/phi2-sentiment-lora")

# Load from Hub later
from peft import PeftModel
model = PeftModel.from_pretrained(base, "your-username/phi2-sentiment-lora")
```

> [!success] Adapter files are small
> A LoRA adapter for a 7B model with r=16 is typically 50–150 MB, compared to 14 GB for the full fp16 model. Push the adapter; users load the base model separately.

---

[[02-lora-and-qlora]] | [[04-training-data]]
