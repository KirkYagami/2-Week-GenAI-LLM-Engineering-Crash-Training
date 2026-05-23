# LoRA and QLoRA

Full fine-tuning updates all 7 billion parameters in a 7B model — expensive, memory-intensive, and often unnecessary. LoRA (Low-Rank Adaptation) gets 80–95% of the benefit by updating a tiny fraction of parameters. QLoRA stacks quantization on top, making it possible to fine-tune a 7B model on a single consumer GPU.

## Learning objectives

- Understand the LoRA math: why low-rank matrices work
- Configure `LoraConfig` with rank, alpha, target modules
- Set up QLoRA with `BitsAndBytesConfig` 4-bit quantization
- Know what rank, alpha, and dropout control
- Merge LoRA weights back into the base model for deployment

---

## The LoRA idea

A weight matrix W (e.g., 4096×4096 in a large transformer) is updated during full fine-tuning: W' = W + ΔW. The key observation: for most task adaptations, ΔW has *low intrinsic rank* — most of the useful change lives in a small subspace.

LoRA decomposes ΔW into two small matrices:

```
ΔW ≈ A × B
where A is (d × r) and B is (r × k)
and r << min(d, k)
```

For a 4096×4096 matrix with rank r=16:
- Full fine-tuning: 16,777,216 parameters
- LoRA: 4096×16 + 16×4096 = 131,072 parameters — **128x fewer**

Only A and B are trained. The original W is frozen.

---

## LoRA implementation with PEFT

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank — higher = more parameters, more capacity
    lora_alpha=32,                 # Scaling factor: effective_lr = alpha / r
    lora_dropout=0.1,              # Dropout on LoRA layers (regularization)
    target_modules=["q_proj", "v_proj"],  # Which attention matrices to adapt
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 2,097,152 || all params: 2,781,085,696 || trainable%: 0.0754
```

### Choosing target modules

```python
# See all named modules in the model
for name, module in base_model.named_modules():
    print(name)

# Common choices by architecture
TARGET_MODULES = {
    "llama":  ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi-2":  ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "gpt-2":  ["c_attn", "c_proj"],  # GPT-2 uses fused QKV
}
```

Start with `q_proj` and `v_proj`. If performance is insufficient, add `k_proj`, `o_proj`, and the MLP layers.

---

## LoRA hyperparameter guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `r` (rank) | 4–64 | Higher = more capacity, more parameters |
| `lora_alpha` | 2×r to 4×r | Effective learning rate scale |
| `lora_dropout` | 0.05–0.15 | Regularization; use 0 for small datasets |
| `target_modules` | q,v → all | More modules = more parameters, better fit |

> [!tip] Start with r=16, alpha=32
> This combination (alpha = 2×r) is the most common default. Increase r to 32 or 64 if the model isn't adapting well. The learning rate interacts with alpha/r — if you change r, scale alpha proportionally.

---

## QLoRA: quantize the base, train the adapters

QLoRA keeps the base model in 4-bit (NF4) precision and trains LoRA adapters in 16-bit. This reduces memory by 4x compared to 16-bit LoRA.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — best for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # Upcast for computation
    bnb_4bit_use_double_quant=True,     # Quantize the quantization constants too
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Required before adding LoRA to a quantized model
model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### Memory requirements

| Technique | VRAM for 7B | VRAM for 13B |
|-----------|-------------|--------------|
| Full FT (fp16) | 28 GB | 52 GB |
| LoRA (fp16) | 16 GB | 30 GB |
| QLoRA (4-bit) | 6 GB | 10 GB |
| QLoRA (4-bit, batch=4) | 12 GB | 20 GB |

A single NVIDIA RTX 3090 (24 GB) can QLoRA-train a 7B model with batch size 4–8.

---

## Merging LoRA weights for deployment

After training, merge the LoRA adapters into the base model weights. The merged model runs inference at the same speed as the original — no adapter overhead.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base + adapters
base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, "./my-lora-adapter")

# Merge adapters into base weights
merged = model.merge_and_unload()

# Save the merged model (standard HF format — no PEFT dependency at inference time)
merged.save_pretrained("./my-finetuned-model")
tokenizer.save_pretrained("./my-finetuned-model")

print("Merged model saved — ready for deployment")
```

> [!info] Keep the adapter separate for multi-task serving
> If you're serving multiple fine-tuned variants of the same base model, keep adapters separate and swap them at inference time with `PeftModel.from_pretrained()`. This saves storage: one base model + N small adapters instead of N full model copies.

---

## LoRA vs full fine-tuning: when is LoRA not enough?

LoRA underperforms full fine-tuning in two situations:
1. **Massively different output distribution** — if the task requires writing in a completely different format or language the model barely knows, full FT has more capacity
2. **Very large datasets** (>100K examples) — LoRA's parameter budget becomes the bottleneck; full FT can utilize all the training signal

For the majority of real-world task adaptation (classification, extraction, style, domain specialization), LoRA with r=16 on q,v,k,o projection matrices matches or comes within 2–3% of full fine-tuning on benchmarks.

---

[[01-fine-tuning-overview]] | [[03-peft]]
