# LoRA and QLoRA

Fine-tuning questions separate candidates who have read about it from those who have actually run a training job. Know the parameter math — interviewers will ask you to reason about memory requirements on the spot.

---

## Q1: What is LoRA and why is it more memory-efficient than full fine-tuning?

??? "Show answer"
    **LoRA (Low-Rank Adaptation)** freezes the pretrained model weights and adds small trainable decomposition matrices to specific layers. Instead of updating a weight matrix W (shape d×d), it learns two matrices A (d×r) and B (r×d), where r << d.

    The weight update is: `ΔW = A × B`

    Memory efficiency: for a 7B parameter model, full fine-tuning requires storing gradients and optimizer states for all 7B parameters — typically 4× the model size, or ~112 GB. LoRA with rank 16 might train only ~0.1% of parameters — requiring a fraction of that memory.

    ```python
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

    lora_config = LoraConfig(
        r=16,                          # rank — higher = more capacity, more memory
        lora_alpha=32,                 # scaling factor = alpha / r
        target_modules=["q_proj", "v_proj"],  # which layers to adapt
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # trainable params: 2,621,440 || all params: 3,214,033,920 || trainable%: 0.0816%
    ```

---

## Q2: What do the rank `r` and `lora_alpha` parameters control?

??? "Show answer"
    **`r` (rank)**: controls the dimensionality of the LoRA matrices and thus the number of trainable parameters. Higher rank = more expressive adaptation, more memory.

    Common values:
    - `r=4` or `r=8` — minimal adaptation, format/style changes
    - `r=16` — good default for most tasks
    - `r=64` — heavy adaptation, approaching full fine-tuning quality (but at higher cost)

    **`lora_alpha`**: scaling factor applied to the LoRA matrices during training: `scale = alpha / r`. Setting `alpha = 2 * r` (e.g., `r=16, alpha=32`) is the most common convention — it keeps the effective learning rate stable as you vary rank.

    If you double `r`, double `alpha` to maintain the same effective learning rate. Think of alpha as controlling how strongly the LoRA update influences the frozen weights.

---

## Q3: What is QLoRA and how does NF4 quantization enable it?

??? "Show answer"
    **QLoRA** = Quantized LoRA. The base model weights are quantized to 4-bit NF4 format (reducing memory by ~8×), while LoRA adapters are trained in bfloat16. This enables fine-tuning 7B–70B models on a single consumer or cloud GPU.

    **NF4 (Normal Float 4)**: a 4-bit data type whose quantization bins are spaced to match the normal distribution of pretrained weights. More accurate than standard 4-bit integer quantization for this specific distribution.

    ```python
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # quantize the quantization constants too
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)  # enable gradient checkpointing
    model = get_peft_model(model, lora_config)
    ```

    Memory rule of thumb: a 7B model in NF4 requires ~5 GB for weights + ~3 GB for LoRA adapters and optimizer states = ~8 GB total, fitting on a single T4 GPU.

---

## Q4: What is catastrophic forgetting and how does LoRA avoid it?

??? "Show answer"
    **Catastrophic forgetting**: when fine-tuning updates all weights, the model forgets previously learned knowledge. A model fine-tuned on medical Q&A may lose its general language capabilities.

    LoRA avoids this because **the pretrained weights are frozen** — only the small adapter matrices are trained. The base knowledge is preserved in the frozen weights; the adapters learn the task-specific delta.

    This also makes LoRA adapters composable: you can train multiple adapters for different tasks and swap them at inference time without maintaining multiple copies of the base model.

    ```python
    # Merge adapter into base model for deployment (no runtime overhead)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("./fine-tuned-model")

    # Or keep adapter separate for hot-swapping
    model.save_pretrained("./adapters/medical-qa")
    # Load a different adapter for a different task
    model.load_adapter("./adapters/legal-qa")
    ```

---

## Q5: Walk me through a complete QLoRA fine-tuning pipeline.

??? "Show answer"
    ```python
    import os, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset

    # 1. Load tokenizer and quantized model
    model_id = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    # 2. Configure LoRA
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

    # 3. Load dataset (chat template format)
    dataset = load_dataset("json", data_files="train.jsonl", split="train")

    # 4. Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",  # field containing formatted conversation
        max_seq_length=2048,
        args=TrainingArguments(
            output_dir="./checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_steps=100,
        ),
    )
    trainer.train()

    # 5. Merge and save
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained("./fine-tuned-model")
    tokenizer.save_pretrained("./fine-tuned-model")
    ```

---

*Previous: [Tool Use](../04-Agents-and-LangGraph/03-tool-use.md) | Next: [Training Data](02-training-data.md)*
