# llama.cpp

llama.cpp is a C++ implementation of LLM inference, optimized for CPU (and GPU via CUDA/Metal). It introduced the GGUF format — the most widely supported local model format — and remains the fastest pure-CPU LLM inference engine. Ollama uses llama.cpp under the hood.

## Learning objectives

- Understand GGUF format and how quantization levels are named
- Install `llama-cpp-python` and run inference without Ollama
- Integrate llama.cpp with LangChain for drop-in local LLM use
- Configure GPU offloading for mixed CPU/GPU inference

---

## GGUF format

GGUF (GPT-Generated Unified Format) is the file format used by llama.cpp. It stores model weights, tokenizer data, and metadata in a single self-contained file.

```
model.gguf
├── metadata          ← model architecture, context length, vocab size
├── tokenizer         ← vocabulary, special tokens, BPE merges
└── tensors           ← quantized weights (Q4_K_M, Q8_0, F16, etc.)
```

**Naming convention:**
```
Llama-3.1-8B-Instruct.Q4_K_M.gguf
│              │        │
│              │        └── Quantization: Q4 K-mean Medium
│              └── Model size (8 billion parameters)
└── Model family and variant
```

**Where to find GGUF files:** Search `huggingface.co` for a model name + "GGUF". Most popular models have GGUF versions uploaded by the community (e.g., `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`).

---

## Installation

```bash
# CPU only (works everywhere)
pip install llama-cpp-python

# With CUDA GPU support (requires CUDA toolkit)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# With Metal GPU support (macOS Apple Silicon)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

## Basic inference

```python
from llama_cpp import Llama

# Load a GGUF model
llm = Llama(
    model_path="./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf",
    n_ctx=4096,          # Context window size
    n_threads=8,         # CPU threads (set to physical core count)
    n_gpu_layers=0,      # 0 = CPU only; set to 35 for full GPU offload
    verbose=False        # Suppress loading logs
)

# Simple completion
output = llm(
    prompt="Q: What is retrieval-augmented generation?\nA:",
    max_tokens=200,
    stop=["Q:", "\n\n"],
    echo=False           # Don't include prompt in output
)
print(output["choices"][0]["text"])
```

---

## Chat completions with chat templates

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=35,     # Offload 35 layers to GPU (adjust based on your VRAM)
    chat_format="llama-3",  # Applies the correct chat template
    verbose=False
)

# OpenAI-compatible chat completions
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a concise coding assistant."},
        {"role": "user", "content": "Write a Python function to compute the nth Fibonacci number."}
    ],
    max_tokens=300,
    temperature=0.0,
    stop=["<|eot_id|>"]
)
print(response["choices"][0]["message"]["content"])
```

---

## GPU layer offloading

A key feature of llama.cpp is partial GPU offloading. If you have a GPU with less VRAM than the full model requires, you can offload only some layers to GPU and run the rest on CPU.

```python
import subprocess

def get_model_layer_count(model_path: str) -> int:
    """Estimate number of transformer layers from model size (approximation)."""
    import os
    size_gb = os.path.getsize(model_path) / 1e9
    # Rough heuristic: ~1 layer per 0.1 GB for Q4 7B-class models
    return int(size_gb * 4)

def optimal_gpu_layers(model_path: str, vram_gb: float) -> int:
    """Estimate optimal n_gpu_layers for available VRAM."""
    import os
    model_size_gb = os.path.getsize(model_path) / 1e9
    # Reserve 2 GB for KV cache and other overhead
    available_for_model = max(0, vram_gb - 2.0)
    fraction = available_for_model / model_size_gb
    total_layers = get_model_layer_count(model_path)
    return int(total_layers * min(fraction, 1.0))

# Example: 7B Q4 model (4.1 GB) on a GPU with 6 GB VRAM
# optimal_layers ≈ (6-2)/4.1 * 32 ≈ 31 layers on GPU, rest on CPU
```

```python
from llama_cpp import Llama
import time

def benchmark_generation(llm: Llama, prompt: str, n_tokens: int = 100) -> dict:
    start = time.perf_counter()
    output = llm(prompt, max_tokens=n_tokens, echo=False)
    elapsed = time.perf_counter() - start
    generated = len(output["choices"][0]["text"].split())

    return {
        "tokens_per_second": n_tokens / elapsed,
        "elapsed_sec": elapsed,
        "output": output["choices"][0]["text"][:100]
    }

# Compare: all CPU vs partial GPU offload
llm_cpu = Llama(model_path="./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf",
                n_ctx=2048, n_gpu_layers=0, verbose=False)
llm_gpu = Llama(model_path="./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf",
                n_ctx=2048, n_gpu_layers=20, verbose=False)

prompt = "Explain gradient descent in simple terms:"

cpu_bench = benchmark_generation(llm_cpu, prompt)
gpu_bench = benchmark_generation(llm_gpu, prompt)

print(f"CPU only: {cpu_bench['tokens_per_second']:.1f} tok/s")
print(f"GPU (20 layers): {gpu_bench['tokens_per_second']:.1f} tok/s")
print(f"Speedup: {gpu_bench['tokens_per_second']/cpu_bench['tokens_per_second']:.1f}×")
```

---

## Streaming with llama.cpp

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=35,
    verbose=False
)

# Stream tokens as they generate
print("Streaming response:")
for chunk in llm.create_chat_completion(
    messages=[{"role": "user", "content": "List 5 best practices for RAG pipeline design."}],
    max_tokens=400,
    temperature=0.7,
    stream=True
):
    delta = chunk["choices"][0].get("delta", {})
    if "content" in delta:
        print(delta["content"], end="", flush=True)
print()
```

---

## Using llama.cpp with LangChain

```python
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

# LlamaCpp is a LangChain LLM wrapper
llm = LlamaCpp(
    model_path="./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf",
    n_gpu_layers=35,
    n_ctx=4096,
    n_threads=8,
    temperature=0.0,
    max_tokens=300,
    verbose=False,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

template = """Use the context to answer the question. If the answer is not in context, say so.

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
chain = prompt | llm

context = "The company was founded in 2015 and has 500 employees across 3 offices."
question = "How many employees does the company have?"

# result = chain.invoke({"context": context, "question": question})
# Expected: "The company has 500 employees."
```

> [!info] llama.cpp vs Ollama
> Ollama is llama.cpp with a management layer: model library, automatic GPU detection, REST API, and Modelfile configuration. Use Ollama unless you need programmatic control over loading (partial layer offloading to specific GPUs, custom memory layouts, embedding + generation from the same model instance). For most use cases, Ollama is easier.

---

[[02-ollama]] | [[04-quantization]]
