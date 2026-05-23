# Papers

Essential papers for LLM engineering. Organized by topic. Each entry: title, year, one-line summary, and why it matters for practitioners.

---

## Foundations

**Attention Is All You Need** (Vaswani et al., 2017)
The transformer paper. Self-attention, positional encoding, encoder-decoder architecture.
*Why read it:* Every modern LLM is built on this. Understanding multi-head attention explains why longer contexts are harder and why attention patterns matter.

**BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
Bidirectional transformer pre-training with MLM and NSP objectives.
*Why read it:* BERT's fine-tuning paradigm — pre-train on general data, fine-tune on task-specific data — is still the dominant approach.

**Language Models are Few-Shot Learners** (Brown et al., 2020, GPT-3)
Demonstrates emergent in-context learning: LLMs can perform tasks with just a few examples in the prompt.
*Why read it:* Explains why few-shot prompting works and sets expectations for prompt engineering.

---

## RAG and Retrieval

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020)
The original RAG paper: combining parametric knowledge (LLM) with non-parametric retrieval (dense index).
*Why read it:* The conceptual foundation for every RAG system you'll build.

**RAGAS: Automated Evaluation of Retrieval Augmented Generation** (Es et al., 2023)
Introduces faithfulness, answer relevancy, context recall, and context precision metrics computed without human annotation.
*Why read it:* The go-to evaluation framework for RAG — if you're building RAG, you need to know this.

**Self-RAG: Learning to Retrieve, Generate and Critique** (Asai et al., 2023)
Model learns when to retrieve (not always), generates its own retrieval decisions, and critiques its outputs.
*Why read it:* Advanced RAG technique; explains the "retrieval on demand" pattern.

---

## Fine-Tuning and PEFT

**LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
Fine-tune only low-rank decomposition matrices instead of full weight updates.
*Why read it:* LoRA is the default fine-tuning method for practitioners. Understanding rank r and alpha directly affects your training decisions.

**QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)
4-bit NF4 quantization + LoRA + paged optimizer = fine-tune 65B models on a single 48GB GPU.
*Why read it:* If you're fine-tuning any model > 1B params, you're using QLoRA. The paper explains why NF4 is better than FP4.

---

## Agents and Reasoning

**ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022)
Interleaving chain-of-thought reasoning with tool use actions.
*Why read it:* The conceptual basis for most LLM agent frameworks including LangGraph.

**Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (Wei et al., 2022)
Adding "Let's think step by step" significantly improves reasoning on math and logic tasks.
*Why read it:* The empirical basis for CoT prompting — explains when and why it helps.

---

## Evaluation and Safety

**Evaluating Large Language Models: A Comprehensive Survey** (Chang et al., 2023)
Survey of evaluation frameworks, benchmarks, and metrics across capability, alignment, and safety.
*Why read it:* Gives you vocabulary and frameworks for building your own evaluation systems.

**Constitutional AI: Harmlessness from AI Feedback** (Bai et al., 2022, Anthropic)
Train models to follow a set of principles using AI-generated feedback instead of human labels.
*Why read it:* Explains the RLAIF approach used in modern safety-aligned models.
