# Claude Code Prompt — Contractual Hallucination Eliminator

## CONTEXT / KNOWN ISSUES

1. **Model is Qwen/Qwen3.5-9B** — confirm exact HF Hub model ID before loading
2. **No external APIs** — no Claude API, no OpenAI. All data generation must be done programmatically or with local models only
4. **CONTRADICTS_PRIOR needs a deterministic rule set** — implement it but gate it behind a `--use_contradicts_prior` flag; default eval is 2-class (GROUNDED / ABSENT)
5. **vLLM ROCm support is flaky** — use HF `pipeline` as primary serving method, note vLLM as optional
6. **HuggingFace Space with Gradio is a hard submission requirement** — `demo/app.py` is a first-class deliverable

---

## PROJECT OVERVIEW

Fine-tune Qwen3.5-9B on AMD MI300X (ROCm) to output calibrated uncertainty signals instead of hallucinating answers about contracts.

The model outputs structured JSON with one of three labels:
- GROUNDED: answer exists in the document, with citation
- ABSENT: clause/information is not present in this document
- CONTRADICTS_PRIOR: document deviates from standard practice (implement but make optional via --use_contradicts_prior flag)

---

## REPOSITORY STRUCTURE

Build exactly this structure:

contractual-hallucination-eliminator/
├── data/
│   ├── 01_download_cuad.py
│   ├── 02_perturb_and_generate.py
│   ├── 03_build_dataset.py
│   └── schema.py
├── training/
│   ├── train.py
│   ├── config.yaml
│   └── prompt_template.py
├── eval/
│   ├── benchmark.py
│   ├── metrics.py
│   └── compare_baseline.py
├── serving/
│   └── inference.py
├── demo/
│   ├── app.py
│   └── sample_contracts/
├── requirements.txt
├── requirements_rocm.txt
├── README.md
└── .env.example

---

## DETAILED SPECS

### data/schema.py
- Pydantic v2 models for all data structures
- RawCUADExample, PerturbedExample, LabeledQAExample, BenchmarkResult
- Label enum: GROUNDED, ABSENT, CONTRADICTS_PRIOR
- JSON output schema that the fine-tuned model will produce:
  {
    "question": str,
    "label": "GROUNDED | ABSENT | CONTRADICTS_PRIOR",
    "answer": str | null,
    "citation": str | null,
    "reasoning": str
  }

### data/01_download_cuad.py
- Load CUAD from HuggingFace datasets library (theatricusproject/cuad or equivalent)
- Extract per example: contract_id, contract_text, question, answer_text, answer_start
- Preserve examples with empty answers (these are natural ABSENT examples)
- Save to data/raw/cuad_raw.jsonl
- Print summary stats: total examples, examples with answers, examples without answers

### data/02_perturb_and_generate.py
- Input: data/raw/cuad_raw.jsonl
- This script both perturbs contracts AND generates the final labeled QA examples
  deterministically — no external model calls

STEP 1 — PERTURBATION:
Three strategies applied to CUAD examples that have answer spans:
  1. REMOVE: surgically delete the answer span text from the contract → label = ABSENT
  2. INVERT: negate key legal terms in the answer span:
     - "shall" ↔ "shall not"
     - flip numeric values by ±20% (e.g., "30 days" → "24 days")
     - swap Party A / Party B references within the span
     → label = CONTRADICTS_PRIOR
  3. CONTRADICT: replace the answer span with a randomly sampled answer span
     from a different contract in the dataset (same clause type where possible)
     → label = CONTRADICTS_PRIOR

Keep all original unperturbed examples → label = GROUNDED

STEP 2 — QUESTION GENERATION (no model, fully deterministic):
- CUAD already includes questions for each clause type. Use them directly.
- For perturbed examples, reuse the original CUAD question verbatim — the question
  stays the same, only the contract has changed, which is exactly the right setup
  for testing whether the model hallucinates or correctly detects absence/contradiction.
- Construct the full LabeledQAExample:
  - For GROUNDED: answer = answer_text, citation = answer_text
  - For ABSENT (REMOVE): answer = null, citation = null
  - For CONTRADICTS_PRIOR: answer = the perturbed text, citation = perturbed span
  - reasoning = auto-generated one-sentence string describing the label
    (e.g., "The clause was removed from this version of the contract.")

Target class distribution: 40% GROUNDED, 40% ABSENT, 20% CONTRADICTS_PRIOR
Seed all randomness: seed=42
Output: data/perturbed/labeled.jsonl
Print: counts per class

### data/03_build_dataset.py
- Input: data/perturbed/labeled.jsonl
- Deduplicate on (contract_id, question)
- Stratified split: 80% train, 10% val, 10% test
- Save to data/final/train.jsonl, val.jsonl, test.jsonl
- Print final class distribution for each split
- Optionally push to HuggingFace Hub as a dataset if HF_TOKEN env var is set

### training/prompt_template.py
- System prompt establishing the task
- Input format:
    [CONTRACT]
    {contract_text}
    [/CONTRACT]

    Question: {question}
- Output format: raw JSON only, no explanation outside the JSON object
- Include 3 hardcoded few-shot examples in the system prompt (one per label class)
- Function: format_training_example(example: LabeledQAExample) → {"prompt": str, "completion": str}
- Function: format_inference_prompt(contract_text: str, question: str) → str

### training/config.yaml
model_name: Qwen/Qwen3.5-9B
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
learning_rate: 2e-4
num_epochs: 3
batch_size: 4
gradient_accumulation_steps: 4
max_seq_length: 4096
bf16: true
output_dir: ./checkpoints
hub_model_id: PLACEHOLDER/contractual-hallucination-eliminator
use_contradicts_prior: false

### training/train.py
- Load Qwen3.5-9B in 4-bit quantization using bitsandbytes
- AMD/ROCm detection and fallback:
    is_rocm = torch.cuda.is_available() and 'AMD' in torch.cuda.get_device_name(0)
    if is_rocm and bitsandbytes unavailable: fall back to fp16 (no quantization)
- Apply LoRA via PEFT using values from config.yaml
- Use SFTTrainer from trl
- Evaluate per-class accuracy on val set at end of each epoch
- Save best checkpoint by val accuracy
- Push final model to HF Hub if HF_TOKEN is set
- Log to wandb if WANDB_API_KEY is set, otherwise log to stdout

### eval/metrics.py
Functions:
- compute_hallucination_rate(predictions, labels) → float
  Definition: model predicts GROUNDED when ground truth is ABSENT or CONTRADICTS_PRIOR
- compute_per_class_metrics(predictions, labels) → dict
  Returns precision, recall, f1 per class
- compute_citation_accuracy(predictions, labels, contracts) → float
  For GROUNDED predictions: fraction where citation is a substring of the contract text
- format_benchmark_report(results: dict) → str
  Returns a clean printable table

### eval/benchmark.py
- Load test split from data/final/test.jsonl
- Load fine-tuned model from --model_path (checkpoint dir or HF Hub repo)
- Run inference on all test examples
- Compute all metrics
- Save to eval/results/benchmark_results.json
- Print formatted report to stdout
- CLI: --model_path, --data_path, --output_path, --use_contradicts_prior

### eval/compare_baseline.py
- Load 3 sample contracts from demo/sample_contracts/
- Run hardcoded questions through:
    1. Base Qwen3.5-9B (untuned, zero-shot with same prompt template)
    2. Fine-tuned model
- Output side-by-side markdown table to stdout and save to eval/results/comparison.md
- Columns: Question | Base Model Output | Fine-tuned Output | Ground Truth
- Highlight rows where base model hallucinated (predicted GROUNDED when answer is ABSENT)

### serving/inference.py
Class ContractAnalyzer:
- __init__(self, model_path: str, device: str = "auto")
- analyze(self, contract_text: str, question: str) → LabeledQAExample
- analyze_batch(self, examples: list[dict]) → list[LabeledQAExample]
- Uses HuggingFace pipeline for ROCm compatibility (not vLLM)
- Validates output JSON against schema; retries once with a stricter prompt if JSON parse fails
- Truncates contracts exceeding 8192 tokens with a printed warning

### demo/app.py
Gradio app for HuggingFace Spaces:
- Tab 1 — "Analyze Contract":
    - Left: textarea to paste contract + 3 buttons to load sample contracts
    - Middle: question input + Analyze button
    - Right: result panel with color-coded label badge (green/red/yellow),
      answer text, citation highlighted within contract snippet, reasoning sentence
- Tab 2 — "Benchmark Demo":
    - Hardcoded side-by-side table showing base model vs fine-tuned model
      on 5 representative examples (mix of GROUNDED, ABSENT, CONTRADICTS_PRIOR)
    - Make it visually clear where base model hallucinated
- Load model from HF Hub using HF_TOKEN from Space secrets
- Show a visible warning banner if model fails to load
- Use gr.Blocks() layout

### demo/sample_contracts/
Three short (~500 word) fictional but realistic contract excerpts as .txt files.
Choose content so they're maximally useful for the demo:
  1. software_license.txt — includes a limitation of liability clause
  2. nda.txt — includes a non-compete clause
  3. service_agreement.txt — deliberately missing a termination for convenience clause
     (so "Does this contract include termination for convenience?" → ABSENT)

### requirements.txt
Pin minimum versions:
transformers>=4.45.0
peft>=0.12.0
trl>=0.11.0
datasets>=2.20.0
gradio>=4.40.0
pydantic>=2.0.0
bitsandbytes>=0.43.0
wandb>=0.17.0
accelerate>=0.33.0
huggingface_hub>=0.24.0
torch>=2.3.0

### requirements_rocm.txt
# AMD ROCm setup — install ROCm PyTorch FIRST before this file:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
# Then: pip install -r requirements_rocm.txt
(same as requirements.txt but without torch)

### .env.example
HF_TOKEN=your_hf_token_here
WANDB_API_KEY=your_wandb_key_here
HF_DATASET_REPO=your-username/cuad-hallucination-labeled
HF_MODEL_REPO=your-username/contractual-hallucination-eliminator

### README.md
Include:
1. One-paragraph project overview
2. The three output labels with concrete examples
3. ASCII architecture diagram: CUAD → Perturbation → Dataset → LoRA Fine-tune → Eval → Gradio Demo
4. Setup: standard GPU section and AMD ROCm section (separate)
5. Pipeline steps in order: data → train → eval → demo, with exact commands
6. Benchmark results table (placeholder with correct column headers:
   Model | Hallucination Rate | GROUNDED F1 | ABSENT F1 | CONTRADICTS_PRIOR F1)
7. HuggingFace Space link placeholder
8. License: Apache 2.0

---

## HARD CONSTRAINTS
- No external AI API calls anywhere in the codebase
- Every file fully implemented — no TODOs, no pass stubs, no placeholder functions
- All scripts have argparse CLIs with --help
- All file I/O creates parent directories automatically if they don't exist
- Use pathlib throughout, never os.path
- Type hints on every function signature
- Every script prints clear progress to stdout
```
