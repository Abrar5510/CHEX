# CHEX - Document Intelligence

CHEX is a fine-tuned Qwen3.5-9B model trained on AMD MI300X (ROCm) to produce **calibrated uncertainty signals** when answering questions about legal documents. Instead of confidently fabricating answers, CHEX outputs a structured JSON object that explicitly signals whether the requested information is present, absent, or contradicts standard terms — eliminating the hallucination problem that plagues off-the-shelf LLMs in document analysis.

---

## Output Labels

CHEX classifies every contract question into one of three labels:

| Label | Meaning | Example |
|-------|---------|---------|
| **GROUNDED** | The answer exists verbatim in the contract | *"The contract specifies a $50,000 liability cap in Section 5."* |
| **ABSENT** | The clause or information is not present anywhere in the contract | *"No termination for convenience clause exists in this document."* |
| **CONTRADICTS_PRIOR** | A clause exists but its terms deviate from standard legal practice | *"The non-compete clause uses 'shall engage' instead of 'shall not engage', inverting the standard obligation."* |

### JSON Output Format

```json
{
  "question": "Does this contract include a limitation of liability clause?",
  "label": "GROUNDED",
  "answer": "Licensor's total cumulative liability shall not exceed $50,000",
  "citation": "LICENSOR'S TOTAL CUMULATIVE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT SHALL NOT EXCEED FIFTY THOUSAND DOLLARS ($50,000)",
  "reasoning": "The contract explicitly includes a $50,000 liability cap in Section 5."
}
```

---

## Architecture

```
CUAD Dataset (HuggingFace: theatticusproject/cuad)
  84,325 rows · 510 contracts · 41 clause categories
       │
       ▼
data/01_download_cuad.py ──────────► data/raw/cuad_raw.jsonl
       │
       ▼
data/02_perturb_and_generate.py ───► data/perturbed/labeled.jsonl
  REMOVE     → ABSENT              (delete answer span from contract)
  INVERT     → CONTRADICTS_PRIOR   (negate "shall", flip numbers ±20%, swap Party A/B)
  CONTRADICT → CONTRADICTS_PRIOR   (replace span with one from a different contract)
  Target distribution: 40% GROUNDED · 40% ABSENT · 20% CONTRADICTS_PRIOR · seed=42
       │
       ▼
data/03_build_dataset.py ──────────► data/final/{train,val,test}.jsonl
  Stratified 80/10/10 split · Deduplicated on (contract_id, question, label)
       │
       ▼
training/train.py (LoRA r=16 on Qwen/Qwen3.5-9B, AMD MI300X ROCm)
  4-bit NF4 quantization (bitsandbytes) · fp16 fallback for ROCm
  SFTTrainer · 3 epochs · Per-class accuracy logged per epoch
       │
       ▼
checkpoints/final/ ────────────────► HuggingFace Hub (optional)
       │
       ├──► eval/benchmark.py ──────► Hallucination rate · Per-class F1 · Citation accuracy
       │
       └──► demo/app.py ────────────► Gradio Space (2 tabs: Analyze + Benchmark)
```

---

## Setup

### Standard GPU (NVIDIA)

```bash
# Clone the repository
git clone https://github.com/Abrar5510/CHEX
cd CHEX

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your HF_TOKEN, WANDB_API_KEY, etc.
```

### AMD ROCm (MI300X / MI250X)

```bash
# Step 1: Install ROCm PyTorch first
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1

# Step 2: Install remaining dependencies
pip install -r requirements_rocm.txt

# Optional: ROCm bitsandbytes (experimental)
pip install bitsandbytes \
    --extra-index-url https://huggingface.github.io/bitsandbytes-rocm/

cp .env.example .env
```

---

## Pipeline

Run these steps in order:

```bash
# 1. Download CUAD dataset (~84k rows, ~500 contracts)
python data/01_download_cuad.py \
    --output data/raw/cuad_raw.jsonl

# 2. Generate labeled dataset (deterministic, no model calls)
#    Add --use_contradicts_prior to enable 3-class mode
python data/02_perturb_and_generate.py \
    --input data/raw/cuad_raw.jsonl \
    --output data/perturbed/labeled.jsonl \
    --use_contradicts_prior \
    --seed 42

# 3. Build stratified train/val/test splits
python data/03_build_dataset.py \
    --input data/perturbed/labeled.jsonl \
    --output_dir data/final

# 4. Fine-tune on AMD MI300X / NVIDIA GPU
python training/train.py \
    --config training/config.yaml \
    --train_data data/final/train.jsonl \
    --val_data data/final/val.jsonl

# 5. Run benchmark evaluation
python eval/benchmark.py \
    --model_path ./checkpoints/final \
    --data_path data/final/test.jsonl \
    --output_path eval/results/benchmark_results.json

# 6. Side-by-side baseline comparison
python eval/compare_baseline.py \
    --finetuned_model_path ./checkpoints/final \
    --output_path eval/results/comparison.md

# 7. Launch Gradio demo locally
python demo/app.py
```

---

## Benchmark Results

| Model | Hallucination Rate ↓ | GROUNDED F1 | ABSENT F1 | CONTRADICTS_PRIOR F1 |
|-------|---------------------|-------------|-----------|----------------------|
| Qwen3.5-9B (base, zero-shot) | TBD | TBD | TBD | TBD |
| CHEX (fine-tuned, LoRA r=16) | TBD | TBD | TBD | TBD |

*Results to be populated after training on AMD MI300X hardware.*

**Hallucination rate** = fraction of examples where the model predicts GROUNDED when the ground truth is ABSENT or CONTRADICTS_PRIOR.

---

## HuggingFace Space

[https://huggingface.co/spaces/PLACEHOLDER/chex-demo](https://huggingface.co/spaces/PLACEHOLDER/chex-demo)

---

## Project Structure

```
CHEX/
├── data/
│   ├── schema.py                   # Pydantic v2 models (Label, LabeledQAExample, etc.)
│   ├── 01_download_cuad.py         # Download CUAD from HuggingFace
│   ├── 02_perturb_and_generate.py  # Generate labeled dataset (no model calls)
│   └── 03_build_dataset.py         # Dedup + stratified split
├── training/
│   ├── config.yaml                 # All hyperparameters
│   ├── prompt_template.py          # Few-shot system prompt + format functions
│   └── train.py                    # LoRA fine-tuning with ROCm support
├── eval/
│   ├── metrics.py                  # Hallucination rate, F1, citation accuracy
│   ├── benchmark.py                # Full test-set evaluation
│   └── compare_baseline.py         # Base vs fine-tuned comparison table
├── serving/
│   └── inference.py                # ContractAnalyzer class (HF pipeline)
├── demo/
│   ├── app.py                      # Gradio Blocks app (2 tabs)
│   └── sample_contracts/           # 3 demo contracts (.txt)
├── requirements.txt
├── requirements_rocm.txt
├── .env.example
└── README.md
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## AMD Fine-Tuning Hackathon — Track 2

Built for the AMD Fine-Tuning Hackathon. Fine-tuning performed on AMD MI300X GPU using ROCm. All data generation is fully local and deterministic (no external AI API calls). The complete pipeline from raw CUAD data to a deployed Gradio Space is reproducible end-to-end with `seed=42`.

---

*CHEX - Document Intelligence*
