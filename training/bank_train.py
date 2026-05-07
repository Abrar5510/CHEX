"""
Fine-tune Qwen/Qwen3.5-9B with LoRA on the bank statement QA dataset.

Fully self-contained — does not modify or depend on training/train.py.
Uses the same infrastructure (LoRA, SFTTrainer, bitsandbytes, ROCm detection)
but with the bank-statement system prompt from training/prompt_template.py.

Features:
  - 4-bit NF4 quantization via bitsandbytes (when available)
  - Automatic ROCm detection with fp16 fallback
  - LoRA via PEFT on q/k/v/o projection layers
  - SFTTrainer (trl >= 0.11) with SFTConfig
  - Per-class accuracy logged after each epoch (GROUNDED / ABSENT)
  - Best checkpoint saved by val accuracy
  - Optional push to HF Hub (HF_TOKEN env var)
  - Optional wandb logging (WANDB_API_KEY env var)

Usage:
    # Step 1 — build dataset (one-time)
    python data/bank_statement/01_build_dataset.py
    python data/bank_statement/02_split_dataset.py

    # Step 2 — train
    python training/bank_train.py

    # With custom paths / resume
    python training/bank_train.py \\
        --config training/bank_config.yaml \\
        --train_data data/bank_statement/final/train.jsonl \\
        --val_data   data/bank_statement/final/val.jsonl \\
        --resume_from_checkpoint ./checkpoints_bank/checkpoint-500
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import Label, LabeledQAExample
from training.prompt_template import (
    BANK_STATEMENT_SYSTEM_PROMPT,
    build_bank_chat_messages,
)


# ---------------------------------------------------------------------------
# Environment detection  (identical logic to train.py, kept self-contained)
# ---------------------------------------------------------------------------

def detect_environment() -> tuple[str, bool, bool]:
    if not torch.cuda.is_available():
        print("WARNING: No CUDA device found. Training on CPU (very slow).")
        return "cpu", False, False

    device_name = torch.cuda.get_device_name(0)
    is_rocm = "AMD" in device_name or "gfx" in device_name.lower()
    print(f"{'AMD' if is_rocm else 'NVIDIA'} GPU detected: {device_name}")

    bnb_available = importlib.util.find_spec("bitsandbytes") is not None
    if not bnb_available:
        print("WARNING: bitsandbytes not available — falling back to fp16.")

    return "cuda", is_rocm, bnb_available


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, is_rocm: bool, bnb_available: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model: {model_name}")
    if bnb_available:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        print("Model loaded with 4-bit NF4 quantization")
    else:
        dtype = torch.float16 if is_rocm else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Model loaded in {'fp16' if is_rocm else 'bf16'} (no quantization)")

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def apply_lora(model, config: dict, bnb_available: bool):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore

    if bnb_available:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Dataset loading + prompt building
# ---------------------------------------------------------------------------

def load_split(path: Path) -> list[LabeledQAExample]:
    examples: list[LabeledQAExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(LabeledQAExample.model_validate_json(line))
    return examples


def _build_training_text(ex: LabeledQAExample, tokenizer, eos_token: str) -> str:
    """Build the full ChatML training string for one bank statement example."""
    from data.schema import ModelOutput

    messages = build_bank_chat_messages(ex.contract_text, ex.question)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    output = ModelOutput(
        question=ex.question,
        label=ex.label,
        answer=ex.answer,
        citation=ex.citation,
        reasoning=ex.reasoning,
    )
    return prompt + output.model_dump_json() + eos_token


def build_hf_dataset(examples: list[LabeledQAExample], tokenizer, config: dict):
    from datasets import Dataset  # type: ignore

    eos = tokenizer.eos_token or "<|im_end|>"
    texts = [_build_training_text(ex, tokenizer, eos) for ex in examples]
    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# Per-class accuracy callback
# ---------------------------------------------------------------------------

class BankAccuracyCallback:
    """Evaluate per-class accuracy on the bank statement validation set."""

    def __init__(
        self,
        val_examples: list[LabeledQAExample],
        tokenizer,
        model,
    ) -> None:
        self.val_examples = val_examples
        self.tokenizer = tokenizer
        self.model = model
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def evaluate(self, epoch: int) -> float:
        from transformers import pipeline as hf_pipeline  # type: ignore

        print(f"\n--- Epoch {epoch} bank statement validation ---")
        pipe = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=False,
            return_full_text=False,
        )

        predictions: list[str] = []
        ground_truths: list[str] = []

        for ex in self.val_examples[:200]:
            messages = build_bank_chat_messages(
                ex.contract_text[:2000], ex.question
            )
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            try:
                result = pipe(prompt)
                raw = result[0]["generated_text"]
                m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
                pred = m.group(1) if m else "ABSENT"
            except Exception:
                pred = "ABSENT"

            predictions.append(pred)
            ground_truths.append(ex.label.value)

        correct = sum(p == g for p, g in zip(predictions, ground_truths))
        overall = correct / len(predictions) if predictions else 0.0
        counts = Counter(ground_truths)

        print(f"Overall accuracy: {overall:.3f} ({correct}/{len(predictions)})")
        for lbl in [Label.GROUNDED, Label.ABSENT]:
            total_lbl = counts.get(lbl.value, 0)
            if total_lbl == 0:
                continue
            correct_lbl = sum(
                1 for p, g in zip(predictions, ground_truths)
                if g == lbl.value and p == lbl.value
            )
            print(f"  {lbl.value:<22}: {correct_lbl}/{total_lbl}  ({100*correct_lbl/total_lbl:.1f}%)")

        if overall > self.best_accuracy:
            self.best_accuracy = overall
            self.best_epoch = epoch
            print(f"  *** New best: {overall:.3f} at epoch {epoch} ***")

        return overall


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    config_path: Path,
    train_data_path: Path,
    val_data_path: Path,
    resume_from_checkpoint: Optional[str],
) -> None:
    from trl import SFTConfig, SFTTrainer  # type: ignore

    with config_path.open("r") as fh:
        config = yaml.safe_load(fh)

    device, is_rocm, bnb_available = detect_environment()

    model, tokenizer = load_model_and_tokenizer(
        config["model_name"], is_rocm, bnb_available
    )
    model = apply_lora(model, config, bnb_available)

    print(f"\nLoading training data from {train_data_path}...")
    train_examples = load_split(train_data_path)
    print(f"Loading validation data from {val_data_path}...")
    val_examples = load_split(val_data_path)
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Val  : {len(val_examples)} examples")

    train_dataset = build_hf_dataset(train_examples, tokenizer, config)
    val_dataset = build_hf_dataset(val_examples, tokenizer, config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    print(f"Logging to: {'wandb' if report_to == 'wandb' else 'stdout'}")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        bf16=config.get("bf16", True) and not is_rocm,
        fp16=is_rocm,
        max_seq_length=config["max_seq_length"],
        dataset_text_field="text",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=config.get("save_total_limit", 3),
        report_to=report_to,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit" if bnb_available else "adamw_torch",
    )

    acc_callback = BankAccuracyCallback(
        val_examples=val_examples,
        tokenizer=tokenizer,
        model=model,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class EpochEndCallback(TrainerCallback):
        def __init__(self, cb: BankAccuracyCallback) -> None:
            self.cb = cb

        def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ) -> None:
            self.cb.evaluate(int(state.epoch))

    trainer.add_callback(EpochEndCallback(acc_callback))

    print(f"\nStarting bank statement fine-tuning for {config['num_epochs']} epochs...")
    start_time = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Best val accuracy: {acc_callback.best_accuracy:.3f} at epoch {acc_callback.best_epoch}")

    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Final model saved to: {final_path}")

    hf_token = os.environ.get("HF_TOKEN")
    hub_model_id = config.get("hub_model_id", "")
    if hf_token and hub_model_id and "PLACEHOLDER" not in hub_model_id:
        print(f"\nPushing model to HuggingFace Hub: {hub_model_id}")
        trainer.model.push_to_hub(hub_model_id, token=hf_token)
        tokenizer.push_to_hub(hub_model_id, token=hf_token)
        print(f"Model pushed to: https://huggingface.co/{hub_model_id}")
    else:
        print("\nSkipping Hub push (set HF_TOKEN and hub_model_id in bank_config.yaml).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen/Qwen3.5-9B on bank statement QA data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/bank_config.yaml"),
        help="Training config YAML (default: training/bank_config.yaml)",
    )
    parser.add_argument(
        "--train_data",
        type=Path,
        default=Path("data/bank_statement/final/train.jsonl"),
        help="Training JSONL",
    )
    parser.add_argument(
        "--val_data",
        type=Path,
        default=Path("data/bank_statement/final/val.jsonl"),
        help="Validation JSONL",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory to resume from",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        config_path=args.config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
