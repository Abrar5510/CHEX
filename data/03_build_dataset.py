"""
Deduplicate, stratify-split, and finalize the labeled dataset.

Splits: 80% train / 10% val / 10% test (stratified on label).
Optionally pushes to HuggingFace Hub if HF_TOKEN is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import Label, LabeledQAExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate and stratify-split the labeled CUAD dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/perturbed/labeled.jsonl"),
        help="Path to labeled JSONL from 02_perturb_and_generate.py",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/final"),
        help="Directory to write train/val/test JSONL files (default: data/final)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Push dataset to HuggingFace Hub (requires HF_TOKEN env var)",
    )
    parser.add_argument(
        "--hub_repo",
        type=str,
        default=None,
        help="HF Hub repo ID, e.g. your-username/cuad-hallucination-labeled",
    )
    return parser.parse_args()


def load_and_deduplicate(path: Path) -> list[LabeledQAExample]:
    """Load JSONL and deduplicate on (contract_id, question, label)."""
    seen: set[tuple[str, str, str]] = set()
    examples: list[LabeledQAExample] = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = LabeledQAExample.model_validate_json(line)
            key = (ex.contract_id, ex.question, ex.label.value)
            if key not in seen:
                seen.add(key)
                examples.append(ex)

    return examples


def stratified_split(
    examples: list[LabeledQAExample],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[LabeledQAExample], list[LabeledQAExample], list[LabeledQAExample]]:
    """Stratified split into train / val / test."""
    from sklearn.model_selection import train_test_split  # type: ignore

    labels = [ex.label.value for ex in examples]

    # Check minimum class counts
    counts = Counter(labels)
    for lbl, cnt in counts.items():
        if cnt < 3:
            raise ValueError(
                f"Class '{lbl}' has only {cnt} examples — too few for a 3-way stratified split."
            )

    test_ratio = 1.0 - train_ratio - val_ratio
    # First: hold out test set
    train_val, test, labels_tv, _ = train_test_split(
        examples,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )
    # Then: split train vs val
    relative_val = val_ratio / (train_ratio + val_ratio)
    train, val, _, _ = train_test_split(
        train_val,
        labels_tv,
        test_size=relative_val,
        stratify=labels_tv,
        random_state=seed,
    )
    return train, val, test


def print_distribution(name: str, examples: list[LabeledQAExample]) -> None:
    total = len(examples)
    counts = Counter(ex.label.value for ex in examples)
    print(f"\n{name} ({total} examples):")
    for lbl in [Label.GROUNDED, Label.ABSENT, Label.CONTRADICTS_PRIOR]:
        n = counts.get(lbl.value, 0)
        pct = 100.0 * n / total if total > 0 else 0.0
        print(f"  {lbl.value:<22}: {n:>6}  ({pct:.1f}%)")


def save_jsonl(examples: list[LabeledQAExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")
    print(f"  Saved {len(examples)} examples → {path}")


def push_to_hub(
    train: list[LabeledQAExample],
    val: list[LabeledQAExample],
    test: list[LabeledQAExample],
    hub_repo: str,
    hf_token: str,
) -> None:
    from datasets import Dataset, DatasetDict  # type: ignore

    print(f"\nPushing to HuggingFace Hub: {hub_repo}")
    hf_ds = DatasetDict(
        {
            "train": Dataset.from_list([ex.model_dump() for ex in train]),
            "validation": Dataset.from_list([ex.model_dump() for ex in val]),
            "test": Dataset.from_list([ex.model_dump() for ex in test]),
        }
    )
    hf_ds.push_to_hub(hub_repo, token=hf_token)
    print(f"Dataset pushed to: https://huggingface.co/datasets/{hub_repo}")


def main() -> None:
    args = parse_args()

    print(f"Loading labeled examples from {args.input}...")
    examples = load_and_deduplicate(args.input)
    print(f"Loaded {len(examples)} unique examples after deduplication")

    print("\nSplitting into train / val / test (80/10/10, stratified)...")
    train, val, test = stratified_split(examples)

    print_distribution("Train", train)
    print_distribution("Val  ", val)
    print_distribution("Test ", test)

    output_dir: Path = args.output_dir
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(val, output_dir / "val.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")

    if args.push_to_hub:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("\nWARNING: --push_to_hub requested but HF_TOKEN env var is not set. Skipping.")
        elif not args.hub_repo:
            print("\nWARNING: --push_to_hub requires --hub_repo. Skipping.")
        else:
            push_to_hub(train, val, test, args.hub_repo, hf_token)

    print("\nDone.")


if __name__ == "__main__":
    main()
