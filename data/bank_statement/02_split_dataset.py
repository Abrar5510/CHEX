"""
Bank Statement Training Data Builder — Stage 2

Reads combined.jsonl, deduplicates, and produces stratified 80/10/10
train/val/test splits.

Output:
    data/bank_statement/final/train.jsonl
    data/bank_statement/final/val.jsonl
    data/bank_statement/final/test.jsonl

Usage:
    python data/bank_statement/02_split_dataset.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.schema import Label, LabeledQAExample

SEED = 42
random.seed(SEED)

RAW_FILE = Path(__file__).parent / "raw" / "combined.jsonl"
OUT_DIR = Path(__file__).parent / "final"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO = 0.10 (remainder)


def load_and_deduplicate(path: Path) -> list[LabeledQAExample]:
    examples: list[LabeledQAExample] = []
    seen: set[tuple[str, str, str]] = set()

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = LabeledQAExample.model_validate_json(line)
            key = (ex.contract_text[:200], ex.question, ex.label.value)
            if key not in seen:
                seen.add(key)
                examples.append(ex)

    print(f"Loaded {len(examples)} unique examples from {path}")
    return examples


def stratified_split(
    examples: list[LabeledQAExample],
) -> tuple[list[LabeledQAExample], list[LabeledQAExample], list[LabeledQAExample]]:
    by_label: dict[str, list[LabeledQAExample]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label.value].append(ex)

    train, val, test = [], [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])
        print(f"  {label:<22}: {n_train} train / {n_val} val / {n - n_train - n_val} test")

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def write_split(examples: list[LabeledQAExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.model_dump_json() + "\n")
    print(f"Wrote {len(examples)} examples to {path}")


def main() -> None:
    if not RAW_FILE.exists():
        print(f"ERROR: {RAW_FILE} not found. Run 01_build_dataset.py first.")
        sys.exit(1)

    examples = load_and_deduplicate(RAW_FILE)
    print(f"\nSplitting {len(examples)} examples (80/10/10)...")
    train, val, test = stratified_split(examples)

    write_split(train, OUT_DIR / "train.jsonl")
    write_split(val,   OUT_DIR / "val.jsonl")
    write_split(test,  OUT_DIR / "test.jsonl")

    print(f"\nDone. Total: {len(train)} train / {len(val)} val / {len(test)} test")


if __name__ == "__main__":
    main()
