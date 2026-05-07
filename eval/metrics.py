"""
Evaluation metrics for CHEX.

Functions:
  compute_hallucination_rate  — fraction of GROUNDED predictions when truth is not GROUNDED
  compute_per_class_metrics   — precision / recall / F1 per label class
  compute_citation_accuracy   — fraction of GROUNDED predictions with valid citation substring
  format_benchmark_report     — pretty-print ASCII table of all metrics
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import Label, LabeledQAExample, ModelOutput


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_hallucination_rate(
    predictions: list[Label],
    ground_truths: list[Label],
) -> float:
    """
    Hallucination rate: fraction of examples where the model predicts GROUNDED
    but the ground truth is ABSENT or CONTRADICTS_PRIOR.
    """
    if not predictions:
        return 0.0
    hallucinations = sum(
        1
        for p, g in zip(predictions, ground_truths)
        if p == Label.GROUNDED and g != Label.GROUNDED
    )
    return hallucinations / len(predictions)


def compute_per_class_metrics(
    predictions: list[Label],
    ground_truths: list[Label],
) -> dict[str, dict[str, float]]:
    """
    Returns {class_name: {"precision": float, "recall": float, "f1": float}}
    for every label class that appears in ground_truths.
    """
    all_labels = sorted(set(g.value for g in ground_truths) | set(p.value for p in predictions))
    results: dict[str, dict[str, float]] = {}

    for lbl in all_labels:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p.value == lbl and g.value == lbl)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p.value == lbl and g.value != lbl)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p.value != lbl and g.value == lbl)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[lbl] = {"precision": precision, "recall": recall, "f1": f1}

    return results


def compute_citation_accuracy(
    predictions: list[ModelOutput],
    ground_truths: list[LabeledQAExample],
    contracts: list[str],
) -> float:
    """
    For examples where the model predicted GROUNDED:
    fraction where the predicted citation is a non-empty substring of the contract.
    """
    grounded_pairs = [
        (p, contract)
        for p, g, contract in zip(predictions, ground_truths, contracts)
        if p.label == Label.GROUNDED
    ]
    if not grounded_pairs:
        return 0.0

    correct = sum(
        1
        for p, contract in grounded_pairs
        if p.citation and p.citation.strip() and p.citation in contract
    )
    return correct / len(grounded_pairs)


def compute_overall_accuracy(
    predictions: list[Label],
    ground_truths: list[Label],
) -> float:
    if not predictions:
        return 0.0
    return sum(p == g for p, g in zip(predictions, ground_truths)) / len(predictions)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_benchmark_report(results: dict) -> str:
    """
    Format a benchmark results dict into a readable ASCII table.

    Expected keys in results:
      hallucination_rate, overall_accuracy, citation_accuracy,
      per_class (dict of class → {precision, recall, f1}),
      total_examples, latency_ms_mean
    """
    lines: list[str] = []
    sep = "=" * 62

    lines.append(sep)
    lines.append("  CHEX Benchmark Report")
    lines.append(sep)

    total = results.get("total_examples", 0)
    lines.append(f"  Total examples       : {total}")
    lines.append(f"  Overall accuracy     : {results.get('overall_accuracy', 0.0):.3f}")
    lines.append(f"  Hallucination rate   : {results.get('hallucination_rate', 0.0):.3f}")
    lines.append(f"  Citation accuracy    : {results.get('citation_accuracy', 0.0):.3f}")
    lines.append(f"  Mean latency (ms)    : {results.get('latency_ms_mean', 0.0):.1f}")
    lines.append("")

    # Per-class table
    per_class = results.get("per_class", {})
    if per_class:
        header = f"  {'Class':<25}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
        lines.append(header)
        lines.append("  " + "-" * 48)
        for lbl in [Label.GROUNDED.value, Label.ABSENT.value, Label.CONTRADICTS_PRIOR.value]:
            if lbl not in per_class:
                continue
            m = per_class[lbl]
            lines.append(
                f"  {lbl:<25}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['f1']:>6.3f}"
            )

    lines.append(sep)
    return "\n".join(lines)
