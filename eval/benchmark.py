"""
Run full benchmark evaluation on the test split.

Loads the fine-tuned model via ContractAnalyzer, runs inference on every test
example, computes all metrics, saves results to JSON, and prints a report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import BenchmarkResult, Label, LabeledQAExample, ModelOutput
from eval.metrics import (
    compute_citation_accuracy,
    compute_hallucination_rate,
    compute_overall_accuracy,
    compute_per_class_metrics,
    format_benchmark_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CHEX fine-tuned model on the test split."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint directory or HF Hub repo ID",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/final/test.jsonl"),
        help="Path to test JSONL (default: data/final/test.jsonl)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("eval/results/benchmark_results.json"),
        help="Where to save results JSON (default: eval/results/benchmark_results.json)",
    )
    parser.add_argument(
        "--use_contradicts_prior",
        action="store_true",
        default=False,
        help="Include CONTRADICTS_PRIOR in evaluation (default: 2-class GROUNDED/ABSENT only)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap number of test examples (useful for quick runs)",
    )
    return parser.parse_args()


def load_test_examples(
    path: Path,
    use_contradicts_prior: bool,
    max_examples: int | None,
) -> list[LabeledQAExample]:
    examples: list[LabeledQAExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ex = LabeledQAExample.model_validate_json(line)
            if not use_contradicts_prior and ex.label == Label.CONTRADICTS_PRIOR:
                continue
            examples.append(ex)
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def run_benchmark(
    model_path: str,
    data_path: Path,
    output_path: Path,
    use_contradicts_prior: bool,
    max_examples: int | None,
) -> None:
    from serving.inference import ContractAnalyzer  # type: ignore

    print(f"Loading model from: {model_path}")
    analyzer = ContractAnalyzer(model_path=model_path)

    print(f"\nLoading test examples from {data_path}...")
    examples = load_test_examples(data_path, use_contradicts_prior, max_examples)
    print(f"Evaluating {len(examples)} examples")
    if not use_contradicts_prior:
        print("  (CONTRADICTS_PRIOR examples excluded — use --use_contradicts_prior to include)")

    benchmark_results: list[BenchmarkResult] = []
    predictions: list[Label] = []
    ground_truths: list[Label] = []
    model_outputs: list[ModelOutput] = []
    contracts: list[str] = []
    latencies: list[float] = []

    for i, ex in enumerate(examples):
        t0 = time.perf_counter()
        output = analyzer.analyze(ex.contract_text, ex.question)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        citation_present = (
            output.citation is not None
            and output.citation.strip() != ""
            and output.citation in ex.contract_text
        )

        result = BenchmarkResult(
            example_id=ex.contract_id,
            predicted_label=output.label,
            ground_truth_label=ex.label,
            predicted_answer=output.answer,
            ground_truth_answer=ex.answer,
            citation_present=citation_present,
            latency_ms=elapsed_ms,
        )
        benchmark_results.append(result)
        predictions.append(output.label)
        ground_truths.append(ex.label)
        model_outputs.append(output)
        contracts.append(ex.contract_text)
        latencies.append(elapsed_ms)

        if (i + 1) % 50 == 0 or (i + 1) == len(examples):
            print(f"  [{i+1}/{len(examples)}] processed...")

    # Compute metrics
    hallucination_rate = compute_hallucination_rate(predictions, ground_truths)
    overall_accuracy = compute_overall_accuracy(predictions, ground_truths)
    per_class = compute_per_class_metrics(predictions, ground_truths)
    citation_acc = compute_citation_accuracy(model_outputs, examples, contracts)
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    metrics = {
        "total_examples": len(examples),
        "overall_accuracy": overall_accuracy,
        "hallucination_rate": hallucination_rate,
        "citation_accuracy": citation_acc,
        "latency_ms_mean": mean_latency,
        "per_class": per_class,
        "model_path": model_path,
        "use_contradicts_prior": use_contradicts_prior,
    }

    # Save full results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output = {
        "metrics": metrics,
        "results": [r.model_dump() for r in benchmark_results],
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(full_output, fh, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print report
    print("\n" + format_benchmark_report(metrics))


def main() -> None:
    args = parse_args()
    run_benchmark(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        use_contradicts_prior=args.use_contradicts_prior,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
