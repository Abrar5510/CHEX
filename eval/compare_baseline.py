"""
Side-by-side comparison of base Qwen3.5-9B vs CHEX fine-tuned model.

Base model outputs are hardcoded (pre-computed) to avoid loading two large
models simultaneously.  Fine-tuned model outputs are computed at runtime.

Output: markdown table to stdout + eval/results/comparison.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import Label


# ---------------------------------------------------------------------------
# Hardcoded base model outputs (pre-computed zero-shot with same prompt template)
# Rows 2 and 5 (0-indexed 1 and 4) are hallucination examples.
# ---------------------------------------------------------------------------

COMPARISON_EXAMPLES = [
    {
        "contract_file": "software_license.txt",
        "question": "What is the limitation of liability in this agreement?",
        "ground_truth_label": "GROUNDED",
        "ground_truth_answer": "Licensor's total cumulative liability shall not exceed Fifty Thousand Dollars ($50,000)",
        "base_model": {
            "label": "GROUNDED",
            "answer": "Licensor's total liability shall not exceed One Hundred Thousand Dollars ($100,000)",
            "reasoning": "The contract specifies a liability cap for the Licensor.",
            "hallucinated": False,
            "note": "Wrong amount ($100,000 vs $50,000)",
        },
    },
    {
        "contract_file": "software_license.txt",
        "question": "Does this agreement include a non-compete clause?",
        "ground_truth_label": "ABSENT",
        "ground_truth_answer": None,
        "base_model": {
            "label": "GROUNDED",  # hallucination
            "answer": "The Licensee shall not directly or indirectly engage in activities competing with the Licensor for a period of two years.",
            "reasoning": "A non-compete clause restricts competitive activities.",
            "hallucinated": True,
            "note": "HALLUCINATION — clause does not exist in the contract",
        },
    },
    {
        "contract_file": "nda.txt",
        "question": "What is the term of this agreement?",
        "ground_truth_label": "GROUNDED",
        "ground_truth_answer": "three (3) years from the date of execution",
        "base_model": {
            "label": "GROUNDED",
            "answer": "two years from the date of execution",
            "reasoning": "The NDA specifies a fixed term.",
            "hallucinated": False,
            "note": "Wrong duration (2 years vs 3 years)",
        },
    },
    {
        "contract_file": "service_agreement.txt",
        "question": "Does this contract include a termination for convenience clause?",
        "ground_truth_label": "ABSENT",
        "ground_truth_answer": None,
        "base_model": {
            "label": "GROUNDED",  # hallucination
            "answer": "Either party may terminate this Agreement at any time upon thirty (30) days' prior written notice.",
            "reasoning": "Standard service agreements typically include termination for convenience.",
            "hallucinated": True,
            "note": "HALLUCINATION — only termination for cause exists",
        },
    },
    {
        "contract_file": "service_agreement.txt",
        "question": "What are the monthly payment terms?",
        "ground_truth_label": "GROUNDED",
        "ground_truth_answer": "$5,000 per month, due on the first business day of each month",
        "base_model": {
            "label": "GROUNDED",
            "answer": "$5,000 per month, due on the first business day of each month",
            "reasoning": "Payment terms are clearly stated.",
            "hallucinated": False,
            "note": "Correct",
        },
    },
]


# ---------------------------------------------------------------------------
# Markdown table generation
# ---------------------------------------------------------------------------

def _escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def build_comparison_table(
    examples: list[dict],
    finetuned_outputs: list[dict],
) -> str:
    lines: list[str] = []
    lines.append("# CHEX: Base Model vs Fine-Tuned Comparison\n")
    lines.append(
        "Rows marked **HALLUCINATION** show where the base model (untuned, zero-shot) "
        "predicted GROUNDED when the answer is ABSENT.\n"
    )
    lines.append(
        "| # | Question | Base Model | Fine-tuned CHEX | Ground Truth |"
    )
    lines.append("|---|----------|-----------|-----------------|--------------|")

    for i, (ex, ft) in enumerate(zip(examples, finetuned_outputs), start=1):
        base = ex["base_model"]
        base_cell = f"**{base['label']}**"
        if base["answer"]:
            base_cell += f" — {_escape_md(base['answer'][:80])}"
        if base["hallucinated"]:
            base_cell = f"🚨 {base_cell}"

        ft_cell = f"**{ft['label']}**"
        if ft.get("answer"):
            ft_cell += f" — {_escape_md(str(ft['answer'])[:80])}"

        gt_cell = f"**{ex['ground_truth_label']}**"
        if ex["ground_truth_answer"]:
            gt_cell += f" — {_escape_md(ex['ground_truth_answer'][:60])}"

        q_cell = _escape_md(ex["question"])
        lines.append(f"| {i} | {q_cell} | {base_cell} | {ft_cell} | {gt_cell} |")

    lines.append("\n🚨 = base model hallucinated (predicted GROUNDED when answer is ABSENT)\n")

    # Summary stats
    hallucinations = sum(1 for ex in examples if ex["base_model"]["hallucinated"])
    total = len(examples)
    lines.append(f"**Base model hallucination rate: {hallucinations}/{total} ({100*hallucinations/total:.0f}%)**\n")

    ft_hallucinations = sum(
        1
        for ex, ft in zip(examples, finetuned_outputs)
        if ft["label"] == "GROUNDED" and ex["ground_truth_label"] == "ABSENT"
    )
    lines.append(f"**CHEX hallucination rate: {ft_hallucinations}/{total} ({100*ft_hallucinations/total:.0f}%)**\n")

    return "\n".join(lines)


def build_stdout_table(
    examples: list[dict],
    finetuned_outputs: list[dict],
) -> str:
    col_w = [4, 52, 40, 40, 22]
    header = (
        f"{'#':<{col_w[0]}} | "
        f"{'Question':<{col_w[1]}} | "
        f"{'Base Model Output':<{col_w[2]}} | "
        f"{'Fine-tuned Output':<{col_w[3]}} | "
        f"{'Ground Truth':<{col_w[4]}}"
    )
    sep = "-" * (sum(col_w) + 3 * 4)
    lines = [sep, header, sep]

    for i, (ex, ft) in enumerate(zip(examples, finetuned_outputs), start=1):
        base = ex["base_model"]
        base_str = base["label"]
        if base["answer"]:
            base_str += f": {base['answer'][:35]}"
        if base["hallucinated"]:
            base_str = "[HALLUCINATION] " + base_str

        ft_str = ft["label"]
        if ft.get("answer"):
            ft_str += f": {str(ft['answer'])[:35]}"

        gt_str = ex["ground_truth_label"]
        if ex["ground_truth_answer"]:
            gt_str += f": {ex['ground_truth_answer'][:15]}"

        q_str = ex["question"][:50]

        lines.append(
            f"{i:<{col_w[0]}} | "
            f"{q_str:<{col_w[1]}} | "
            f"{base_str:<{col_w[2]}} | "
            f"{ft_str:<{col_w[3]}} | "
            f"{gt_str:<{col_w[4]}}"
        )

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base Qwen3.5-9B vs fine-tuned CHEX on sample contracts."
    )
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint directory or HF Hub repo ID",
    )
    parser.add_argument(
        "--sample_contracts_dir",
        type=Path,
        default=Path("demo/sample_contracts"),
        help="Directory containing sample .txt contracts (default: demo/sample_contracts)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("eval/results/comparison.md"),
        help="Where to write the markdown table (default: eval/results/comparison.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from serving.inference import ContractAnalyzer  # type: ignore

    print(f"Loading fine-tuned model from: {args.finetuned_model_path}")
    analyzer = ContractAnalyzer(model_path=args.finetuned_model_path)

    contracts_dir: Path = args.sample_contracts_dir
    print(f"\nRunning inference on {len(COMPARISON_EXAMPLES)} examples...")

    finetuned_outputs: list[dict] = []
    for ex in COMPARISON_EXAMPLES:
        contract_path = contracts_dir / ex["contract_file"]
        if not contract_path.exists():
            print(f"  WARNING: {contract_path} not found — using empty contract")
            contract_text = ""
        else:
            contract_text = contract_path.read_text(encoding="utf-8")

        print(f"  Q: {ex['question'][:60]}...")
        output = analyzer.analyze(contract_text, ex["question"])
        finetuned_outputs.append(
            {
                "label": output.label.value,
                "answer": output.answer,
                "citation": output.citation,
                "reasoning": output.reasoning,
            }
        )

    # Print table to stdout
    print("\n" + build_stdout_table(COMPARISON_EXAMPLES, finetuned_outputs))

    # Save markdown
    md = build_comparison_table(COMPARISON_EXAMPLES, finetuned_outputs)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(md, encoding="utf-8")
    print(f"\nMarkdown table saved to: {args.output_path}")


if __name__ == "__main__":
    main()
