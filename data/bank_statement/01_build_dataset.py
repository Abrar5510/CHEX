"""
Bank Statement Training Data Builder — Stage 1

Downloads and combines multiple HuggingFace datasets to create a bank-statement
QA training corpus in LabeledQAExample format (same schema as contract training).

Datasets used:
  1. virattt/financial-qa-10K  — grounded Q&A over financial documents (7k rows)
  2. karthiksagarn/bank-statement-categorization — transaction→category pairs (7.5k rows)
     Converted to GROUNDED/ABSENT Q&A examples by generating questions.

Output: data/bank_statement/raw/combined.jsonl

Usage:
    python data/bank_statement/01_build_dataset.py [--hf_token YOUR_TOKEN]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.schema import Label, LabeledQAExample

OUT_DIR = Path(__file__).parent / "raw"
OUT_FILE = OUT_DIR / "combined.jsonl"

SEED = 42
random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset 1: virattt/financial-qa-10K
# Columns: question, answer, context, ticker, filing
# We treat these as GROUNDED examples (answer exists in context).
# ---------------------------------------------------------------------------

def load_financial_qa(hf_token: str | None) -> list[LabeledQAExample]:
    from datasets import load_dataset  # type: ignore

    print("Loading virattt/financial-qa-10K...")
    ds = load_dataset("virattt/financial-qa-10K", split="train", token=hf_token)
    print(f"  Loaded {len(ds)} rows")

    examples: list[LabeledQAExample] = []
    for i, row in enumerate(ds):
        context = (row.get("context") or "").strip()
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not context or not question or not answer:
            continue

        # All rows in this dataset are grounded (answer is in context)
        examples.append(
            LabeledQAExample(
                contract_id=f"finqa_{i}",
                contract_text=context,
                question=question,
                label=Label.GROUNDED,
                answer=answer,
                citation=_find_citation(context, answer),
                reasoning="The answer is explicitly stated in the financial document.",
            )
        )

    print(f"  Built {len(examples)} GROUNDED examples from financial-qa-10K")
    return examples


def _find_citation(text: str, answer: str) -> str | None:
    """Return the answer itself as citation if it appears verbatim in context."""
    if answer.lower() in text.lower():
        return answer
    # Try first 8 words of answer
    words = answer.split()[:8]
    snippet = " ".join(words)
    if snippet.lower() in text.lower():
        return snippet
    return None


# ---------------------------------------------------------------------------
# Dataset 2: karthiksagarn/bank-statement-categorization
# Columns: description (transaction text), category (label)
# Strategy: generate GROUNDED and ABSENT Q&A pairs from transaction descriptions.
# ---------------------------------------------------------------------------

CATEGORY_QUESTIONS = {
    "Income":              "Is there any salary or income payment in this statement?",
    "Education":           "Are there any education or tuition payments?",
    "Travel & Transport":  "Are there any travel or transport charges?",
    "Groceries":           "Are there any grocery or supermarket payments?",
    "Bills & Utilities":   "Are there any utility bill payments?",
    "Entertainment":       "Are there any entertainment expenses?",
    "Health & Fitness":    "Are there any health or fitness payments?",
    "Shopping":            "Are there any retail shopping transactions?",
    "Food & Drinks":       "Are there any food or restaurant payments?",
    "Investments":         "Are there any investment transactions?",
    "Miscellaneous":       "Are there any miscellaneous charges?",
    "Withdrawals":         "Are there any ATM or cash withdrawals?",
}

ABSENT_CATEGORY_PAIRS: list[tuple[str, str]] = [
    ("Income",             "Education"),
    ("Groceries",          "Investments"),
    ("Travel & Transport", "Health & Fitness"),
    ("Entertainment",      "Bills & Utilities"),
    ("Shopping",           "Withdrawals"),
    ("Food & Drinks",      "Travel & Transport"),
]


def load_bank_categorization(hf_token: str | None) -> list[LabeledQAExample]:
    from datasets import load_dataset  # type: ignore

    print("Loading karthiksagarn/bank-statement-categorization...")
    ds = load_dataset("karthiksagarn/bank-statement-categorization", split="train", token=hf_token)
    print(f"  Loaded {len(ds)} rows")

    # Group transactions by category
    by_category: dict[str, list[str]] = {}
    for row in ds:
        cat = (row.get("category") or "").strip()
        desc = (row.get("description") or "").strip()
        if cat and desc:
            by_category.setdefault(cat, []).append(desc)

    examples: list[LabeledQAExample] = []
    example_id = 0

    # GROUNDED: bundle a few transactions of the same category into a mini-statement,
    # then ask the category question.
    for category, descs in by_category.items():
        question = CATEGORY_QUESTIONS.get(category)
        if not question:
            continue
        random.shuffle(descs)
        # Make batches of 5–8 transactions each
        batch_size = random.randint(5, 8)
        for start in range(0, min(len(descs), 500), batch_size):
            batch = descs[start : start + batch_size]
            statement = _make_mini_statement(batch)
            citation = batch[0]  # first transaction as citation
            examples.append(
                LabeledQAExample(
                    contract_id=f"bankcat_grounded_{example_id}",
                    contract_text=statement,
                    question=question,
                    label=Label.GROUNDED,
                    answer=f"Yes — found transaction: {citation}",
                    citation=citation,
                    reasoning=f"The statement contains a {category} transaction.",
                )
            )
            example_id += 1

    # ABSENT: bundle transactions from category A, ask question for category B
    for present_cat, absent_cat in ABSENT_CATEGORY_PAIRS:
        present_descs = by_category.get(present_cat, [])
        absent_question = CATEGORY_QUESTIONS.get(absent_cat)
        if not present_descs or not absent_question:
            continue
        random.shuffle(present_descs)
        batch_size = random.randint(5, 8)
        for start in range(0, min(len(present_descs), 200), batch_size):
            batch = present_descs[start : start + batch_size]
            statement = _make_mini_statement(batch)
            examples.append(
                LabeledQAExample(
                    contract_id=f"bankcat_absent_{example_id}",
                    contract_text=statement,
                    question=absent_question,
                    label=Label.ABSENT,
                    answer=None,
                    citation=None,
                    reasoning=f"No {absent_cat} transactions appear in the statement.",
                )
            )
            example_id += 1

    print(f"  Built {len(examples)} examples from bank-statement-categorization")
    return examples


def _make_mini_statement(descriptions: list[str]) -> str:
    """Format a list of transaction descriptions as a simple bank statement block."""
    lines = ["Date       Description                              Amount"]
    lines.append("-" * 60)
    for i, desc in enumerate(descriptions):
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        amount = round(random.uniform(5, 500), 2)
        lines.append(f"{day:02d}/{month:02d}/2025  {desc:<40}  £{amount:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic ABSENT examples for financial-qa-10K
# Add ABSENT variants by asking questions whose answers don't appear in context.
# ---------------------------------------------------------------------------

ABSENT_QUESTION_TEMPLATES = [
    "What was the total ATM cash withdrawal this period?",
    "Are there any mortgage payments in this document?",
    "What recurring subscription charges are listed?",
    "Is there a pension contribution mentioned?",
    "What is the opening balance of the account?",
]


def generate_absent_variants(grounded: list[LabeledQAExample]) -> list[LabeledQAExample]:
    """For a random sample of grounded contexts, ask unrelated questions → ABSENT."""
    sample = random.sample(grounded, min(1500, len(grounded)))
    examples: list[LabeledQAExample] = []
    for i, ex in enumerate(sample):
        q = ABSENT_QUESTION_TEMPLATES[i % len(ABSENT_QUESTION_TEMPLATES)]
        examples.append(
            LabeledQAExample(
                contract_id=f"finqa_absent_{i}",
                contract_text=ex.contract_text,
                question=q,
                label=Label.ABSENT,
                answer=None,
                citation=None,
                reasoning="The requested information does not appear in this financial document.",
            )
        )
    print(f"  Generated {len(examples)} synthetic ABSENT examples")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build bank statement training dataset.")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token")
    args = parser.parse_args()

    hf_token = args.hf_token or __import__("os").environ.get("HF_TOKEN")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grounded_finqa = load_financial_qa(hf_token)
    absent_finqa = generate_absent_variants(grounded_finqa)
    bank_cat = load_bank_categorization(hf_token)

    all_examples = grounded_finqa + absent_finqa + bank_cat
    random.shuffle(all_examples)

    # Label distribution
    from collections import Counter
    dist = Counter(ex.label.value for ex in all_examples)
    print(f"\nLabel distribution: {dict(dist)}")
    print(f"Total examples: {len(all_examples)}")

    with OUT_FILE.open("w", encoding="utf-8") as fh:
        for ex in all_examples:
            fh.write(ex.model_dump_json() + "\n")

    print(f"\nWrote {len(all_examples)} examples to {OUT_FILE}")


if __name__ == "__main__":
    main()
