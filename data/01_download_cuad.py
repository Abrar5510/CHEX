"""Download CUAD from HuggingFace and save raw examples to JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root: python data/01_download_cuad.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import RawCUADExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CUAD dataset from HuggingFace and save to JSONL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/cuad_raw.jsonl"),
        help="Destination JSONL file (default: data/raw/cuad_raw.jsonl)",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path(".cache"),
        help="HuggingFace datasets cache directory (default: .cache)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap the number of examples (useful for quick testing)",
    )
    return parser.parse_args()


def download_and_save(output: Path, cache_dir: Path, max_examples: int | None) -> None:
    from datasets import load_dataset  # type: ignore

    # theatticusproject/cuad returns PDF-only rows; theatticusproject/cuad-qa uses a
    # legacy dataset script no longer supported. chenghao/cuad_qa is a parquet mirror
    # with the same SQuAD-style schema (id, title, context, question, answers).
    print("Loading CUAD QA dataset from HuggingFace (chenghao/cuad_qa)...")
    ds = load_dataset(
        "chenghao/cuad_qa",
        split="train",
        cache_dir=str(cache_dir),
        verification_mode="no_checks",
    )
    first = ds[0]
    print(f"Row keys: {list(first.keys())}")

    print(f"Dataset loaded: {len(ds)} total rows")

    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with_answers = 0
    without_answers = 0
    contract_titles: set[str] = set()

    with output.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(ds):
            if max_examples is not None and i >= max_examples:
                break

            # Handle both nested-dict schema {"answers": {"text": [...], "answer_start": [...]}}
            # and flat schema with direct "answer" / "answer_start" columns.
            if "answers" in row:
                answers = row["answers"]
                answer_texts: list[str] = answers["text"]
                answer_starts: list[int] = answers["answer_start"]
                answer_text = answer_texts[0] if answer_texts else None
                answer_start = answer_starts[0] if answer_starts else None
            elif "answer" in row:
                answer_text = row["answer"] or None
                answer_start = row.get("answer_start") or None
            else:
                answer_text = None
                answer_start = None

            row_id = row.get("id") or row.get("uid") or str(i)
            context = row.get("context") or row.get("contract_text") or ""

            example = RawCUADExample(
                contract_id=row_id,
                contract_text=context,
                question=row.get("question", ""),
                answer_text=answer_text,
                answer_start=answer_start,
            )

            fh.write(example.model_dump_json() + "\n")
            total += 1

            if answer_text is not None:
                with_answers += 1
            else:
                without_answers += 1

            # Extract contract title from the id field (format: "ContractTitle__QuestionType__N")
            title = row_id.split("__")[0] if "__" in row_id else row_id
            contract_titles.add(title)

            if total % 5000 == 0:
                print(f"  Processed {total} rows...")

    print(f"\n--- Summary ---")
    print(f"Total examples written : {total}")
    print(f"  With answers (GROUNDED candidates) : {with_answers}")
    print(f"  Without answers (ABSENT candidates): {without_answers}")
    print(f"Unique contract titles              : {len(contract_titles)}")
    print(f"Output saved to: {output}")


def main() -> None:
    args = parse_args()
    download_and_save(
        output=args.output,
        cache_dir=args.cache_dir,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
