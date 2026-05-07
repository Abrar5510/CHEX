"""
Perturb CUAD examples and generate labeled QA dataset — fully deterministic, no model calls.

Perturbation strategies:
  REMOVE     → ABSENT         : surgically delete the answer span from the contract
  INVERT     → CONTRADICTS_PRIOR: negate legal terms, flip numbers ±20%, swap Party A/B
  CONTRADICT → CONTRADICTS_PRIOR: replace answer span with one from a different contract

Target class distribution: 40% GROUNDED, 40% ABSENT, 20% CONTRADICTS_PRIOR (seed=42)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import Label, LabeledQAExample, RawCUADExample


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

_UNIT_PATTERN = re.compile(
    r"(\b\d+(?:\.\d+)?)\s*(days?|months?|years?|hours?|weeks?|percent|%|dollars?|\$)",
    re.IGNORECASE,
)

_SHALL_NOT_PLACEHOLDER = "<<<SHALL_NOT_PLACEHOLDER>>>"
_PARTY_B_PLACEHOLDER = "<<<PARTY_B_PLACEHOLDER>>>"


def apply_remove(contract_text: str, answer_text: str, answer_start: int) -> str:
    """Delete the answer span from the contract text."""
    before = contract_text[:answer_start]
    after = contract_text[answer_start + len(answer_text):]
    # Avoid double spaces at the join point
    result = before + after
    result = re.sub(r"  +", " ", result)
    return result


def _flip_number(value_str: str, rng: random.Random) -> str:
    """Flip a numeric value by ±20%."""
    try:
        n = float(value_str)
    except ValueError:
        return value_str
    if n == 0:
        return value_str
    factor = 0.8 if rng.random() > 0.5 else 1.2
    flipped = n * factor
    if n == int(n):
        return str(int(round(flipped)))
    return f"{flipped:.1f}"


def apply_invert(answer_text: str, rng: random.Random) -> str:
    """
    Negate key legal terms in the answer span:
      - "shall not" ↔ "shall" (via placeholder to prevent double-negation)
      - numeric values with legal units flipped ±20%
      - Party A ↔ Party B (via placeholder)
    """
    text = answer_text

    # Shall swap (protect "shall not" first)
    text = re.sub(r"\bshall not\b", _SHALL_NOT_PLACEHOLDER, text, flags=re.IGNORECASE)
    text = re.sub(r"\bshall\b", "shall not", text, flags=re.IGNORECASE)
    text = text.replace(_SHALL_NOT_PLACEHOLDER, "shall")

    # Numeric flip — only values adjacent to legal unit words
    def replacer(m: re.Match) -> str:
        flipped = _flip_number(m.group(1), rng)
        return f"{flipped} {m.group(2)}"

    text = _UNIT_PATTERN.sub(replacer, text)

    # Party A ↔ Party B swap
    text = re.sub(r"\bParty A\b", _PARTY_B_PLACEHOLDER, text)
    text = re.sub(r"\bParty B\b", "Party A", text)
    text = text.replace(_PARTY_B_PLACEHOLDER, "Party B")

    return text


def apply_contradict(
    contract_text: str,
    answer_text: str,
    answer_start: int,
    question: str,
    question_to_spans: dict[str, list[tuple[str, str]]],
    source_contract_id: str,
    rng: random.Random,
) -> Optional[tuple[str, str]]:
    """
    Replace the answer span with a span from a different contract that answered
    the same question.  Returns (new_contract_text, replacement_span) or None
    if no candidate exists.
    """
    candidates = [
        (cid, span)
        for cid, span in question_to_spans.get(question, [])
        if cid != source_contract_id and span != answer_text
    ]
    if not candidates:
        return None

    _, replacement = rng.choice(candidates)
    before = contract_text[:answer_start]
    after = contract_text[answer_start + len(answer_text):]
    new_text = before + replacement + after
    return new_text, replacement


# ---------------------------------------------------------------------------
# Label construction helpers
# ---------------------------------------------------------------------------

REASONING = {
    Label.GROUNDED: "The contract explicitly states the requested clause.",
    Label.ABSENT: "This clause is not present in this contract.",
    "ABSENT_REMOVE": "The clause was removed from this version of the contract.",
    Label.CONTRADICTS_PRIOR: "The contract contains terms that deviate from the standard.",
}


def make_grounded(raw: RawCUADExample) -> LabeledQAExample:
    return LabeledQAExample(
        contract_id=raw.contract_id,
        contract_text=raw.contract_text,
        question=raw.question,
        label=Label.GROUNDED,
        answer=raw.answer_text,
        citation=raw.answer_text,
        reasoning=REASONING[Label.GROUNDED],
    )


def make_absent_natural(raw: RawCUADExample) -> LabeledQAExample:
    return LabeledQAExample(
        contract_id=raw.contract_id,
        contract_text=raw.contract_text,
        question=raw.question,
        label=Label.ABSENT,
        answer=None,
        citation=None,
        reasoning=REASONING[Label.ABSENT],
    )


def make_absent_remove(raw: RawCUADExample) -> Optional[LabeledQAExample]:
    if raw.answer_text is None or raw.answer_start is None:
        return None
    perturbed_text = apply_remove(raw.contract_text, raw.answer_text, raw.answer_start)
    return LabeledQAExample(
        contract_id=raw.contract_id + "__REMOVE",
        contract_text=perturbed_text,
        question=raw.question,
        label=Label.ABSENT,
        answer=None,
        citation=None,
        reasoning=REASONING["ABSENT_REMOVE"],
    )


def make_contradicts_invert(raw: RawCUADExample, rng: random.Random) -> Optional[LabeledQAExample]:
    if raw.answer_text is None or raw.answer_start is None:
        return None
    perturbed_span = apply_invert(raw.answer_text, rng)
    if perturbed_span == raw.answer_text:
        # Inversion had no effect — skip
        return None
    perturbed_text = (
        raw.contract_text[: raw.answer_start]
        + perturbed_span
        + raw.contract_text[raw.answer_start + len(raw.answer_text):]
    )
    return LabeledQAExample(
        contract_id=raw.contract_id + "__INVERT",
        contract_text=perturbed_text,
        question=raw.question,
        label=Label.CONTRADICTS_PRIOR,
        answer=perturbed_span,
        citation=perturbed_span,
        reasoning=REASONING[Label.CONTRADICTS_PRIOR],
    )


def make_contradicts_contradict(
    raw: RawCUADExample,
    question_to_spans: dict[str, list[tuple[str, str]]],
    rng: random.Random,
) -> Optional[LabeledQAExample]:
    if raw.answer_text is None or raw.answer_start is None:
        return None
    result = apply_contradict(
        raw.contract_text,
        raw.answer_text,
        raw.answer_start,
        raw.question,
        question_to_spans,
        raw.contract_id,
        rng,
    )
    if result is None:
        return None
    new_text, replacement = result
    return LabeledQAExample(
        contract_id=raw.contract_id + "__CONTRADICT",
        contract_text=new_text,
        question=raw.question,
        label=Label.CONTRADICTS_PRIOR,
        answer=replacement,
        citation=replacement,
        reasoning=REASONING[Label.CONTRADICTS_PRIOR],
    )


# ---------------------------------------------------------------------------
# Distribution targeting
# ---------------------------------------------------------------------------

def compute_targets(
    n_grounded: int,
    n_absent_natural: int,
    target_g: float = 0.40,
    target_a: float = 0.40,
    target_c: float = 0.20,
) -> tuple[int, int, int]:
    """
    Given existing GROUNDED and natural ABSENT counts, compute how many
    additional ABSENT (via REMOVE) and CONTRADICTS_PRIOR we need to hit the
    target distribution.

    Returns (n_keep_grounded, n_extra_absent, n_contradicts_needed).
    """
    # We keep all grounded. Solve for total T such that:
    #   n_grounded / T = target_g  →  T = n_grounded / target_g
    T = int(n_grounded / target_g)
    target_absent_total = int(T * target_a)
    target_contradicts = int(T * target_c)

    extra_absent = max(0, target_absent_total - n_absent_natural)
    return n_grounded, extra_absent, target_contradicts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_raw(path: Path) -> list[RawCUADExample]:
    examples: list[RawCUADExample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(RawCUADExample.model_validate_json(line))
    return examples


def build_question_to_spans(
    answered: list[RawCUADExample],
) -> dict[str, list[tuple[str, str]]]:
    """Map each question text → [(contract_id, answer_text), ...] for CONTRADICT strategy."""
    mapping: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ex in answered:
        if ex.answer_text:
            mapping[ex.question].append((ex.contract_id, ex.answer_text))
    return dict(mapping)


def generate(
    raw_path: Path,
    output_path: Path,
    use_contradicts_prior: bool,
    seed: int,
) -> None:
    rng = random.Random(seed)

    print(f"Loading raw examples from {raw_path}...")
    all_raw = load_raw(raw_path)
    print(f"  Loaded {len(all_raw)} raw examples")

    answered = [ex for ex in all_raw if ex.answer_text is not None]
    unanswered = [ex for ex in all_raw if ex.answer_text is None]
    print(f"  With answers   : {len(answered)}")
    print(f"  Without answers: {len(unanswered)}")

    question_to_spans = build_question_to_spans(answered)

    # Shuffle answered pool deterministically
    rng.shuffle(answered)

    # Determine how many of each type we need
    n_grounded, n_extra_absent, n_contradicts = compute_targets(
        n_grounded=len(answered),
        n_absent_natural=len(unanswered),
    )
    print(f"\nTarget generation plan:")
    print(f"  GROUNDED (original)         : {n_grounded}")
    print(f"  ABSENT (natural)            : {len(unanswered)}")
    print(f"  ABSENT (REMOVE perturbation): {n_extra_absent}")
    if use_contradicts_prior:
        print(f"  CONTRADICTS_PRIOR (target)  : {n_contradicts}")
    else:
        print(f"  CONTRADICTS_PRIOR           : skipped (--use_contradicts_prior not set)")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labeled: list[LabeledQAExample] = []

    # --- GROUNDED: all answered examples ---
    for ex in answered:
        labeled.append(make_grounded(ex))

    # --- ABSENT: natural unanswered ---
    for ex in unanswered:
        labeled.append(make_absent_natural(ex))

    # --- ABSENT: REMOVE perturbation ---
    remove_pool = rng.sample(answered, min(n_extra_absent * 2, len(answered)))
    remove_count = 0
    for ex in remove_pool:
        if remove_count >= n_extra_absent:
            break
        item = make_absent_remove(ex)
        if item is not None:
            labeled.append(item)
            remove_count += 1

    # --- CONTRADICTS_PRIOR: INVERT and CONTRADICT ---
    if use_contradicts_prior:
        contradicts_pool = rng.sample(answered, min(n_contradicts * 3, len(answered)))
        cp_count = 0
        for i, ex in enumerate(contradicts_pool):
            if cp_count >= n_contradicts:
                break
            # Alternate between INVERT and CONTRADICT
            if i % 2 == 0:
                item = make_contradicts_invert(ex, rng)
            else:
                item = make_contradicts_contradict(ex, question_to_spans, rng)
            if item is not None:
                labeled.append(item)
                cp_count += 1

    # Shuffle final list
    rng.shuffle(labeled)

    # Write output
    with output_path.open("w", encoding="utf-8") as fh:
        for ex in labeled:
            fh.write(ex.model_dump_json() + "\n")

    # Print class counts
    from collections import Counter
    counts = Counter(ex.label.value for ex in labeled)
    total = len(labeled)
    print(f"\n--- Output class distribution ({total} total) ---")
    for lbl in [Label.GROUNDED, Label.ABSENT, Label.CONTRADICTS_PRIOR]:
        n = counts.get(lbl.value, 0)
        pct = 100.0 * n / total if total > 0 else 0.0
        print(f"  {lbl.value:<22}: {n:>6}  ({pct:.1f}%)")
    print(f"\nOutput saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perturb CUAD examples and generate labeled QA dataset. "
            "Fully deterministic — no external model calls."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/cuad_raw.jsonl"),
        help="Path to raw CUAD JSONL (default: data/raw/cuad_raw.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/perturbed/labeled.jsonl"),
        help="Output JSONL path (default: data/perturbed/labeled.jsonl)",
    )
    parser.add_argument(
        "--use_contradicts_prior",
        action="store_true",
        default=False,
        help="Generate CONTRADICTS_PRIOR examples (default: off, 2-class mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(
        raw_path=args.input,
        output_path=args.output,
        use_contradicts_prior=args.use_contradicts_prior,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
