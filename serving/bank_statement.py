"""
BankStatementAnalyzer — extracts text from bank statements (PDF, CSV, plain text)
and runs summarisation or Q&A using the CHEX fine-tuned model.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import BankStatementSummary, Label, ModelOutput
from training.prompt_template import (
    build_bank_chat_messages,
    format_bank_inference_prompt,
)

SUMMARISE_QUESTION = "SUMMARISE"

STRICT_SUFFIX = (
    "\n\nIMPORTANT: You must output ONLY a valid JSON object. "
    "Do not include any text before or after the JSON."
)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    if importlib.util.find_spec("pdfplumber") is None:
        raise ImportError(
            "pdfplumber is required for PDF support. Install it with: pip install pdfplumber"
        )
    import pdfplumber  # type: ignore

    text_parts: list[str] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def parse_csv(file_path: str | Path) -> str:
    """
    Read a CSV bank statement and format it as a readable text block.
    Returns a string with one transaction per line.
    """
    if importlib.util.find_spec("pandas") is None:
        raise ImportError(
            "pandas is required for CSV support. Install it with: pip install pandas"
        )
    import pandas as pd  # type: ignore

    df = pd.read_csv(str(file_path))
    # Normalise column names to lower-case for robust detection
    df.columns = [c.strip().lower() for c in df.columns]

    lines: list[str] = []
    for _, row in df.iterrows():
        parts = [str(v).strip() for v in row.values if str(v).strip() not in ("", "nan")]
        lines.append(", ".join(parts))

    header = ", ".join(df.columns.tolist())
    return f"{header}\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_json(raw_text: str) -> dict:
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw_text, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw_text[:300]!r}")
    return json.loads(match.group())


def _parse_summary(raw_text: str) -> BankStatementSummary:
    data = _extract_json(raw_text)
    return BankStatementSummary(
        total_credits=data.get("total_credits"),
        total_debits=data.get("total_debits"),
        largest_transaction=data.get("largest_transaction"),
        recurring_payments=data.get("recurring_payments") or [],
        flags=data.get("flags") or [],
        raw_reasoning=data.get("raw_reasoning", ""),
    )


def _parse_qa(raw_text: str, question: str) -> ModelOutput:
    data = _extract_json(raw_text)
    label_str = data.get("label", "ABSENT").upper()
    try:
        label = Label(label_str)
    except ValueError:
        label = Label.ABSENT
    return ModelOutput(
        question=data.get("question", question),
        label=label,
        answer=data.get("answer"),
        citation=data.get("citation"),
        reasoning=data.get("reasoning", ""),
    )


# ---------------------------------------------------------------------------
# BankStatementAnalyzer
# ---------------------------------------------------------------------------

class BankStatementAnalyzer:
    """
    Wraps the CHEX model pipeline for bank statement analysis.

    Args:
        contract_analyzer: A loaded ContractAnalyzer instance (reuses its pipeline).
            If None, the analyzer operates in extraction-only mode (no LLM calls).
    """

    MAX_STATEMENT_TOKENS = 8192

    def __init__(self, contract_analyzer=None) -> None:
        self._ca = contract_analyzer

    # ------------------------------------------------------------------ #
    # Public text-extraction API                                           #
    # ------------------------------------------------------------------ #

    def extract_text_from_pdf(self, file_path: str | Path) -> str:
        return extract_text_from_pdf(file_path)

    def parse_csv(self, file_path: str | Path) -> str:
        return parse_csv(file_path)

    # ------------------------------------------------------------------ #
    # Internal inference helpers                                           #
    # ------------------------------------------------------------------ #

    def _truncate(self, text: str) -> str:
        if self._ca is None:
            return text
        tokens = self._ca._tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.MAX_STATEMENT_TOKENS:
            tokens = tokens[: self.MAX_STATEMENT_TOKENS]
            return self._ca._tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def _build_prompt(self, statement_text: str, question: str, strict: bool = False) -> str:
        messages = build_bank_chat_messages(statement_text, question)
        if strict:
            messages[-1]["content"] += STRICT_SUFFIX
        if self._ca is not None:
            try:
                return self._ca._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return format_bank_inference_prompt(statement_text, question) + (
            STRICT_SUFFIX if strict else ""
        )

    def _run_pipeline(self, prompt: str) -> str:
        if self._ca is None:
            raise RuntimeError("No model loaded — cannot run inference.")
        result = self._ca._pipe(prompt)
        return result[0]["generated_text"]

    # ------------------------------------------------------------------ #
    # Public analysis API                                                  #
    # ------------------------------------------------------------------ #

    def summarize(self, statement_text: str) -> BankStatementSummary:
        """
        Auto-generate a structured financial summary of the bank statement.
        Retries once with a stricter prompt on parse failure.
        """
        statement_text = self._truncate(statement_text)

        for attempt in range(2):
            prompt = self._build_prompt(statement_text, SUMMARISE_QUESTION, strict=attempt == 1)
            try:
                raw = self._run_pipeline(prompt)
                return _parse_summary(raw)
            except Exception as e:
                if attempt == 0:
                    print(f"  Summary parse attempt 1 failed ({e}). Retrying...")
                else:
                    print(f"  Summary parse attempt 2 failed ({e}). Returning empty summary.")

        return BankStatementSummary(raw_reasoning="Model output could not be parsed.")

    def answer_question(self, statement_text: str, question: str) -> ModelOutput:
        """
        Answer a specific question about the bank statement.
        Retries once with a stricter prompt on parse failure.
        """
        statement_text = self._truncate(statement_text)

        for attempt in range(2):
            prompt = self._build_prompt(statement_text, question, strict=attempt == 1)
            try:
                raw = self._run_pipeline(prompt)
                return _parse_qa(raw, question)
            except Exception as e:
                if attempt == 0:
                    print(f"  Q&A parse attempt 1 failed ({e}). Retrying...")
                else:
                    print(f"  Q&A parse attempt 2 failed ({e}). Returning safe fallback.")

        return ModelOutput(
            question=question,
            label=Label.ABSENT,
            answer=None,
            citation=None,
            reasoning="Model output could not be parsed after two attempts.",
        )
