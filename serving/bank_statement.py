"""
BankStatementAnalyzer — converts bank statements in multiple formats to plain text
and runs summarisation or Q&A using the CHEX fine-tuned model.

Supported input formats:
- PDF (.pdf)
- CSV (.csv)
- Plain text (.txt / .text)
- Excel (.xlsx)
- OFX/QFX (.ofx / .qfx) (lightweight tag-based extraction)
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

def extract_text_from_pdf(
    file_path: str | Path, password: Optional[str] = None
) -> str:
    """
    Extract all text from a (potentially encrypted) PDF using pdfplumber.

    If the PDF is password-protected, provide `password`.
    """
    if importlib.util.find_spec("pdfplumber") is None:
        raise ImportError(
            "pdfplumber is required for PDF support. Install it with: pip install pdfplumber"
        )
    import pdfplumber  # type: ignore

    text_parts: list[str] = []
    # pdfplumber's `open` supports forwarding `password` to pdfminer in most versions.
    # We keep a fallback for older versions where the argument may not exist.
    try:
        with pdfplumber.open(str(file_path), password=password or "") as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except TypeError:
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


def parse_txt(file_path: str | Path) -> str:
    """
    Read a plain text bank statement into a model-friendly string.

    Note: for multi-statement paste-style inputs, clients should split on their
    own delimiter; this function just returns the file contents.
    """
    p = Path(file_path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Best-effort fallback for odd encodings.
        return p.read_text(encoding="utf-8", errors="replace")


def parse_xlsx(file_path: str | Path) -> str:
    """
    Read the first sheet from an Excel file and format it as one transaction
    per line (similar to `parse_csv`).
    """
    if importlib.util.find_spec("pandas") is None:
        raise ImportError("pandas is required for XLSX support. Install it with: pip install pandas")
    import pandas as pd  # type: ignore

    # Let pandas pick the engine; openpyxl is a requirement for .xlsx.
    df = pd.read_excel(str(file_path), sheet_name=0)
    if df is None or df.empty:
        return ""

    # Normalise column names to lower-case for robust detection
    df.columns = [str(c).strip().lower() for c in df.columns]

    lines: list[str] = []
    for _, row in df.iterrows():
        parts = [str(v).strip() for v in row.values if str(v).strip() not in ("", "nan", "NaN")]
        lines.append(", ".join(parts))

    header = ", ".join(df.columns.tolist())
    return f"{header}\n" + "\n".join(lines)


def _format_ofx_date(d: str) -> str:
    d = (d or "").strip()
    if len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return d


def parse_ofx(file_path: str | Path) -> str:
    """
    Lightweight OFX/QFX extraction.

    OFX is XML-like tag syntax but often not strict XML; we use regex to capture
    transaction blocks and convert them to a consistent text format.
    """
    p = Path(file_path)
    raw = p.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    # OFX commonly wraps each transaction as: <STMTTRN> ... </STMTTRN>
    blocks = re.findall(r"<STMTTRN>(.*?)</STMTTRN>", text, flags=re.IGNORECASE | re.DOTALL)
    if not blocks:
        # Some variants nest transactions differently; fall back to returning the whole file.
        # We still try to keep it useful by stripping extra whitespace.
        return text.strip()

    def get_tag(block: str, tag: str) -> str:
        m = re.search(rf"<{tag}>([^<]*)", block, flags=re.IGNORECASE)
        return (m.group(1) if m else "").strip()

    lines: list[str] = []
    for b in blocks:
        dt = get_tag(b, "DTPOSTED") or get_tag(b, "DTTRAN")
        name = get_tag(b, "NAME") or get_tag(b, "PAYEE")
        memo = get_tag(b, "MEMO") or get_tag(b, "TRNTYPE")
        amt = get_tag(b, "TRNAMT") or get_tag(b, "AMOUNT")

        if not any([dt, name, memo, amt]):
            continue

        dt = _format_ofx_date(dt)
        desc_parts = [p for p in [name, memo] if p]
        desc = " - ".join(desc_parts) if desc_parts else "Transaction"
        lines.append(f"{dt}, {desc}, {amt}".strip(", "))

    header = "Date, Description, Amount"
    return header + "\n" + "\n".join(lines) if lines else header + "\n" + text.strip()[:5000]


def detect_statement_format(file_path: str | Path) -> str:
    """
    Return a best-effort statement format string for dispatcher logic.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".csv":
        return "csv"
    if suffix in (".txt", ".text"):
        return "txt"
    if suffix in (".xlsx",):
        return "xlsx"
    if suffix in (".ofx", ".qfx"):
        return "ofx"
    return "unknown"


def extract_text_from_file(
    file_path: str | Path, *, password: Optional[str] = None
) -> str:
    """
    Convert a supported statement file into plain text for the model.
    """
    fmt = detect_statement_format(file_path)

    if fmt == "pdf":
        return extract_text_from_pdf(file_path, password=password)
    if fmt == "csv":
        return parse_csv(file_path)
    if fmt == "txt":
        return parse_txt(file_path)
    if fmt == "xlsx":
        return parse_xlsx(file_path)
    if fmt == "ofx":
        return parse_ofx(file_path)

    # Content fallback for OFX/QFX-ish files without/unknown extensions.
    raw_head = Path(file_path).read_bytes()[:4096]
    head = raw_head.decode("utf-8", errors="replace").upper()
    if "<OFX" in head or "<QFX" in head:
        return parse_ofx(file_path)

    raise ValueError(f"Unsupported statement format for file: {file_path}")


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

    def extract_text_from_pdf(
        self, file_path: str | Path, password: Optional[str] = None
    ) -> str:
        return extract_text_from_pdf(file_path, password=password)

    def parse_csv(self, file_path: str | Path) -> str:
        return parse_csv(file_path)

    def extract_text_from_file(
        self, file_path: str | Path, *, password: Optional[str] = None
    ) -> str:
        """
        Unified extraction entrypoint for supported statement file formats.
        """
        return extract_text_from_file(file_path, password=password)

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
