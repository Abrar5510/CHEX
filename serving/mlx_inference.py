"""
MLX-backed inference helpers for CHEX.

This module mirrors the remote-MLX request/parse approach used in `demo/app.py`,
but exposes clean Python functions that can be reused by HTTP (FastAPI) and
Gradio API endpoints.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from data.schema import BankStatementSummary, ModelOutput
from serving.bank_statement import extract_text_from_file

MLX_SERVER_URL = os.environ.get("MLX_SERVER_URL", "").rstrip("/")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a contract analysis assistant specializing in detecting hallucinations \
and calibrated uncertainty. Given a contract text and a question about a specific \
clause, output a single JSON object with exactly these fields:

  question  : the question asked (copy verbatim)
  label     : one of GROUNDED, ABSENT, or CONTRADICTS_PRIOR
               - GROUNDED         : the information exists verbatim in the contract
               - ABSENT           : the contract does not contain this clause at all
               - CONTRADICTS_PRIOR: the contract contains a clause but it deviates \
from standard legal terms (e.g., inverted obligations, non-standard timeframes)
  answer    : the answer text if GROUNDED or CONTRADICTS_PRIOR, null if ABSENT
  citation  : the exact verbatim span from the contract that supports the answer, \
null if ABSENT
  reasoning : one sentence explaining your classification

Output ONLY the JSON object. No preamble, no markdown fences, no text outside the JSON.
"""

BANK_SYSTEM_PROMPT = """\
You are a financial analysis assistant specialising in bank statement review. \
Given a bank statement (plain text, CSV/Excel-derived, OFX/QFX-derived, or PDF-extracted) and either a \
summary request or a specific question, produce a single JSON object.

For SUMMARY mode (question is "SUMMARISE"):
Output a JSON object with exactly these fields:
  total_credits      : total money received (e.g. "£3,420.50") or null
  total_debits       : total money spent (e.g. "£2,105.30") or null
  largest_transaction: description + amount of the single largest transaction or null
  recurring_payments : list of detected recurring charges (e.g. ["Netflix £9.99", "Gym £35.00"]) or []
  flags              : list of unusual or suspicious items (e.g. ["Large cash withdrawal £800"]) or []
  raw_reasoning      : one sentence summarising your analysis

For Q&A mode (any other question), output a JSON object with exactly these fields:
  question  : the question asked (copy verbatim)
  label     : one of GROUNDED, ABSENT, or CONTRADICTS_PRIOR
  answer    : the answer text if GROUNDED or CONTRADICTS_PRIOR, null if ABSENT
  citation  : the exact verbatim span from the statement, null if ABSENT
  reasoning : one sentence explaining your classification

Output ONLY the JSON object. No preamble, no markdown fences, no text outside the JSON.
"""

STRICT_SUFFIX = (
    "\n\nIMPORTANT: You must output ONLY a valid JSON object. "
    "Do not include any text before or after the JSON."
)

SUMMARISE_QUESTION = "SUMMARISE"

MAX_CHARS = 32000  # keep remote inference requests snappy
MAX_STATEMENTS = 6


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _extract_json_str(raw_text: str) -> str:
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw_text, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw_text[:300]!r}")
    return match.group()


def _parse_model_output(raw_text: str) -> ModelOutput:
    return ModelOutput.model_validate_json(_extract_json_str(raw_text))


def _parse_summary(raw_text: str) -> BankStatementSummary:
    data = json.loads(_extract_json_str(raw_text))
    return BankStatementSummary(
        total_credits=data.get("total_credits"),
        total_debits=data.get("total_debits"),
        largest_transaction=data.get("largest_transaction"),
        recurring_payments=data.get("recurring_payments") or [],
        flags=data.get("flags") or [],
        raw_reasoning=data.get("raw_reasoning", ""),
    )


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _build_contract_messages(contract_text: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[CONTRACT]\n{contract_text}\n[/CONTRACT]\n\nQuestion: {question}"},
    ]


def _build_bank_messages(statement_text: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": BANK_SYSTEM_PROMPT},
        {"role": "user", "content": f"[STATEMENT]\n{statement_text}\n[/STATEMENT]\n\nQuestion: {question}"},
    ]


def _apply_strict(messages: list[dict], strict: bool) -> list[dict]:
    if not strict:
        return messages
    out = list(messages)
    out[-1] = dict(out[-1])
    out[-1]["content"] += STRICT_SUFFIX
    return out


def _truncate(text: str) -> str:
    text = text or ""
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS]
    return text


# ---------------------------------------------------------------------------
# Remote MLX call
# ---------------------------------------------------------------------------

def _run_inference(messages: list[dict]) -> str:
    if not MLX_SERVER_URL:
        raise RuntimeError("MLX_SERVER_URL not set.")

    import urllib.request

    payload = json.dumps(
        {
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{MLX_SERVER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Contract API
# ---------------------------------------------------------------------------

def analyse_contract_json(contract_text: str, question: str) -> ModelOutput:
    if not (contract_text or "").strip():
        raise ValueError("contract_text is required")
    if not (question or "").strip():
        raise ValueError("question is required")

    contract_text = _truncate(contract_text)
    messages = _build_contract_messages(contract_text, question)

    for attempt in range(2):
        raw = _run_inference(_apply_strict(messages, strict=attempt == 1))
        try:
            return _parse_model_output(raw)
        except Exception:
            if attempt == 1:
                raise
    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# Bank statement helpers
# ---------------------------------------------------------------------------

def _split_statements(paste_text: str) -> list[str]:
    text = (paste_text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?m)^[ \t]*-{3,}[ \t]*$", text)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned if cleaned else [text]


def _combined_statement_text(statement_texts: list[str]) -> str:
    return "\n\n".join(
        f"===== Statement {i+1}/{len(statement_texts)} =====\n\n{st.strip()}"
        for i, st in enumerate(statement_texts)
        if st.strip()
    ).strip()


def _collect_union(items: list[list[str] | None]) -> list[str]:
    out: list[str] = []
    for arr in items:
        for x in (arr or []):
            if x not in out:
                out.append(x)
    return out


def extract_statement_texts_from_uploads(
    uploads: list[tuple[str, bytes]],
    *,
    pdf_password: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """
    Convert uploaded files (filename, bytes) into statement text blocks using
    `serving.bank_statement.extract_text_from_file`.
    """
    statement_texts: list[str] = []
    errors: list[str] = []
    for idx, (filename, content) in enumerate(uploads, start=1):
        suffix = Path(filename).suffix or ".bin"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="chex_stmt_")
        try:
            tmp.write(content)
            tmp.close()
            try:
                text = extract_text_from_file(tmp.name, password=pdf_password)
                if not (text or "").strip():
                    errors.append(f"File #{idx} produced no extractable text.")
                else:
                    statement_texts.append(text)
            except Exception as e:
                errors.append(f"File #{idx} extraction error: {e}")
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except Exception:
                pass
    return statement_texts, errors


def analyse_bank_json(
    *,
    statement_text: str | None = None,
    uploads: Optional[list[tuple[str, bytes]]] = None,
    pdf_password: str | None = None,
    question: str | None = None,
) -> dict:
    """
    Bank statement API:
    - Always produces per-statement summaries.
    - Optionally produces a Q&A classification if `question` is provided.
    """
    statement_texts: list[str] = []
    errors: list[str] = []

    statement_texts.extend(_split_statements(statement_text or ""))

    if uploads:
        uploaded_texts, uploaded_errors = extract_statement_texts_from_uploads(
            uploads, pdf_password=(pdf_password or "").strip() or None
        )
        statement_texts.extend([t for t in uploaded_texts if (t or "").strip()])
        errors.extend(uploaded_errors)

    statement_texts = [t for t in statement_texts if (t or "").strip()]
    if not statement_texts:
        raise ValueError("Provide either statement_text or at least one file upload.")

    if len(statement_texts) > MAX_STATEMENTS:
        errors.append(f"Too many statements provided; only the first {MAX_STATEMENTS} were used.")
        statement_texts = statement_texts[:MAX_STATEMENTS]

    summaries: list[BankStatementSummary] = []
    for st in statement_texts:
        st = _truncate(st)
        messages = _build_bank_messages(st, SUMMARISE_QUESTION)
        summary: Optional[BankStatementSummary] = None
        for attempt in range(2):
            raw = _run_inference(_apply_strict(messages, strict=attempt == 1))
            try:
                summary = _parse_summary(raw)
                break
            except Exception:
                if attempt == 1:
                    summary = BankStatementSummary(raw_reasoning="Model output could not be parsed.")
        summaries.append(summary or BankStatementSummary(raw_reasoning="Model output could not be parsed."))

    combined_text = _combined_statement_text(statement_texts)

    answer: Optional[ModelOutput] = None
    if (question or "").strip():
        q = (question or "").strip()
        qa_messages = _build_bank_messages(_truncate(combined_text), q)
        for attempt in range(2):
            raw = _run_inference(_apply_strict(qa_messages, strict=attempt == 1))
            try:
                answer = _parse_model_output(raw)
                break
            except Exception:
                if attempt == 1:
                    raise

    recurring_union = _collect_union([s.recurring_payments for s in summaries])
    flags_union = _collect_union([s.flags for s in summaries])

    return {
        "summaries": [s.model_dump() for s in summaries],
        "recurring_payments_union": recurring_union,
        "flags_union": flags_union,
        "answer": answer.model_dump() if answer else None,
        "errors": errors,
        "combined_text": combined_text,
    }

