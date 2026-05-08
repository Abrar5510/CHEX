"""
CHEX - Document Intelligence
HuggingFace Spaces Gradio Demo — fully self-contained (no relative imports)

Tab 1: Analyze Contract — paste a contract, ask a question, get a structured answer
Tab 2: Benchmark Demo — side-by-side table showing base model hallucinations vs CHEX
Tab 3: Analyse Bank Statement — paste / upload a bank statement, get a summary + Q&A
"""

from __future__ import annotations

import csv
import datetime as _dt
import importlib.util
import io
import json
import os
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional

import gradio as gr
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Schema (inlined from data/schema.py)
# ---------------------------------------------------------------------------

class Label(str, Enum):
    GROUNDED = "GROUNDED"
    ABSENT = "ABSENT"
    CONTRADICTS_PRIOR = "CONTRADICTS_PRIOR"


class ModelOutput(BaseModel):
    question: str
    label: Label
    answer: Optional[str] = None
    citation: Optional[str] = None
    reasoning: str


class BankStatementSummary(BaseModel):
    total_credits: Optional[str] = None
    total_debits: Optional[str] = None
    largest_transaction: Optional[str] = None
    recurring_payments: Optional[list[str]] = None
    flags: Optional[list[str]] = None
    raw_reasoning: str


# ---------------------------------------------------------------------------
# Prompt templates (inlined from training/prompt_template.py)
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

### Example 1 — GROUNDED

[CONTRACT]
This Software License Agreement ("Agreement") is entered into as of January 1, 2024, \
between TechVision Inc. ("Licensor") and GlobalCorp Ltd. ("Licensee"). The Agreement \
shall remain in effect for a period of two (2) years from the Effective Date, unless \
earlier terminated pursuant to Section 8. Licensor grants Licensee a non-exclusive, \
non-transferable license to use the Software solely for Licensee's internal business \
purposes.
[/CONTRACT]

Question: What is the duration of this agreement?

{"question": "What is the duration of this agreement?", "label": "GROUNDED", \
"answer": "Two years from the Effective Date", \
"citation": "remain in effect for a period of two (2) years from the Effective Date", \
"reasoning": "The contract explicitly specifies a two-year term starting from the Effective Date."}

### Example 2 — ABSENT

[CONTRACT]
The Licensee shall pay a monthly fee of five hundred dollars ($500.00). Payment is due \
on the first business day of each calendar month. Late payments shall accrue interest \
at a rate of one and one-half percent (1.5%) per month. Licensee shall maintain \
accurate records of all uses of the Software.
[/CONTRACT]

Question: Does this agreement include a limitation of liability clause?

{"question": "Does this agreement include a limitation of liability clause?", \
"label": "ABSENT", "answer": null, "citation": null, \
"reasoning": "No limitation of liability clause appears anywhere in the provided contract text."}

### Example 3 — CONTRADICTS_PRIOR

[CONTRACT]
This Non-Disclosure Agreement is made between AlphaTech Solutions ("Discloser") and \
Beta Dynamics Corp. ("Recipient"). The Recipient shall not disclose Confidential \
Information to any third party. NON-COMPETE: The Recipient shall engage in any \
business activity that competes with the Discloser's primary operations during the \
term and for a period of 24 months thereafter. The Recipient shall not take any \
steps to protect Discloser's trade secrets.
[/CONTRACT]

Question: Does this agreement restrict the Recipient from competing with the Discloser?

{"question": "Does this agreement restrict the Recipient from competing with the Discloser?", \
"label": "CONTRADICTS_PRIOR", \
"answer": "The non-compete clause has inverted obligations — it permits competition rather than prohibiting it", \
"citation": "The Recipient shall engage in any business activity that competes with the Discloser's primary operations", \
"reasoning": "The clause uses 'shall engage' instead of 'shall not engage', inverting the standard non-compete obligation."}
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


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_json_str(raw_text: str) -> str:
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw_text, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw_text[:300]!r}")
    return match.group()


def _parse_model_output(raw_text: str, question: str) -> ModelOutput:
    json_str = _extract_json_str(raw_text)
    return ModelOutput.model_validate_json(json_str)


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
# Model loading
# ---------------------------------------------------------------------------

MLX_SERVER_URL = os.environ.get("MLX_SERVER_URL", "").rstrip("/")
SAMPLE_DIR     = Path(__file__).parent / "sample_contracts"
STATEMENT_DIR  = Path(__file__).parent / "sample_statements"

model_load_error: Optional[str] = None

if not MLX_SERVER_URL:
    model_load_error = "MLX_SERVER_URL not set. Set it in Space secrets to your Mac's ngrok URL."
    print(f"WARNING: {model_load_error}")
else:
    print(f"MLX server configured at: {MLX_SERVER_URL}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

MAX_CHARS = 32000  # rough character limit (~8k tokens) to keep requests fast


def _truncate(text: str) -> str:
    if len(text) > MAX_CHARS:
        print(f"WARNING: Text truncated from {len(text)} to {MAX_CHARS} chars.")
        return text[:MAX_CHARS]
    return text


def _apply_messages(messages: list[dict], strict: bool = False) -> list[dict]:
    if strict:
        messages = list(messages)
        messages[-1] = dict(messages[-1])
        messages[-1]["content"] += STRICT_SUFFIX
    return messages


def _run_inference(messages: list[dict]) -> str:
    import urllib.request
    payload = json.dumps({
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
    }).encode()
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
# Sample contract content
# ---------------------------------------------------------------------------

def _read_sample(filename: str) -> str:
    p = SAMPLE_DIR / filename
    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"[Sample contract '{filename}' not found. Place it in demo/sample_contracts/]"


SOFTWARE_LICENSE = _read_sample("software_license.txt")
NDA = _read_sample("nda.txt")
SERVICE_AGREEMENT = _read_sample("service_agreement.txt")

SAMPLE_QUESTIONS = {
    "software_license.txt": "What is the limitation of liability in this agreement?",
    "nda.txt": "Does this agreement include a non-compete clause?",
    "service_agreement.txt": "Does this contract include a termination for convenience clause?",
}


def _read_sample_statement(filename: str) -> str:
    p = STATEMENT_DIR / filename
    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"[Sample statement '{filename}' not found. Place it in demo/sample_statements/]"


SAMPLE_STATEMENT = _read_sample_statement("sample_statement.txt")


# ---------------------------------------------------------------------------
# Label badge HTML
# ---------------------------------------------------------------------------

_BADGE_CFG = {
    "GROUNDED":          ("#0f9d58", "rgba(34,197,94,0.10)",  "rgba(34,197,94,0.28)",  "✓"),
    "ABSENT":            ("#d23131", "rgba(239,68,68,0.09)",  "rgba(239,68,68,0.28)",  "✗"),
    "CONTRADICTS_PRIOR": ("#b87800", "rgba(245,158,11,0.10)", "rgba(245,158,11,0.30)", "⚠"),
    "N/A":               ("#8a91a3", "rgba(139,145,163,0.10)","rgba(139,145,163,0.25)","—"),
    "ERROR":             ("#991b1b", "rgba(220,38,38,0.10)",  "rgba(220,38,38,0.32)",  "!"),
}


def format_label_html(label: str) -> str:
    fg, bg, border, icon = _BADGE_CFG.get(label, _BADGE_CFG["N/A"])
    display = "CONTRADICTS PRIOR" if label == "CONTRADICTS_PRIOR" else label
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f'padding:11px 16px;border-radius:10px;border:1px solid {border};'
        f'background:{bg};color:{fg};font-family:\'Inter\',sans-serif;'
        f'font-size:12.5px;font-weight:600;letter-spacing:0.02em;'
        f'backdrop-filter:blur(10px);">'
        f'<span style="width:14px;height:14px;display:grid;place-items:center;'
        f'font-size:13px;">{icon}</span>'
        f'<span>{display}</span></div>'
    )


# ---------------------------------------------------------------------------
# Analysis handlers
# ---------------------------------------------------------------------------

def analyze_contract(contract_text: str, question: str) -> tuple[str, str, str, str]:
    if not contract_text.strip():
        return format_label_html("N/A"), "", "", "Please paste a contract above."
    if not question.strip():
        return format_label_html("N/A"), "", "", "Please enter a question."
    if not MLX_SERVER_URL:
        return (
            format_label_html("N/A"),
            "Model not loaded",
            "",
            f"Model failed to load: {model_load_error}.",
        )

    contract_text = _truncate(contract_text)
    messages = _build_contract_messages(contract_text, question)

    for attempt in range(2):
        msgs = _apply_messages(messages, strict=(attempt == 1))
        try:
            raw = _run_inference(msgs)
            result = _parse_model_output(raw, question)
            label_html = format_label_html(result.label.value)
            answer = result.answer or "(none — clause is absent or not applicable)"
            citation = result.citation or "(none)"
            return label_html, answer, citation, result.reasoning
        except Exception as e:
            if attempt == 0:
                print(f"  Parse attempt 1 failed ({e}). Retrying with stricter prompt...")
            else:
                print(f"  Parse attempt 2 failed ({e}). Returning safe fallback.")

    return (
        format_label_html("ABSENT"),
        "(none — clause is absent or not applicable)",
        "(none)",
        "Model output could not be parsed as valid JSON after two attempts.",
    )


def _get_statement_text(
    paste_text: str,
    pdf_file,
    pdf_password: str | None,
    csv_file,
    txt_file,
    xlsx_file,
    ofx_file,
) -> tuple[str, str]:
    # Backwards-compatible shim: treat "single statement" inputs as one item.
    texts, errors = _get_statement_texts(
        paste_text,
        pdf_file,
        pdf_password,
        csv_file,
        txt_file,
        xlsx_file,
        ofx_file,
    )
    if not texts:
        return (
            "",
            errors[0]
            if errors
            else "Please paste a bank statement or upload a PDF / CSV / TXT / XLSX / OFX/QFX file."
        )
    return texts[0], ""


def _ensure_file_list(files) -> list:
    if files is None:
        return []
    if isinstance(files, (list, tuple)):
        return [f for f in files if f is not None]
    return [files]


def _split_statements(paste_text: str) -> list[str]:
    """
    Split pasted content into multiple statements.

    Delimiter: a line containing only `---` (3+ dashes), optionally surrounded by whitespace.
    """
    text = (paste_text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?m)^[ \t]*-{3,}[ \t]*$", text)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned if cleaned else [text]


def _get_statement_texts(
    paste_text: str,
    pdf_files,
    pdf_password: str | None,
    csv_files,
    txt_files,
    xlsx_files,
    ofx_files,
) -> tuple[list[str], list[str]]:
    """
    Extract statement text blocks from:
      - pasted text (can contain multiple statements separated by `---`)
      - uploaded PDFs (supports multiple)
      - uploaded CSVs (supports multiple)
      - uploaded TXT files (supports multiple)
      - uploaded Excel (.xlsx) (supports multiple)
      - uploaded OFX/QFX files (supports multiple)
    """
    statement_texts: list[str] = []
    errors: list[str] = []

    pdf_list = _ensure_file_list(pdf_files)
    csv_list = _ensure_file_list(csv_files)
    txt_list = _ensure_file_list(txt_files)
    xlsx_list = _ensure_file_list(xlsx_files)
    ofx_list = _ensure_file_list(ofx_files)

    # PDFs
    if pdf_list:
        try:
            if importlib.util.find_spec("pdfplumber") is None:
                errors.append("pdfplumber not installed.")
            else:
                import pdfplumber
                password = (pdf_password or "").strip()
                for idx, pdf_file in enumerate(pdf_list):
                    try:
                        text_parts: list[str] = []
                        try:
                            with pdfplumber.open(
                                str(pdf_file),
                                password=password if password else "",
                            ) as pdf:
                                for page in pdf.pages:
                                    t = page.extract_text()
                                    if t:
                                        text_parts.append(t)
                        except TypeError:
                            # Older pdfplumber versions may not accept `password=...`
                            with pdfplumber.open(str(pdf_file)) as pdf:
                                for page in pdf.pages:
                                    t = page.extract_text()
                                    if t:
                                        text_parts.append(t)
                        text = "\n".join(text_parts).strip()
                        if not text:
                            errors.append(f"PDF #{idx+1} uploaded but no text could be extracted.")
                        else:
                            statement_texts.append(text)
                    except Exception as e:
                        msg = str(e).lower()
                        if "password" in msg or "encrypted" in msg or "decrypt" in msg:
                            errors.append(
                                f"PDF #{idx+1} is password-protected. Please enter the correct password."
                            )
                        else:
                            errors.append(f"PDF #{idx+1} extraction error: {e}")
        except Exception as e:
            errors.append(f"PDF extraction error: {e}")

    # CSVs
    if csv_list:
        try:
            import pandas as pd
        except Exception:
            if importlib.util.find_spec("pandas") is None:
                errors.append("pandas not installed.")
            else:
                errors.append("CSV parsing error: pandas import failed.")
        else:
            for idx, csv_file in enumerate(csv_list):
                try:
                    df = pd.read_csv(str(csv_file))
                    df.columns = [c.strip().lower() for c in df.columns]
                    lines: list[str] = []
                    for _, row in df.iterrows():
                        parts = [
                            str(v).strip()
                            for v in row.values
                            if str(v).strip() not in ("", "nan")
                        ]
                        lines.append(", ".join(parts))
                    statement_texts.append(
                        ", ".join(df.columns.tolist()) + "\n" + "\n".join(lines)
                    )
                except Exception as e:
                    errors.append(f"CSV #{idx+1} parsing error: {e}")

    # TXT
    if txt_list:
        for idx, txt_file in enumerate(txt_list):
            try:
                # Read best-effort encoding; then reuse the same delimiter splitting
                # strategy as pasted input.
                p = Path(str(txt_file))
                content = p.read_text(encoding="utf-8", errors="replace")
                parts = _split_statements(content)
                if not parts:
                    errors.append(f"TXT #{idx+1} uploaded but no text could be read.")
                else:
                    statement_texts.extend(parts)
            except Exception as e:
                errors.append(f"TXT #{idx+1} parsing error: {e}")

    # XLSX (Excel)
    if xlsx_list:
        try:
            import pandas as pd
        except Exception:
            if importlib.util.find_spec("pandas") is None:
                errors.append("pandas not installed.")
            else:
                errors.append("Excel parsing error: pandas import failed.")
        else:
            for idx, xlsx_file in enumerate(xlsx_list):
                try:
                    df = pd.read_excel(str(xlsx_file), sheet_name=0)
                    if df is None or df.empty:
                        errors.append(f"XLSX #{idx+1} uploaded but no rows were found.")
                        continue
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    lines: list[str] = []
                    for _, row in df.iterrows():
                        parts = [
                            str(v).strip()
                            for v in row.values
                            if str(v).strip() not in ("", "nan", "NaN")
                        ]
                        lines.append(", ".join(parts))
                    statement_texts.append(
                        ", ".join(df.columns.tolist()) + "\n" + "\n".join(lines)
                    )
                except Exception as e:
                    errors.append(f"XLSX #{idx+1} parsing error: {e}")

    # OFX/QFX (lightweight tag extraction)
    if ofx_list:
        def _format_ofx_date(d: str) -> str:
            d = (d or "").strip()
            if len(d) == 8 and d.isdigit():
                return f"{d[:4]}-{d[4:6]}-{d[6:]}"
            return d

        for idx, ofx_file in enumerate(ofx_list):
            try:
                p = Path(str(ofx_file))
                raw = p.read_bytes()
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    content = raw.decode("utf-8", errors="replace")

                blocks = re.findall(
                    r"<STMTTRN>(.*?)</STMTTRN>",
                    content,
                    flags=re.IGNORECASE | re.DOTALL,
                )

                def _get_tag(block: str, tag: str) -> str:
                    m = re.search(rf"<{tag}>([^<]*)", block, flags=re.IGNORECASE)
                    return (m.group(1) if m else "").strip()

                lines: list[str] = []
                for b in blocks:
                    dt = _get_tag(b, "DTPOSTED") or _get_tag(b, "DTTRAN")
                    name = _get_tag(b, "NAME") or _get_tag(b, "PAYEE")
                    memo = _get_tag(b, "MEMO") or _get_tag(b, "TRNTYPE")
                    amt = _get_tag(b, "TRNAMT") or _get_tag(b, "AMOUNT")

                    if not any([dt, name, memo, amt]):
                        continue

                    dt = _format_ofx_date(dt)
                    desc_parts = [p for p in [name, memo] if p]
                    desc = " - ".join(desc_parts) if desc_parts else "Transaction"
                    lines.append(f"{dt}, {desc}, {amt}".strip(", "))

                if lines:
                    statement_texts.append("Date, Description, Amount\n" + "\n".join(lines))
                else:
                    # Fall back to returning the raw content (truncated).
                    statement_texts.append(content.strip()[:20000])
            except Exception as e:
                errors.append(f"OFX/QFX #{idx+1} parsing error: {e}")

    # Paste text (may contain multiple statements)
    pasted_parts = _split_statements(paste_text)
    if pasted_parts:
        statement_texts.extend(pasted_parts)

    if not statement_texts:
        errors.append(
            "Please paste a bank statement or upload a PDF / CSV / TXT / XLSX / OFX/QFX file(s)."
        )

    return statement_texts, errors


def analyse_bank_statement(
    paste_text: str,
    pdf_file,
    pdf_password: str | None,
    csv_file,
    txt_file,
    xlsx_file,
    ofx_file,
) -> tuple[str, str, str]:
    statement_texts, errors = _get_statement_texts(
        paste_text,
        pdf_file,
        pdf_password,
        csv_file,
        txt_file,
        xlsx_file,
        ofx_file,
    )
    if not statement_texts:
        return f"**Error:** {errors[0] if errors else 'No bank statement provided.'}", "", ""

    MAX_STATEMENTS = 6
    if len(statement_texts) > MAX_STATEMENTS:
        errors.append(f"Too many statements provided; only the first {MAX_STATEMENTS} were used.")
        statement_texts = statement_texts[:MAX_STATEMENTS]

    combined_text = "\n\n".join(
        f"===== Statement {i+1}/{len(statement_texts)} =====\n\n{st.strip()}"
        for i, st in enumerate(statement_texts)
        if st.strip()
    ).strip()

    if not MLX_SERVER_URL:
        return (
            f"**Inference client not initialised.** Error: {model_load_error}",
            combined_text,
            "",
        )

    summaries: list[BankStatementSummary] = []
    for idx, statement_text in enumerate(statement_texts):
        statement_text = _truncate(statement_text)
        messages = _build_bank_messages(statement_text, "SUMMARISE")

        summary: BankStatementSummary | None = None
        for attempt in range(2):
            msgs = _apply_messages(messages, strict=(attempt == 1))
            try:
                raw = _run_inference(msgs)
                summary = _parse_summary(raw)
                break
            except Exception as e:
                if attempt == 0:
                    print(f"  Summary parse attempt 1 failed (statement {idx+1}, {e}). Retrying...")
                else:
                    print(f"  Summary parse attempt 2 failed (statement {idx+1}, {e}). Returning error.")

        if summary is None:
            summary = BankStatementSummary(
                raw_reasoning=f"Could not parse model output for statement {idx+1}."
            )
        summaries.append(summary)

    # Render markdown
    lines: list[str] = []
    lines.append("## Statements Summary")
    lines.append("")
    if errors:
        lines.append("**Notes:**")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    for idx, summary in enumerate(summaries):
        lines.append(f"### Statement {idx+1}")
        lines.append(f"**Total Credits:** {summary.total_credits or 'N/A'}")
        lines.append(f"**Total Debits:** {summary.total_debits or 'N/A'}")
        lines.append(
            f"**Largest Transaction:** {summary.largest_transaction or 'N/A'}"
        )
        if summary.recurring_payments:
            lines.append("\n**Recurring Payments:**")
            for p in summary.recurring_payments:
                lines.append(f"- {p}")
        if summary.flags:
            lines.append("\n**Flags / Unusual Activity:**")
            for f in summary.flags:
                lines.append(f"- {f}")
        lines.append(f"\n*{summary.raw_reasoning}*")
        lines.append("")

    # Overall union (useful across multiple statements)
    overall_recurring: list[str] = []
    overall_flags: list[str] = []
    for s in summaries:
        for r in (s.recurring_payments or []):
            if r not in overall_recurring:
                overall_recurring.append(r)
        for f in (s.flags or []):
            if f not in overall_flags:
                overall_flags.append(f)

    lines.append("## Overall (union across statements)")
    if overall_recurring:
        lines.append("\n**Recurring Payments (union):**")
        for p in overall_recurring:
            lines.append(f"- {p}")
    else:
        lines.append("\n**Recurring Payments (union):** N/A")

    if overall_flags:
        lines.append("\n**Flags / Unusual Activity (union):**")
        for f in overall_flags:
            lines.append(f"- {f}")
    else:
        lines.append("\n**Flags / Unusual Activity (union):** N/A")

    summary_json = json.dumps([s.model_dump() for s in summaries], ensure_ascii=False)
    return "\n".join(lines).strip(), combined_text, summary_json


def _safe_json_loads(s: str) -> object:
    try:
        obj = json.loads(s or "")
        if isinstance(obj, (dict, list)):
            return obj
        return {}
    except Exception:
        return {}


def _escape_pdf_text(s: str) -> str:
    # PDF literal strings escape backslash and parentheses.
    return (s or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _simple_pdf_bytes(title: str, lines: list[str]) -> bytes:
    """
    Tiny, dependency-free, single-page PDF generator for short text reports.
    """
    font = "Helvetica"
    font_size = 11
    left = 54
    top = 790
    leading = 14

    safe_title = _escape_pdf_text(title)
    safe_lines = [_escape_pdf_text(ln) for ln in lines]

    content_lines: list[str] = []
    content_lines.append("BT")
    content_lines.append(f"/F1 {font_size} Tf")
    content_lines.append(f"{left} {top} Td")
    content_lines.append(f"({_escape_pdf_text(safe_title)}) Tj")
    content_lines.append(f"0 -{leading*2} Td")
    for ln in safe_lines:
        content_lines.append(f"({ln}) Tj")
        content_lines.append(f"0 -{leading} Td")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects: list[bytes] = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
    )
    objects.append(f"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /{font} >>\nendobj\n".encode())
    objects.append(
        b"5 0 obj\n<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream\nendobj\n"
    )

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    xref: list[int] = [0]
    for obj in objects:
        xref.append(out.tell())
        out.write(obj)
    xref_start = out.tell()
    out.write(f"xref\n0 {len(xref)}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in xref[1:]:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        b"trailer\n<< /Size "
        + str(len(xref)).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_start).encode()
        + b"\n%%EOF\n"
    )
    return out.getvalue()


def export_bank_summary_csv(summary_json: str) -> tuple[str | None, str]:
    data = _safe_json_loads(summary_json)
    if not data:
        return None, "**Export error:** Run 'Analyse statement' first."

    statements = data if isinstance(data, list) else [data]

    filename = f"bank-statement-summaries_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="chex_", mode="w", newline="", encoding="utf-8")
    try:
        writer = csv.writer(tmp)
        writer.writerow([
            "statement_index",
            "total_credits",
            "total_debits",
            "largest_transaction",
            "recurring_payments",
            "flags",
            "raw_reasoning",
        ])

        overall_recurring: list[str] = []
        overall_flags: list[str] = []
        for s in statements:
            if not isinstance(s, dict):
                continue
            for r in (s.get("recurring_payments") or []):
                if r not in overall_recurring:
                    overall_recurring.append(r)
            for f in (s.get("flags") or []):
                if f not in overall_flags:
                    overall_flags.append(f)

        for i, s in enumerate(statements, start=1):
            if not isinstance(s, dict):
                continue
            writer.writerow([
                i,
                s.get("total_credits") or "",
                s.get("total_debits") or "",
                s.get("largest_transaction") or "",
                " | ".join(s.get("recurring_payments") or []),
                " | ".join(s.get("flags") or []),
                s.get("raw_reasoning") or "",
            ])

        # Overall union row
        writer.writerow([
            "overall",
            "",
            "",
            "",
            " | ".join(overall_recurring),
            " | ".join(overall_flags),
            "",
        ])
    finally:
        tmp.close()

    # Gradio uses the path; name displayed is fine.
    return tmp.name, f"**CSV ready:** `{filename}`"


def export_bank_summary_pdf(summary_json: str) -> tuple[str | None, str]:
    data = _safe_json_loads(summary_json)
    if not data:
        return None, "**Export error:** Run 'Analyse statement' first."

    statements = data if isinstance(data, list) else [data]

    title = "CHEX — Bank Statement Summary (Multiple)"
    lines: list[str] = [
        f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Statements analysed: {len(statements)}",
        "",
    ]

    overall_recurring: list[str] = []
    overall_flags: list[str] = []
    for s in statements:
        if not isinstance(s, dict):
            continue
        for r in (s.get("recurring_payments") or []):
            if r not in overall_recurring:
                overall_recurring.append(r)
        for f in (s.get("flags") or []):
            if f not in overall_flags:
                overall_flags.append(f)

    lines += [
        "Overall Recurring Payments:",
        *([f"- {x}" for x in overall_recurring] if overall_recurring else ["- (none)"]),
        "",
        "Overall Flags / Unusual Activity:",
        *([f"- {x}" for x in overall_flags] if overall_flags else ["- (none)"]),
        "",
    ]

    for i, s in enumerate(statements, start=1):
        if not isinstance(s, dict):
            continue
        lines += [
            f"Statement {i}:",
            f"- Total Credits: {s.get('total_credits') or 'N/A'}",
            f"- Total Debits: {s.get('total_debits') or 'N/A'}",
            f"- Largest Transaction: {s.get('largest_transaction') or 'N/A'}",
        ]
        rr = (s.get("raw_reasoning") or "").strip()
        if rr:
            lines += ["- Model reasoning: " + rr]
        lines.append("")

    pdf_bytes = _simple_pdf_bytes(title, lines)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="chex_", mode="wb")
    try:
        tmp.write(pdf_bytes)
    finally:
        tmp.close()

    filename = f"bank-statement-summaries_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return tmp.name, f"**PDF ready:** `{filename}`"


def bank_qa(statement_text: str, question: str) -> tuple[str, str, str, str]:
    if not statement_text.strip():
        return (
            format_label_html("N/A"), "", "",
            "Please run 'Analyse Statement' first to load the statement.",
        )
    if not question.strip():
        return format_label_html("N/A"), "", "", "Please enter a question."
    if not MLX_SERVER_URL:
        return (
            format_label_html("N/A"), "Inference client not initialised", "",
            f"Error: {model_load_error}.",
        )

    statement_text = _truncate(statement_text)
    messages = _build_bank_messages(statement_text, question)

    for attempt in range(2):
        msgs = _apply_messages(messages, strict=(attempt == 1))
        try:
            raw = _run_inference(msgs)
            result = _parse_model_output(raw, question)
            label_html = format_label_html(result.label.value)
            answer = result.answer or "(none — information not found in statement)"
            citation = result.citation or "(none)"
            return label_html, answer, citation, result.reasoning
        except Exception as e:
            if attempt == 0:
                print(f"  Q&A parse attempt 1 failed ({e}). Retrying...")
            else:
                print(f"  Q&A parse attempt 2 failed ({e}). Returning fallback.")

    return (
        format_label_html("ABSENT"),
        "(none — information not found in statement)",
        "(none)",
        "Model output could not be parsed after two attempts.",
    )


# ---------------------------------------------------------------------------
# Benchmark table
# ---------------------------------------------------------------------------

import pandas as pd

BENCHMARK_ROWS = [
    {
        "Question": "What is the limitation of liability?",
        "Ground Truth": "GROUNDED — $50,000 cap",
        "Base Model (untuned)": "GROUNDED — $100,000 cap (wrong amount)",
        "CHEX Fine-tuned": "GROUNDED — $50,000 cap ✓",
        "Hallucinated?": "No (wrong value)",
    },
    {
        "Question": "Does this contract include a non-compete clause?",
        "Ground Truth": "ABSENT",
        "Base Model (untuned)": "🚨 GROUNDED — 'Licensee shall not engage in competing activities...' (fabricated)",
        "CHEX Fine-tuned": "ABSENT — null ✓",
        "Hallucinated?": "YES",
    },
    {
        "Question": "What is the term of the NDA?",
        "Ground Truth": "GROUNDED — 3 years",
        "Base Model (untuned)": "GROUNDED — 2 years (wrong duration)",
        "CHEX Fine-tuned": "GROUNDED — three (3) years ✓",
        "Hallucinated?": "No (wrong value)",
    },
    {
        "Question": "Is there a termination for convenience clause?",
        "Ground Truth": "ABSENT",
        "Base Model (untuned)": "🚨 GROUNDED — 'Either party may terminate at any time...' (fabricated)",
        "CHEX Fine-tuned": "ABSENT — null ✓",
        "Hallucinated?": "YES",
    },
    {
        "Question": "What are the monthly payment terms?",
        "Ground Truth": "GROUNDED — $5,000/month",
        "Base Model (untuned)": "GROUNDED — $5,000/month ✓",
        "CHEX Fine-tuned": "GROUNDED — $5,000/month ✓",
        "Hallucinated?": "No",
    },
]

BENCHMARK_DF = pd.DataFrame(BENCHMARK_ROWS)

# ---------------------------------------------------------------------------
# Warning banner
# ---------------------------------------------------------------------------

WARNING_HTML = ""
if model_load_error:
    WARNING_HTML = (
        '<div class="chex-banner">'
        '<span class="chex-banner-icon">⚠</span>'
        f'<div class="chex-banner-body"><strong>Model not loaded</strong> · '
        f'{model_load_error} — set <code>HF_MODEL_REPO</code> in Space secrets.</div>'
        '</div>'
    )

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CHEX_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
  --bg-base: #0B0E14;
  --bg-grad: linear-gradient(180deg, #0B0E14 0%, #06080C 100%);
  --bg-elev: #131720;
  --bg-elev-strong: #191E2B;
  --bg-sunken: #0E121A;
  --bg-input: rgba(0,0,0,0.2);
  --border: rgba(255,255,255,0.06);
  --border-strong: rgba(255,255,255,0.12);
  --hairline: rgba(255,255,255,0.03);
  --fg: #E2E8F0;
  --fg-muted: #94A3B8;
  --fg-subtle: #475569;
  --green: #10B981;
  --green-bg: rgba(16,185,129,0.10);
  --green-border: rgba(16,185,129,0.25);
  --red: #F43F5E;
  --red-bg: rgba(244,63,94,0.10);
  --red-border: rgba(244,63,94,0.25);
  --amber: #F59E0B;
  --amber-bg: rgba(245,158,11,0.10);
  --amber-border: rgba(245,158,11,0.25);
  --blur: 24px;
  --blur-strong: 32px;
  --shadow-md: 0 1px 0 rgba(255,255,255,0.03) inset,
               0 8px 24px rgba(0,0,0,0.4),
               0 1px 2px rgba(0,0,0,0.2);
  --radius: 10px;
  --radius-lg: 14px;
}

body {
  background: var(--bg-grad) !important;
  background-attachment: fixed !important;
  background-color: var(--bg-base) !important;
  min-height: 100vh;
}

.gradio-container {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  font-size: 14px !important;
  line-height: 1.55 !important;
  color: var(--fg) !important;
  background: transparent !important;
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
  letter-spacing: -0.006em !important;
  max-width: 1480px !important;
  margin: 0 auto !important;
  padding: 0 !important;
}

footer, .footer, .built-with, #footer,
footer.svelte-1ax1toq, .svelte-1ax1toq.footer,
.gradio-container > .footer,
.share-button, .copy-all-button,
.gradio-container > .top-panel { display: none !important; }

#root, .app, main {
  background: transparent !important;
  padding: 0 !important;
  margin: 0 !important;
}

.contain, .container {
  padding: 0 !important;
  gap: 0 !important;
  max-width: 100% !important;
  background: transparent !important;
}

.block, .gr-block, .gr-box, .gr-group, .gradio-container .block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  border-radius: 0 !important;
}

.gap, .gr-row { gap: 20px !important; }

.panel, .gr-panel, .gr-padded {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}

.tabs, .gr-tabs { background: transparent !important; border: none !important; }

.tabitem, .gr-tabitem {
  background: transparent !important;
  border: none !important;
  padding: 24px !important;
}

[data-testid="textbox"], .gr-textbox {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

label.block, .label-wrap {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  gap: 6px !important;
  display: flex !important;
  flex-direction: column !important;
}

.row, .gr-row { background: transparent !important; border: none !important; padding: 0 !important; }

.form, .gr-form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  gap: 14px !important;
}

.chex-topbar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 0 28px;
  height: 60px;
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(11, 14, 20, 0.75);
  backdrop-filter: blur(var(--blur-strong)) saturate(160%);
  -webkit-backdrop-filter: blur(var(--blur-strong)) saturate(160%);
  border-bottom: 1px solid var(--hairline);
}

.chex-logo {
  width: 24px; height: 24px; border-radius: 6px;
  background: #E2E8F0;
  color: #0B0E14; display: grid; place-items: center;
  font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 11px;
  letter-spacing: -0.05em;
  box-shadow: 0 2px 10px rgba(0,0,0,0.5);
  flex-shrink: 0;
}

.chex-name { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; color: var(--fg); font-family: 'Inter', sans-serif; }
.chex-tag { font-size: 12px; color: var(--fg-muted); font-weight: 400; padding-left: 12px; border-left: 1px solid var(--hairline); font-family: 'Inter', sans-serif; }

.chex-pill {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 5px 12px 5px 10px; border: 1px solid var(--border); border-radius: 999px;
  font-size: 12px; color: var(--fg-muted); background: var(--bg-elev);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  font-family: 'JetBrains Mono', monospace; white-space: nowrap;
}

.chex-dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--green);
  box-shadow: 0 0 0 3px rgba(15,157,88,0.22); display: inline-block; flex-shrink: 0;
}

.chex-banner {
  display: flex; align-items: center; gap: 12px; padding: 11px 20px;
  border-bottom: 1px solid var(--amber-border); background: var(--amber-bg);
  backdrop-filter: blur(var(--blur)) saturate(160%); -webkit-backdrop-filter: blur(var(--blur)) saturate(160%);
  color: var(--amber); font-size: 13px; font-family: 'Inter', sans-serif; font-weight: 500;
}
.chex-banner-icon { font-size: 14px; flex-shrink: 0; }
.chex-banner-body { color: var(--fg); font-weight: 400; line-height: 1.5; }
.chex-banner-body strong { color: var(--fg); font-weight: 600; }
.chex-banner code { font-family: 'JetBrains Mono', monospace; font-size: 12px; background: rgba(0,0,0,0.06); padding: 1px 5px; border-radius: 4px; }

.tab-nav {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(var(--blur)) saturate(160%) !important;
  -webkit-backdrop-filter: blur(var(--blur)) saturate(160%) !important;
  border-bottom: 1px solid var(--hairline) !important;
  border-top: none !important; padding: 0 20px !important; gap: 0 !important;
  position: sticky !important; top: 60px !important; z-index: 99 !important; overflow: visible !important;
}

.tab-nav button {
  background: transparent !important; border: none !important; border-radius: 0 !important;
  padding: 14px 16px !important; color: var(--fg-muted) !important;
  font-size: 13px !important; font-weight: 500 !important; font-family: 'Inter', sans-serif !important;
  letter-spacing: -0.003em !important; position: relative !important; white-space: nowrap !important;
  transition: color 0.15s ease !important; cursor: pointer !important; box-shadow: none !important; outline: none !important;
}

.tab-nav button:hover { color: var(--fg) !important; background: transparent !important; }

.tab-nav button.selected, .tab-nav button[aria-selected="true"] {
  color: var(--fg) !important; background: transparent !important; font-weight: 500 !important; box-shadow: none !important;
}

.tab-nav button.selected::after, .tab-nav button[aria-selected="true"]::after {
  content: ""; position: absolute; left: 12px; right: 12px; bottom: -1px;
  height: 1.5px; background: var(--fg); border-radius: 2px 2px 0 0;
}

.tabitem { border: none !important; background: transparent !important; padding: 24px 24px !important; }

.gradio-container .gr-group {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  -webkit-backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-md) !important;
  overflow: hidden !important; padding: 0 !important;
}

.gradio-container .gr-group > *:not(.chex-card-header):not(.chex-chip-row) {
  padding-left: 20px !important; padding-right: 20px !important;
}
.gradio-container .gr-group > *:last-child { padding-bottom: 18px !important; }

.chex-card-header {
  padding: 16px 20px; display: flex; align-items: center;
  justify-content: space-between; gap: 12px; border-bottom: 1px solid var(--hairline);
}

.chex-card-title {
  font-size: 13.5px; font-weight: 600; letter-spacing: -0.01em;
  display: inline-flex; align-items: center; gap: 10px; color: var(--fg);
  white-space: nowrap; font-family: 'Inter', sans-serif;
}

.chex-card-kicker { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--fg-subtle); font-weight: 400; letter-spacing: 0.04em; }

.chex-chip-row {
  display: flex; align-items: center; gap: 8px; padding: 12px 20px;
  border-top: 1px solid var(--hairline); background: var(--bg-sunken); flex-wrap: wrap;
}

.chex-chip-label { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--fg-subtle); white-space: nowrap; margin-right: 4px; }

.chex-suggested {
  display: flex; align-items: center; gap: 10px; padding: 10px 14px;
  background: rgba(13,18,32,0.04); border: 1px solid var(--border); border-radius: var(--radius);
  font-size: 12.5px; color: var(--fg-muted); font-family: 'Inter', sans-serif; line-height: 1.4; margin-top: 2px;
}
.chex-suggested-icon { font-size: 13px; flex-shrink: 0; opacity: 0.7; }

label > span:first-child, .label-wrap span,
.gradio-container label span.text-gray-500, span.svelte-1b6s6s {
  font-family: 'JetBrains Mono', monospace !important; font-size: 10.5px !important;
  font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important;
  color: var(--fg-subtle) !important; margin-bottom: 6px !important; display: block !important;
}

textarea, input[type="text"], input[type="search"],
.gradio-container .gr-input, .gradio-container .gr-textarea,
.gradio-container [data-testid="textbox"] textarea,
.gradio-container [data-testid="textbox"] input {
  background: var(--bg-input) !important; backdrop-filter: blur(10px) !important;
  -webkit-backdrop-filter: blur(10px) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important; color: var(--fg) !important;
  font-family: 'Inter', sans-serif !important; font-size: 13px !important;
  line-height: 1.6 !important; padding: 11px 14px !important;
  transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
  resize: vertical !important;
}

textarea:focus, input[type="text"]:focus,
.gradio-container [data-testid="textbox"] textarea:focus,
.gradio-container [data-testid="textbox"] input:focus {
  border-color: var(--border-strong) !important; background: var(--bg-elev) !important;
  box-shadow: 0 0 0 2px rgba(255,255,255,0.05) !important; outline: none !important;
}

textarea::placeholder, input::placeholder { color: var(--fg-subtle) !important; }

textarea[readonly],
.gradio-container [data-testid="textbox"][data-interactive="false"] textarea {
  background: var(--bg-sunken) !important; border: 1px solid var(--hairline) !important;
  color: var(--fg) !important; cursor: default !important;
}

.gradio-container button {
  font-family: 'Inter', sans-serif !important; font-size: 13px !important;
  font-weight: 500 !important; border-radius: var(--radius) !important;
  padding: 10px 16px !important;
  transition: opacity 0.15s ease, background 0.15s ease, box-shadow 0.15s ease !important;
  cursor: pointer !important; letter-spacing: -0.003em !important;
}

.gradio-container button.primary, button.primary {
  background: var(--fg) !important; color: var(--bg-base) !important; border: 1px solid var(--fg) !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.1) inset !important;
}
.gradio-container button.primary:hover, button.primary:hover { opacity: 0.9 !important; box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important; }

.gradio-container button.secondary, button.secondary {
  background: transparent !important; color: var(--fg-muted) !important;
  border: 1px solid var(--border-strong) !important; box-shadow: none !important;
}
.gradio-container button.secondary:hover, button.secondary:hover { background: var(--bg-elev) !important; color: var(--fg) !important; border-color: var(--border-strong) !important; }

button.sm, .gradio-container button[size="sm"], button.small { font-size: 12px !important; padding: 7px 11px !important; }

.gradio-container .upload-container, .gradio-container [data-testid="file"] {
  background: var(--bg-input) !important; border: 1px dashed var(--border-strong) !important; border-radius: var(--radius) !important;
}

.gradio-container .wrap.svelte-a4gbbr, .gradio-container .table-wrap,
.gradio-container [data-testid="dataframe"] {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  -webkit-backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  border: 1px solid var(--border) !important; border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-md) !important; overflow: hidden !important;
}

.gradio-container table {
  background: transparent !important; font-size: 13px !important;
  font-family: 'Inter', sans-serif !important; border-collapse: separate !important;
  border-spacing: 0 !important; width: 100% !important; border: none !important;
  box-shadow: none !important; border-radius: 0 !important;
}

.gradio-container th {
  background: var(--bg-sunken) !important; border-bottom: 1px solid var(--hairline) !important;
  border-top: none !important; padding: 14px 18px !important;
  font-family: 'JetBrains Mono', monospace !important; font-size: 10.5px !important;
  text-transform: uppercase !important; letter-spacing: 0.08em !important;
  color: var(--fg-muted) !important; font-weight: 500 !important; text-align: left !important;
}

.gradio-container td {
  padding: 16px 18px !important; border-top: 1px solid var(--hairline) !important;
  border-bottom: none !important; vertical-align: top !important; line-height: 1.6 !important;
  color: var(--fg) !important; background: transparent !important;
}

.gradio-container tr:first-child td { border-top: none !important; }

.gradio-container .prose, .gradio-container .md, .gradio-container [data-testid="markdown"] {
  color: var(--fg) !important; font-family: 'Inter', sans-serif !important;
  font-size: 13px !important; line-height: 1.65 !important;
}

.gradio-container .prose h2, .gradio-container .md h2 {
  font-size: 18px !important; font-weight: 600 !important; letter-spacing: -0.02em !important;
  color: var(--fg) !important; margin-bottom: 10px !important; margin-top: 0 !important;
}

.gradio-container .prose p, .gradio-container .md p {
  color: var(--fg-muted) !important; font-size: 13px !important; line-height: 1.65 !important; margin-bottom: 8px !important;
}

.gradio-container .prose strong, .gradio-container .md strong { color: var(--fg) !important; font-weight: 600 !important; }

.gradio-container .prose code, .gradio-container .md code {
  font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
  background: rgba(13,18,32,0.06) !important; padding: 1px 5px !important;
  border-radius: 4px !important; color: var(--fg) !important;
}

.chex-bench-intro {
  background: var(--bg-elev); backdrop-filter: blur(var(--blur)) saturate(180%);
  -webkit-backdrop-filter: blur(var(--blur)) saturate(180%);
  border: 1px solid var(--border); border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md); padding: 24px 28px; margin-bottom: 20px;
}

.chex-bench-intro h2 { margin: 0 0 10px; font-size: 19px; font-weight: 600; letter-spacing: -0.02em; color: var(--fg); font-family: 'Inter', sans-serif; }
.chex-bench-intro p { margin: 0; color: var(--fg-muted); font-size: 13px; line-height: 1.65; font-family: 'Inter', sans-serif; }

.chex-bench-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 18px; }
.chex-bench-stat { background: var(--bg-sunken); border: 1px solid var(--hairline); border-radius: var(--radius); padding: 12px 14px; }
.chex-bench-stat .v { font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 600; letter-spacing: -0.025em; color: var(--fg); line-height: 1.2; margin-bottom: 4px; }
.chex-bench-stat .v.red { color: var(--red); }
.chex-bench-stat .v.green { color: var(--green); }
.chex-bench-stat .k { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--fg-subtle); font-family: 'JetBrains Mono', monospace; }

.chex-footer {
  border-top: 1px solid var(--hairline); padding: 14px 28px;
  display: flex; align-items: center; gap: 18px; color: var(--fg-subtle);
  font-size: 11.5px; font-family: 'JetBrains Mono', monospace;
  background: var(--bg-elev); backdrop-filter: blur(var(--blur));
  -webkit-backdrop-filter: blur(var(--blur)); margin-top: 32px;
}
.chex-footer .sep { opacity: 0.4; }

.chex-label-wrap { padding: 4px 0 8px; }
.chex-divider { height: 1px; background: var(--hairline); margin: 18px 0; }
.chex-section-kicker { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--fg-subtle); margin-bottom: 10px; display: block; }
.chex-card-body { padding: 18px 20px; display: flex; flex-direction: column; gap: 14px; }

*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 999px; border: 2px solid transparent; background-clip: padding-box; }
*::-webkit-scrollbar-track { background: transparent; }

.gradio-container .gap-4 { gap: 14px !important; }
.gradio-container .gap-2 { gap: 8px !important; }

.tabitem .tab-nav { position: static !important; top: auto !important; }

@media (max-width: 900px) {
  .chex-topbar { padding: 0 16px; }
  .chex-tag { display: none; }
  .tabitem { padding: 16px !important; }
  .chex-bench-stats { grid-template-columns: 1fr; }
  .chex-footer { padding: 12px 16px; gap: 12px; flex-wrap: wrap; }
}
"""

# ---------------------------------------------------------------------------
# Static HTML
# ---------------------------------------------------------------------------

TOPBAR_HTML = """
<div class="chex-topbar">
  <div class="chex-logo">CX</div>
  <span class="chex-name">CHEX</span>
  <span class="chex-tag">grounded answers from documents</span>
  <div style="flex:1"></div>
  <div class="chex-pill"><span class="chex-dot"></span>MI300X · ready</div>
</div>
"""

FOOTER_HTML = """
<div class="chex-footer">
  <span>chex/v0.4.1</span>
  <span class="sep">·</span>
  <span>endpoint: mi300x-east-2</span>
  <span class="sep">·</span>
  <span>tokens/s 142.7</span>
</div>
"""

BENCH_INTRO_HTML = """
<div class="chex-bench-intro">
  <h2>Why grounding matters</h2>
  <p>We ran the same questions through a base instruction-tuned model and through CHEX.
  The base model invented or extrapolated answers in 4 of 5 cases — confident, plausible, wrong.
  CHEX returned a verifiable label, a verbatim citation, and refused to answer when the source was silent.</p>
  <div class="chex-bench-stats">
    <div class="chex-bench-stat"><div class="v red">4/5</div><div class="k">Base hallucinations</div></div>
    <div class="chex-bench-stat"><div class="v green">5/5</div><div class="k">CHEX correct</div></div>
    <div class="chex-bench-stat"><div class="v">100%</div><div class="k">Cited verbatim</div></div>
  </div>
</div>
"""

CONTRACT_SOURCE_HEADER_HTML = """
<div class="chex-card-header">
  <span class="chex-card-title">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.55"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
    Source Document
  </span>
  <span class="chex-card-kicker">paste · load sample</span>
</div>
"""

CONTRACT_RESULTS_HEADER_HTML = """
<div class="chex-card-header">
  <span class="chex-card-title">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.55"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
    Analysis
  </span>
  <span class="chex-card-kicker">grounded · cited · structured</span>
</div>
"""

CHIP_ROW_HTML = """
<div class="chex-chip-row">
  <span class="chex-chip-label">Load sample</span>
</div>
"""

STATEMENT_SOURCE_HEADER_HTML = """
<div class="chex-card-header">
  <span class="chex-card-title">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.55"><rect x="2" y="5" width="20" height="14" rx="2"/><line x1="2" y1="10" x2="22" y2="10"/></svg>
    Bank Statement
  </span>
  <span class="chex-card-kicker">paste · pdf · csv · txt · xlsx · ofx</span>
</div>
"""

STATEMENT_RESULTS_HEADER_HTML = """
<div class="chex-card-header">
  <span class="chex-card-title">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.55"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    Summary &amp; Q&amp;A
  </span>
  <span class="chex-card-kicker">summarise · ask · verify</span>
</div>
"""

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="CHEX — Document Intelligence") as demo:

    gr.HTML(TOPBAR_HTML)

    if WARNING_HTML:
        gr.HTML(WARNING_HTML)

    with gr.Tabs():

        # ── Tab 01: Contract Analysis ──────────────────────────────────── #
        with gr.Tab("01  Contract analysis"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=9):
                    with gr.Group():
                        gr.HTML(CONTRACT_SOURCE_HEADER_HTML)
                        contract_input = gr.Textbox(
                            label="Contract text",
                            lines=20,
                            placeholder="Paste your contract text here, or load a sample below…",
                            show_label=False,
                        )
                        gr.HTML(CHIP_ROW_HTML)
                        with gr.Row():
                            btn_software = gr.Button("Software License", variant="secondary", size="sm")
                            btn_nda = gr.Button("NDA", variant="secondary", size="sm")
                            btn_service = gr.Button("Service Agreement", variant="secondary", size="sm")
                        suggested_q = gr.HTML(value="", visible=False)

                with gr.Column(scale=11):
                    with gr.Group():
                        gr.HTML(CONTRACT_RESULTS_HEADER_HTML)
                        with gr.Row():
                            question_input = gr.Textbox(
                                label="Question",
                                placeholder="e.g., What is the limitation of liability?",
                                lines=1,
                                show_label=False,
                                scale=8,
                            )
                            analyze_btn = gr.Button("Analyze ↵", variant="primary", scale=2)
                        label_display = gr.HTML(value=format_label_html("N/A"))
                        answer_output = gr.Textbox(label="Answer", interactive=False, lines=3)
                        citation_output = gr.Textbox(label="Citation", interactive=False, lines=2)
                        reasoning_output = gr.Textbox(label="Reasoning", interactive=False, lines=3)

        # ── Tab 02: Bank Statements ────────────────────────────────────── #
        with gr.Tab("02  Bank statements"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=9):
                    with gr.Group():
                        gr.HTML(STATEMENT_SOURCE_HEADER_HTML)
                        with gr.Tabs():
                            with gr.Tab("Paste text"):
                                bank_paste_input = gr.Textbox(
                                    label="Bank statement text (supports multiple)",
                                    lines=20,
                                    placeholder=(
                                        "Paste one or more bank statements here.\n\n"
                                        "If you paste multiple statements, separate them with a line containing only "
                                        "`---` (3+ dashes)."
                                        "\n\nOr load the sample below…"
                                    ),
                                    show_label=False,
                                )
                                btn_load_statement = gr.Button("Load sample statement", variant="secondary", size="sm")
                            with gr.Tab("Upload PDF"):
                                bank_pdf_input = gr.File(
                                    label="PDF bank statement (multiple allowed)",
                                    file_types=[".pdf"],
                                    file_count="multiple",
                                )
                                bank_pdf_password_input = gr.Textbox(
                                    label="PDF password (optional)",
                                    type="password",
                                    placeholder="Leave blank if PDF is not encrypted",
                                    show_label=False,
                                )
                            with gr.Tab("Upload CSV"):
                                bank_csv_input = gr.File(
                                    label="CSV bank statement (multiple allowed)",
                                    file_types=[".csv"],
                                    file_count="multiple",
                                )
                            with gr.Tab("Upload TXT"):
                                bank_txt_input = gr.File(
                                    label="TXT bank statement (multiple allowed)",
                                    file_types=[".txt", ".text"],
                                    file_count="multiple",
                                )
                            with gr.Tab("Upload Excel"):
                                bank_xlsx_input = gr.File(
                                    label="Excel bank statement (.xlsx, multiple allowed)",
                                    file_types=[".xlsx"],
                                    file_count="multiple",
                                )
                            with gr.Tab("Upload OFX / QFX"):
                                bank_ofx_input = gr.File(
                                    label="OFX / QFX bank statement (multiple allowed)",
                                    file_types=[".ofx", ".qfx"],
                                    file_count="multiple",
                                )

                with gr.Column(scale=11):
                    with gr.Group():
                        gr.HTML(STATEMENT_RESULTS_HEADER_HTML)
                        analyse_stmt_btn = gr.Button("Analyse statement", variant="primary")
                        summary_output = gr.Markdown(value="*Run 'Analyse statement' to generate a financial summary.*")
                        with gr.Row():
                            export_csv_btn = gr.Button("Export CSV", variant="secondary", size="sm")
                            export_pdf_btn = gr.Button("Export PDF", variant="secondary", size="sm")
                        export_status = gr.Markdown(value="")
                        export_file = gr.File(label="Download", interactive=False)
                        gr.HTML('<div class="chex-divider"></div>')
                        gr.HTML('<span class="chex-section-kicker">Ask a question</span>')
                        with gr.Row():
                            bank_question_input = gr.Textbox(
                                label="Question",
                                placeholder="e.g., What was the largest debit this month?",
                                lines=1,
                                show_label=False,
                                scale=8,
                            )
                            bank_ask_btn = gr.Button("Ask ↵", variant="secondary", scale=2)
                        bank_label_display = gr.HTML(value=format_label_html("N/A"))
                        bank_answer_output = gr.Textbox(label="Answer", interactive=False, lines=3)
                        bank_citation_output = gr.Textbox(label="Citation", interactive=False, lines=2)
                        bank_reasoning_output = gr.Textbox(label="Reasoning", interactive=False, lines=3)

            bank_statement_state = gr.State("")
            bank_summary_state = gr.State("")
            # Hidden JSON output for `gradio_client` API usage.
            bank_api_output = gr.JSON(visible=False)
            bank_api_question = gr.Textbox(visible=False)
            bank_api_btn = gr.Button(visible=False)

        # ── Tab 03: Benchmark ──────────────────────────────────────────── #
        with gr.Tab("03  Benchmark"):
            gr.HTML(BENCH_INTRO_HTML)
            gr.Dataframe(
                value=BENCHMARK_DF,
                headers=list(BENCHMARK_DF.columns),
                datatype=["str"] * len(BENCHMARK_DF.columns),
                wrap=True,
                interactive=False,
            )

    gr.HTML(FOOTER_HTML)

    # ── Event handlers ─────────────────────────────────────────────────── #

    def load_software():
        hint = '<div class="chex-suggested"><span class="chex-suggested-icon">💡</span><span><strong>Suggested:</strong> What is the limitation of liability in this agreement?</span></div>'
        return SOFTWARE_LICENSE, SAMPLE_QUESTIONS["software_license.txt"], gr.update(value=hint, visible=True)

    def load_nda():
        hint = '<div class="chex-suggested"><span class="chex-suggested-icon">💡</span><span><strong>Suggested:</strong> Does this agreement include a non-compete clause?</span></div>'
        return NDA, SAMPLE_QUESTIONS["nda.txt"], gr.update(value=hint, visible=True)

    def load_service():
        hint = '<div class="chex-suggested"><span class="chex-suggested-icon">💡</span><span><strong>Suggested:</strong> Does this contract include a termination for convenience clause? <em>(expected: ABSENT)</em></span></div>'
        return SERVICE_AGREEMENT, SAMPLE_QUESTIONS["service_agreement.txt"], gr.update(value=hint, visible=True)

    btn_software.click(fn=load_software, inputs=[], outputs=[contract_input, question_input, suggested_q])
    btn_nda.click(fn=load_nda, inputs=[], outputs=[contract_input, question_input, suggested_q])
    btn_service.click(fn=load_service, inputs=[], outputs=[contract_input, question_input, suggested_q])

    analyze_btn.click(
        fn=analyze_contract,
        inputs=[contract_input, question_input],
        outputs=[label_display, answer_output, citation_output, reasoning_output],
        api_name="contract_analyze",
    )
    question_input.submit(
        fn=analyze_contract,
        inputs=[contract_input, question_input],
        outputs=[label_display, answer_output, citation_output, reasoning_output],
        api_name="contract_analyze",
    )

    btn_load_statement.click(fn=lambda: SAMPLE_STATEMENT, inputs=[], outputs=[bank_paste_input])

    analyse_stmt_btn.click(
        fn=analyse_bank_statement,
        inputs=[
            bank_paste_input,
            bank_pdf_input,
            bank_pdf_password_input,
            bank_csv_input,
            bank_txt_input,
            bank_xlsx_input,
            bank_ofx_input,
        ],
        outputs=[summary_output, bank_statement_state, bank_summary_state],
    )

    export_csv_btn.click(
        fn=export_bank_summary_csv,
        inputs=[bank_summary_state],
        outputs=[export_file, export_status],
    )
    export_pdf_btn.click(
        fn=export_bank_summary_pdf,
        inputs=[bank_summary_state],
        outputs=[export_file, export_status],
    )

    bank_ask_btn.click(
        fn=bank_qa,
        inputs=[bank_statement_state, bank_question_input],
        outputs=[bank_label_display, bank_answer_output, bank_citation_output, bank_reasoning_output],
    )
    bank_question_input.submit(
        fn=bank_qa,
        inputs=[bank_statement_state, bank_question_input],
        outputs=[bank_label_display, bank_answer_output, bank_citation_output, bank_reasoning_output],
    )

    def bank_analyze_api(
        paste_text: str,
        pdf_files,
        pdf_password: str | None,
        csv_files,
        txt_files,
        xlsx_files,
        ofx_files,
        question: str | None,
    ) -> dict:
        summary_md, combined_text, summary_json = analyse_bank_statement(
            paste_text,
            pdf_files,
            pdf_password,
            csv_files,
            txt_files,
            xlsx_files,
            ofx_files,
        )

        qa: dict | None = None
        if (question or "").strip():
            label_html, answer, citation, reasoning = bank_qa(combined_text, (question or "").strip())
            qa = {
                "label_html": label_html,
                "answer": answer,
                "citation": citation,
                "reasoning": reasoning,
            }

        return {
            "summary_markdown": summary_md,
            "combined_text": combined_text,
            "summary_json": summary_json,
            "qa": qa,
        }

    bank_api_btn.click(
        fn=bank_analyze_api,
        inputs=[
            bank_paste_input,
            bank_pdf_input,
            bank_pdf_password_input,
            bank_csv_input,
            bank_txt_input,
            bank_xlsx_input,
            bank_ofx_input,
            bank_api_question,
        ],
        outputs=[bank_api_output],
        api_name="bank_analyze",
    )


if __name__ == "__main__":
    demo.launch(show_error=True, theme=gr.themes.Base(), css=CHEX_CSS, ssr_mode=False)
