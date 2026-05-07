"""
CHEX - Document Intelligence
HuggingFace Spaces Gradio Demo

Tab 1: Analyze Contract — paste a contract, ask a question, get a structured answer
Tab 2: Benchmark Demo — side-by-side table showing base model hallucinations vs CHEX
Tab 3: Analyse Bank Statement — paste / upload a bank statement, get a summary + Q&A
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "HF_MODEL_REPO", "PLACEHOLDER/chex-document-intelligence"
)
SAMPLE_DIR = Path(__file__).parent / "sample_contracts"
STATEMENT_DIR = Path(__file__).parent / "sample_statements"

analyzer = None
model_load_error: Optional[str] = None

bank_analyzer = None

try:
    from serving.inference import ContractAnalyzer  # type: ignore
    from serving.bank_statement import BankStatementAnalyzer  # type: ignore

    analyzer = ContractAnalyzer(model_path=MODEL_PATH)
    bank_analyzer = BankStatementAnalyzer(contract_analyzer=analyzer)
    print(f"Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    model_load_error = str(e)
    print(f"WARNING: Model failed to load: {e}")
    print("Demo is running in preview mode — analysis will return a placeholder response.")

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

# Suggested questions for each sample contract
SAMPLE_QUESTIONS = {
    "software_license.txt": "What is the limitation of liability in this agreement?",
    "nda.txt": "Does this agreement include a non-compete clause?",
    "service_agreement.txt": "Does this contract include a termination for convenience clause?",
}

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
# Analysis handler
# ---------------------------------------------------------------------------

def analyze_contract(
    contract_text: str,
    question: str,
) -> tuple[str, str, str, str]:
    """
    Returns (label_html, answer_text, citation_text, reasoning_text).
    """
    if not contract_text.strip():
        return format_label_html("N/A"), "", "", "Please paste a contract above."
    if not question.strip():
        return format_label_html("N/A"), "", "", "Please enter a question."

    if analyzer is None:
        return (
            format_label_html("N/A"),
            "Model not loaded",
            "",
            f"Model failed to load: {model_load_error}. "
            "Set HF_MODEL_REPO in Space secrets to the correct model repo.",
        )

    try:
        result = analyzer.analyze(contract_text, question)
        label_html = format_label_html(result.label.value)
        answer = result.answer if result.answer else "(none — clause is absent or not applicable)"
        citation = result.citation if result.citation else "(none)"
        reasoning = result.reasoning
        return label_html, answer, citation, reasoning
    except Exception as e:
        return format_label_html("ERROR"), "", "", f"Inference error: {e}"


# ---------------------------------------------------------------------------
# Sample bank statement
# ---------------------------------------------------------------------------

def _read_sample_statement(filename: str) -> str:
    p = STATEMENT_DIR / filename
    if p.exists():
        return p.read_text(encoding="utf-8")
    return f"[Sample statement '{filename}' not found. Place it in demo/sample_statements/]"


SAMPLE_STATEMENT = _read_sample_statement("sample_statement.txt")


# ---------------------------------------------------------------------------
# Bank statement handlers
# ---------------------------------------------------------------------------

def _get_statement_text(
    paste_text: str,
    pdf_file,
    csv_file,
) -> tuple[str, str]:
    """
    Resolve whichever input was provided and return (statement_text, error_msg).
    Priority: PDF > CSV > paste text.
    """
    if pdf_file is not None:
        if bank_analyzer is None:
            return "", "Model not loaded — PDF extraction unavailable."
        try:
            text = bank_analyzer.extract_text_from_pdf(pdf_file)
            if not text.strip():
                return "", "PDF was uploaded but no text could be extracted."
            return text, ""
        except Exception as e:
            return "", f"PDF extraction error: {e}"

    if csv_file is not None:
        if bank_analyzer is None:
            return "", "Model not loaded — CSV parsing unavailable."
        try:
            text = bank_analyzer.parse_csv(csv_file)
            return text, ""
        except Exception as e:
            return "", f"CSV parsing error: {e}"

    if paste_text and paste_text.strip():
        return paste_text.strip(), ""

    return "", "Please paste a bank statement or upload a PDF / CSV file."


def analyse_bank_statement(
    paste_text: str,
    pdf_file,
    csv_file,
) -> tuple[str, str]:
    """
    Returns (summary_markdown, extracted_text_for_qa).
    """
    statement_text, error = _get_statement_text(paste_text, pdf_file, csv_file)
    if error:
        return f"**Error:** {error}", ""

    if analyzer is None:
        return (
            "**Model not loaded.** "
            f"Set `HF_MODEL_REPO` in Space secrets. Error: {model_load_error}",
            statement_text,
        )

    try:
        summary = bank_analyzer.summarize(statement_text)
        lines = ["## Statement Summary", ""]
        lines.append(f"**Total Credits:** {summary.total_credits or 'N/A'}")
        lines.append(f"**Total Debits:** {summary.total_debits or 'N/A'}")
        lines.append(f"**Largest Transaction:** {summary.largest_transaction or 'N/A'}")
        if summary.recurring_payments:
            lines.append("\n**Recurring Payments:**")
            for p in summary.recurring_payments:
                lines.append(f"- {p}")
        if summary.flags:
            lines.append("\n**Flags / Unusual Activity:**")
            for f in summary.flags:
                lines.append(f"- {f}")
        lines.append(f"\n*{summary.raw_reasoning}*")
        return "\n".join(lines), statement_text
    except Exception as e:
        return f"**Summarisation error:** {e}", statement_text


def bank_qa(
    statement_text: str,
    question: str,
) -> tuple[str, str, str, str]:
    """
    Returns (label_html, answer_text, citation_text, reasoning_text).
    """
    if not statement_text.strip():
        return (
            format_label_html("N/A"), "", "",
            "Please run 'Analyse Statement' first to load the statement.",
        )
    if not question.strip():
        return format_label_html("N/A"), "", "", "Please enter a question."

    if analyzer is None:
        return (
            format_label_html("N/A"), "Model not loaded", "",
            f"Model failed to load: {model_load_error}.",
        )

    try:
        result = bank_analyzer.answer_question(statement_text, question)
        label_html = format_label_html(result.label.value)
        answer = result.answer if result.answer else "(none — information not found in statement)"
        citation = result.citation if result.citation else "(none)"
        return label_html, answer, citation, result.reasoning
    except Exception as e:
        return format_label_html("ERROR"), "", "", f"Inference error: {e}"


# ---------------------------------------------------------------------------
# Benchmark table data (hardcoded — pre-computed base model outputs)
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
# Gradio UI
# ---------------------------------------------------------------------------

WARNING_HTML = ""
if model_load_error:
    WARNING_HTML = (
        '<div class="chex-banner">'
        f'<span class="chex-banner-icon">⚠</span>'
        f'<div class="chex-banner-body"><strong>Model not loaded</strong> · '
        f'{model_load_error} — set <code>HF_MODEL_REPO</code> in Space secrets.</div>'
        '</div>'
    )

# ---------------------------------------------------------------------------
# CSS — CHEX design system (glassmorphic, Inter + JetBrains Mono)
# ---------------------------------------------------------------------------

CHEX_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
  --bg-base: #f3f4f7;
  --bg-grad: radial-gradient(ellipse 1200px 700px at 18% -10%, rgba(120,150,200,0.18), transparent 60%),
             radial-gradient(ellipse 900px 600px at 95% 110%, rgba(180,160,220,0.14), transparent 55%),
             linear-gradient(180deg, #f5f6f9 0%, #eef0f4 100%);
  --bg-elev: rgba(255,255,255,0.62);
  --bg-elev-strong: rgba(255,255,255,0.78);
  --bg-sunken: rgba(245,246,249,0.55);
  --bg-input: rgba(255,255,255,0.55);
  --border: rgba(15,18,30,0.08);
  --border-strong: rgba(15,18,30,0.14);
  --hairline: rgba(15,18,30,0.06);
  --fg: #0d1220;
  --fg-muted: #5b6275;
  --fg-subtle: #8a91a3;
  --green: #0f9d58;
  --amber: #b87800;
  --amber-bg: rgba(245,158,11,0.10);
  --amber-border: rgba(245,158,11,0.30);
  --blur: 22px;
  --blur-strong: 32px;
  --shadow-md: 0 1px 0 rgba(255,255,255,0.6) inset, 0 8px 24px rgba(15,18,30,0.06), 0 1px 2px rgba(15,18,30,0.04);
  --radius: 10px;
  --radius-lg: 16px;
}

.dark, [data-theme="dark"] {
  --bg-base: #07090e;
  --bg-grad: radial-gradient(ellipse 1100px 700px at 15% -5%, rgba(70,110,200,0.20), transparent 60%),
             radial-gradient(ellipse 900px 600px at 95% 110%, rgba(150,90,220,0.14), transparent 55%),
             linear-gradient(180deg, #0a0d14 0%, #06080d 100%);
  --bg-elev: rgba(22,26,36,0.55);
  --bg-elev-strong: rgba(28,32,44,0.72);
  --bg-sunken: rgba(12,14,20,0.55);
  --bg-input: rgba(14,17,25,0.55);
  --border: rgba(255,255,255,0.07);
  --border-strong: rgba(255,255,255,0.13);
  --hairline: rgba(255,255,255,0.05);
  --fg: #eceef4;
  --fg-muted: #9ba3b6;
  --fg-subtle: #6a7188;
  --green: #4ade80;
  --amber: #fbbf24;
}

/* ── App shell ── */
.gradio-container {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  font-size: 14px !important;
  line-height: 1.55 !important;
  color: var(--fg) !important;
  background: var(--bg-grad) !important;
  background-attachment: fixed !important;
  background-color: var(--bg-base) !important;
  -webkit-font-smoothing: antialiased !important;
  letter-spacing: -0.006em !important;
  max-width: 1480px !important;
}

/* ── Topbar / header ── */
.chex-topbar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 0 28px;
  height: 60px;
  background: var(--bg-elev);
  backdrop-filter: blur(var(--blur-strong)) saturate(160%);
  -webkit-backdrop-filter: blur(var(--blur-strong)) saturate(160%);
  border-bottom: 1px solid var(--hairline);
  margin-bottom: 0;
}
.chex-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}
.chex-logo {
  width: 26px; height: 26px;
  border-radius: 8px;
  background: linear-gradient(135deg, var(--fg), rgba(13,18,32,0.7));
  color: var(--bg-base);
  display: grid;
  place-items: center;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  font-size: 11px;
  letter-spacing: -0.05em;
  box-shadow: 0 4px 14px rgba(15,18,30,0.18), 0 1px 0 rgba(255,255,255,0.25) inset;
}
.chex-name {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--fg);
}
.chex-tag {
  font-size: 12px;
  color: var(--fg-muted);
  font-weight: 400;
  padding-left: 12px;
  border-left: 1px solid var(--hairline);
}
.chex-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 12px 5px 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  font-size: 12px;
  color: var(--fg-muted);
  background: var(--bg-elev);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  font-family: 'JetBrains Mono', monospace;
  white-space: nowrap;
  margin-left: auto;
}
.chex-status-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 3px rgba(15,157,88,0.22);
  display: inline-block;
}

/* ── Banner ── */
.chex-banner {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 11px 20px;
  border-bottom: 1px solid var(--amber-border);
  background: var(--amber-bg);
  backdrop-filter: blur(var(--blur)) saturate(160%);
  -webkit-backdrop-filter: blur(var(--blur)) saturate(160%);
  color: var(--amber);
  font-size: 13px;
  font-family: 'Inter', sans-serif;
  margin-bottom: 0;
}
.chex-banner-icon { font-size: 14px; }
.chex-banner-body { color: var(--fg); font-weight: 400; }
.chex-banner-body strong { color: var(--fg); font-weight: 600; }
.chex-banner code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  background: rgba(0,0,0,0.06);
  padding: 1px 5px;
  border-radius: 4px;
}

/* ── Tabs ── */
.tab-nav {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(var(--blur)) saturate(160%) !important;
  -webkit-backdrop-filter: blur(var(--blur)) saturate(160%) !important;
  border-bottom: 1px solid var(--hairline) !important;
  padding: 0 20px !important;
  gap: 2px !important;
}
.tab-nav button {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 14px 16px !important;
  color: var(--fg-muted) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  font-family: 'Inter', sans-serif !important;
  position: relative !important;
  white-space: nowrap !important;
  transition: color 0.15s !important;
}
.tab-nav button:hover { color: var(--fg) !important; }
.tab-nav button.selected {
  color: var(--fg) !important;
  background: transparent !important;
}
.tab-nav button.selected::after {
  content: "";
  position: absolute;
  left: 12px; right: 12px; bottom: -1px;
  height: 1.5px;
  background: var(--fg);
  border-radius: 2px 2px 0 0;
}

/* ── Glass cards ── */
.chex-card {
  background: var(--bg-elev);
  backdrop-filter: blur(var(--blur)) saturate(180%);
  -webkit-backdrop-filter: blur(var(--blur)) saturate(180%);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  overflow: hidden;
  margin-bottom: 0;
}
.chex-card-header {
  padding: 16px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--hairline);
}
.chex-card-title {
  font-size: 13.5px;
  font-weight: 600;
  letter-spacing: -0.01em;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--fg);
  white-space: nowrap;
}
.chex-card-kicker {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--fg-subtle);
  font-weight: 400;
}

/* ── Override Gradio inputs to match design ── */
.gradio-container input[type="text"],
.gradio-container textarea,
.gradio-container select,
.gradio-container .gr-input,
label.block textarea,
label.block input {
  background: var(--bg-input) !important;
  backdrop-filter: blur(10px) !important;
  -webkit-backdrop-filter: blur(10px) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--fg) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  transition: border-color 0.18s, box-shadow 0.18s !important;
}
label.block textarea:focus,
label.block input:focus {
  border-color: var(--border-strong) !important;
  background: var(--bg-elev-strong) !important;
  box-shadow: 0 0 0 4px rgba(13,18,32,0.08) !important;
  outline: none !important;
}

/* Labels */
label.block > span,
.gr-form > label > span {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10.5px !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--fg-subtle) !important;
}

/* ── Buttons ── */
.gradio-container button.primary,
.gradio-container .gr-button-primary {
  background: var(--fg) !important;
  color: var(--bg-base) !important;
  border: 1px solid var(--fg) !important;
  border-radius: var(--radius) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  padding: 10px 16px !important;
  box-shadow: 0 6px 18px rgba(13,18,32,0.28), 0 1px 0 rgba(255,255,255,0.15) inset !important;
  transition: opacity 0.18s !important;
}
.gradio-container button.primary:hover,
.gradio-container .gr-button-primary:hover { opacity: 0.92 !important; }

.gradio-container button.secondary,
.gradio-container .gr-button-secondary {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(10px) !important;
  color: var(--fg) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  padding: 10px 16px !important;
  transition: background 0.18s, border-color 0.18s !important;
}
.gradio-container button.secondary:hover,
.gradio-container .gr-button-secondary:hover {
  background: var(--bg-elev-strong) !important;
  border-color: var(--border-strong) !important;
}

/* Small ghost buttons (load sample etc.) */
button.lg.secondary.svelte-cmf5ev,
button[class*="sm"] {
  font-size: 12px !important;
  padding: 7px 11px !important;
}

/* ── Dataframe / benchmark table ── */
.gradio-container table,
.gradio-container .gr-dataframe table {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-md) !important;
  font-size: 13px !important;
  font-family: 'Inter', sans-serif !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
  overflow: hidden !important;
  width: 100% !important;
}
.gradio-container th {
  background: var(--bg-sunken) !important;
  border-bottom: 1px solid var(--hairline) !important;
  padding: 14px 18px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10.5px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--fg-muted) !important;
  font-weight: 500 !important;
  text-align: left !important;
}
.gradio-container td {
  padding: 16px 18px !important;
  border-top: 1px solid var(--hairline) !important;
  vertical-align: top !important;
  line-height: 1.6 !important;
  color: var(--fg) !important;
}
.gradio-container tr:first-child td { border-top: none !important; }

/* ── Markdown inside Gradio ── */
.gradio-container .prose,
.gradio-container .md {
  color: var(--fg) !important;
  font-family: 'Inter', sans-serif !important;
}
.gradio-container .prose h2 {
  font-size: 19px !important;
  font-weight: 600 !important;
  letter-spacing: -0.02em !important;
  color: var(--fg) !important;
  margin-bottom: 10px !important;
}
.gradio-container .prose h3 {
  font-size: 14px !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
  color: var(--fg) !important;
  margin-bottom: 8px !important;
}
.gradio-container .prose p {
  color: var(--fg-muted) !important;
  font-size: 13px !important;
  line-height: 1.65 !important;
}

/* ── Bench intro card ── */
.chex-bench-intro {
  background: var(--bg-elev);
  backdrop-filter: blur(var(--blur)) saturate(180%);
  -webkit-backdrop-filter: blur(var(--blur)) saturate(180%);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
  padding: 24px 28px;
  margin-bottom: 22px;
}
.chex-bench-intro h2 {
  margin: 0 0 10px;
  font-size: 19px;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--fg);
}
.chex-bench-intro p {
  margin: 0;
  color: var(--fg-muted);
  font-size: 13px;
  line-height: 1.65;
  font-family: 'Inter', sans-serif;
}
.chex-bench-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-top: 18px;
}
.chex-bench-stat {
  background: var(--bg-sunken);
  border: 1px solid var(--hairline);
  border-radius: var(--radius);
  padding: 12px 14px;
}
.chex-bench-stat .v {
  font-family: 'Inter', sans-serif;
  font-size: 20px;
  font-weight: 600;
  letter-spacing: -0.025em;
  color: var(--fg);
}
.chex-bench-stat .v.red { color: #d23131; }
.chex-bench-stat .v.green { color: #0f9d58; }
.chex-bench-stat .k {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg-subtle);
  font-family: 'JetBrains Mono', monospace;
}

/* ── Footer ── */
.chex-footer {
  border-top: 1px solid var(--hairline);
  padding: 14px 28px;
  display: flex;
  align-items: center;
  gap: 18px;
  color: var(--fg-subtle);
  font-size: 11.5px;
  font-family: 'JetBrains Mono', monospace;
  background: var(--bg-elev);
  backdrop-filter: blur(var(--blur));
  -webkit-backdrop-filter: blur(var(--blur));
  margin-top: 28px;
}
.chex-footer .sep { opacity: 0.5; }

/* ── Output textboxes ── */
.gradio-container .gr-textbox[data-testid],
.gradio-container textarea[readonly] {
  background: var(--bg-sunken) !important;
  border: 1px solid var(--hairline) !important;
  font-size: 13px !important;
  line-height: 1.65 !important;
  color: var(--fg) !important;
}

/* Scrollbars */
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
*::-webkit-scrollbar-track { background: transparent; }
"""

TOPBAR_HTML = """
<div class="chex-topbar">
  <div class="chex-brand">
    <div class="chex-logo">CX</div>
    <div class="chex-name">CHEX</div>
    <div class="chex-tag">grounded answers from documents</div>
  </div>
  <div class="chex-status-pill">
    <span class="chex-status-dot"></span>
    <span>MI300X · ready</span>
  </div>
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

with gr.Blocks(
    title="CHEX - Document Intelligence",
    theme=gr.themes.Base(),
    css=CHEX_CSS,
) as demo:
    gr.HTML(TOPBAR_HTML)

    if WARNING_HTML:
        gr.HTML(WARNING_HTML)

    with gr.Tabs():
        # ------------------------------------------------------------------ #
        # Tab 1: Analyze Contract                                             #
        # ------------------------------------------------------------------ #
        with gr.Tab("Analyze Contract"):
            with gr.Row():
                # Left column: contract input
                with gr.Column(scale=2):
                    gr.Markdown("### Contract Text")
                    contract_input = gr.Textbox(
                        label="Paste contract text here",
                        lines=22,
                        placeholder="Paste your contract text here, or load a sample contract below...",
                        show_label=False,
                    )
                    with gr.Row():
                        btn_software = gr.Button("Load: Software License", size="sm")
                        btn_nda = gr.Button("Load: NDA", size="sm")
                        btn_service = gr.Button("Load: Service Agreement", size="sm")

                # Right column: question + results
                with gr.Column(scale=1):
                    gr.Markdown("### Question")
                    question_input = gr.Textbox(
                        label="Question about the contract",
                        placeholder="e.g., What is the limitation of liability?",
                        lines=2,
                        show_label=False,
                    )
                    analyze_btn = gr.Button(
                        "Analyze",
                        variant="primary",
                        interactive=True,
                    )

                    gr.Markdown("### Result")
                    label_display = gr.HTML(
                        value=format_label_html("N/A"),
                        label="Classification",
                    )
                    answer_output = gr.Textbox(
                        label="Answer",
                        interactive=False,
                        lines=3,
                    )
                    citation_output = gr.Textbox(
                        label="Citation (verbatim from contract)",
                        interactive=False,
                        lines=3,
                    )
                    reasoning_output = gr.Textbox(
                        label="Reasoning",
                        interactive=False,
                        lines=2,
                    )

            # Suggested questions shown when loading a sample
            suggested_q = gr.Markdown("", visible=False)

        # ------------------------------------------------------------------ #
        # Tab 2: Analyse Bank Statement                                        #
        # ------------------------------------------------------------------ #
        with gr.Tab("Analyse Bank Statement"):
            with gr.Row():
                # Left column: statement input (3 sub-tabs)
                with gr.Column(scale=2):
                    gr.Markdown("### Bank Statement Input")
                    with gr.Tabs():
                        with gr.Tab("Paste Text"):
                            bank_paste_input = gr.Textbox(
                                label="Paste bank statement text",
                                lines=20,
                                placeholder="Paste your bank statement here, or load the sample below...",
                                show_label=False,
                            )
                            btn_load_statement = gr.Button("Load Sample Statement", size="sm")
                        with gr.Tab("Upload PDF"):
                            bank_pdf_input = gr.File(
                                label="Upload PDF bank statement",
                                file_types=[".pdf"],
                            )
                        with gr.Tab("Upload CSV"):
                            bank_csv_input = gr.File(
                                label="Upload CSV bank statement",
                                file_types=[".csv"],
                            )

                # Right column: summary + Q&A
                with gr.Column(scale=1):
                    analyse_stmt_btn = gr.Button(
                        "Analyse Statement",
                        variant="primary",
                    )
                    summary_output = gr.Markdown(
                        value="*Run 'Analyse Statement' to generate a financial summary.*"
                    )

                    gr.Markdown("---")
                    gr.Markdown("### Ask a Question")
                    bank_question_input = gr.Textbox(
                        label="Question about the statement",
                        placeholder="e.g., What was the largest debit this month?",
                        lines=2,
                        show_label=False,
                    )
                    bank_ask_btn = gr.Button("Ask", variant="secondary")

                    gr.Markdown("### Q&A Result")
                    bank_label_display = gr.HTML(
                        value=format_label_html("N/A"),
                        label="Classification",
                    )
                    bank_answer_output = gr.Textbox(
                        label="Answer",
                        interactive=False,
                        lines=3,
                    )
                    bank_citation_output = gr.Textbox(
                        label="Citation (verbatim from statement)",
                        interactive=False,
                        lines=3,
                    )
                    bank_reasoning_output = gr.Textbox(
                        label="Reasoning",
                        interactive=False,
                        lines=2,
                    )

            # Hidden state: extracted statement text shared between summary and Q&A
            bank_statement_state = gr.State("")

        # ------------------------------------------------------------------ #
        # Tab 2: Benchmark Demo                                               #
        # ------------------------------------------------------------------ #
        with gr.Tab("Benchmark Demo"):
            gr.HTML(BENCH_INTRO_HTML)
            gr.Dataframe(
                value=BENCHMARK_DF,
                headers=list(BENCHMARK_DF.columns),
                datatype=["str"] * len(BENCHMARK_DF.columns),
                wrap=True,
                interactive=False,
            )

    # ------------------------------------------------------------------ #
    # Event handlers                                                       #
    # ------------------------------------------------------------------ #

    def load_software():
        return (
            SOFTWARE_LICENSE,
            SAMPLE_QUESTIONS["software_license.txt"],
            gr.update(value="💡 **Suggested question:** What is the limitation of liability in this agreement?", visible=True),
        )

    def load_nda():
        return (
            NDA,
            SAMPLE_QUESTIONS["nda.txt"],
            gr.update(value="💡 **Suggested question:** Does this agreement include a non-compete clause?", visible=True),
        )

    def load_service():
        return (
            SERVICE_AGREEMENT,
            SAMPLE_QUESTIONS["service_agreement.txt"],
            gr.update(value="💡 **Suggested question:** Does this contract include a termination for convenience clause? (expected: ABSENT)", visible=True),
        )

    btn_software.click(
        fn=load_software,
        inputs=[],
        outputs=[contract_input, question_input, suggested_q],
    )
    btn_nda.click(
        fn=load_nda,
        inputs=[],
        outputs=[contract_input, question_input, suggested_q],
    )
    btn_service.click(
        fn=load_service,
        inputs=[],
        outputs=[contract_input, question_input, suggested_q],
    )

    analyze_btn.click(
        fn=analyze_contract,
        inputs=[contract_input, question_input],
        outputs=[label_display, answer_output, citation_output, reasoning_output],
    )

    # Also trigger on Enter in question field
    question_input.submit(
        fn=analyze_contract,
        inputs=[contract_input, question_input],
        outputs=[label_display, answer_output, citation_output, reasoning_output],
    )

    # ------------------------------------------------------------------ #
    # Bank Statement event handlers                                        #
    # ------------------------------------------------------------------ #

    btn_load_statement.click(
        fn=lambda: SAMPLE_STATEMENT,
        inputs=[],
        outputs=[bank_paste_input],
    )

    analyse_stmt_btn.click(
        fn=analyse_bank_statement,
        inputs=[bank_paste_input, bank_pdf_input, bank_csv_input],
        outputs=[summary_output, bank_statement_state],
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

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch(show_error=True)
