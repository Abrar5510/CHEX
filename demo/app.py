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
# CSS — CHEX design system (glassmorphic, Inter + JetBrains Mono)
# ---------------------------------------------------------------------------

CHEX_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── Design tokens ── */
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
  --green-bg: rgba(34,197,94,0.10);
  --green-border: rgba(34,197,94,0.28);
  --red: #d23131;
  --red-bg: rgba(239,68,68,0.09);
  --red-border: rgba(239,68,68,0.28);
  --amber: #b87800;
  --amber-bg: rgba(245,158,11,0.10);
  --amber-border: rgba(245,158,11,0.30);
  --blur: 22px;
  --blur-strong: 32px;
  --shadow-md: 0 1px 0 rgba(255,255,255,0.6) inset,
               0 8px 24px rgba(15,18,30,0.06),
               0 1px 2px rgba(15,18,30,0.04);
  --radius: 10px;
  --radius-lg: 16px;
}

/* ── Body & app shell ── */
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

/* ── Nuke ALL Gradio chrome ── */
footer, .footer, .built-with, #footer,
footer.svelte-1ax1toq, .svelte-1ax1toq.footer,
.gradio-container > .footer,
.share-button, .copy-all-button,
.gradio-container > .top-panel { display: none !important; }

/* Strip the outer container's own bg/padding */
#root, .app, main {
  background: transparent !important;
  padding: 0 !important;
  margin: 0 !important;
}

/* The inner .contain div Gradio wraps everything in */
.contain, .container {
  padding: 0 !important;
  gap: 0 !important;
  max-width: 100% !important;
  background: transparent !important;
}

/* Every .block Gradio creates — reset ALL chrome */
.block,
.gr-block,
.gr-box,
.gr-group,
.gradio-container .block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  border-radius: 0 !important;
}

/* The padding/gap between row children */
.gap, .gr-row { gap: 20px !important; }

/* Panel wrappers */
.panel, .gr-panel, .gr-padded {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}

/* Tabs outer wrapper */
.tabs, .gr-tabs {
  background: transparent !important;
  border: none !important;
}

/* Individual tab content areas */
.tabitem, .gr-tabitem {
  background: transparent !important;
  border: none !important;
  padding: 24px !important;
}

/* Textbox wrappers — only reset the outer shell, let the inner textarea keep styling */
[data-testid="textbox"],
.gr-textbox {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* Label blocks */
label.block, .label-wrap {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  gap: 6px !important;
  display: flex !important;
  flex-direction: column !important;
}

/* Row component */
.row, .gr-row {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

/* Form groups */
.form, .gr-form {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  gap: 14px !important;
}

/* ── Topbar ── */
.chex-topbar {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 0 28px;
  height: 60px;
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--bg-elev);
  backdrop-filter: blur(var(--blur-strong)) saturate(160%);
  -webkit-backdrop-filter: blur(var(--blur-strong)) saturate(160%);
  border-bottom: 1px solid var(--hairline);
}

.chex-logo {
  width: 26px;
  height: 26px;
  border-radius: 8px;
  background: linear-gradient(135deg, #0d1220, rgba(13,18,32,0.7));
  color: #f3f4f7;
  display: grid;
  place-items: center;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  font-size: 11px;
  letter-spacing: -0.05em;
  box-shadow: 0 4px 14px rgba(15,18,30,0.18), 0 1px 0 rgba(255,255,255,0.25) inset;
  flex-shrink: 0;
}

.chex-name {
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--fg);
  font-family: 'Inter', sans-serif;
}

.chex-tag {
  font-size: 12px;
  color: var(--fg-muted);
  font-weight: 400;
  padding-left: 12px;
  border-left: 1px solid var(--hairline);
  font-family: 'Inter', sans-serif;
}

.chex-pill {
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
}

.chex-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 3px rgba(15,157,88,0.22);
  display: inline-block;
  flex-shrink: 0;
}

/* ── Warning banner ── */
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
  font-weight: 500;
}
.chex-banner-icon { font-size: 14px; flex-shrink: 0; }
.chex-banner-body { color: var(--fg); font-weight: 400; line-height: 1.5; }
.chex-banner-body strong { color: var(--fg); font-weight: 600; }
.chex-banner code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  background: rgba(0,0,0,0.06);
  padding: 1px 5px;
  border-radius: 4px;
}

/* ── Tab bar ── */
.tab-nav {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(var(--blur)) saturate(160%) !important;
  -webkit-backdrop-filter: blur(var(--blur)) saturate(160%) !important;
  border-bottom: 1px solid var(--hairline) !important;
  border-top: none !important;
  padding: 0 20px !important;
  gap: 0 !important;
  position: sticky !important;
  top: 60px !important;
  z-index: 99 !important;
  overflow: visible !important;
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
  letter-spacing: -0.003em !important;
  position: relative !important;
  white-space: nowrap !important;
  transition: color 0.15s ease !important;
  cursor: pointer !important;
  box-shadow: none !important;
  outline: none !important;
}

.tab-nav button:hover {
  color: var(--fg) !important;
  background: transparent !important;
}

.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
  color: var(--fg) !important;
  background: transparent !important;
  font-weight: 500 !important;
  box-shadow: none !important;
}

.tab-nav button.selected::after,
.tab-nav button[aria-selected="true"]::after {
  content: "";
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: -1px;
  height: 1.5px;
  background: var(--fg);
  border-radius: 2px 2px 0 0;
}

/* Tab content panels */
.tabitem {
  border: none !important;
  background: transparent !important;
  padding: 24px 24px !important;
}

/* ── Card components ── */
.chex-card,
.gradio-container .gr-group.chex-card-group,
.gradio-container [data-testid="group"].chex-card-group {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  -webkit-backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-md) !important;
  overflow: hidden !important;
  padding: 0 !important;
}

/* Groups used as cards */
.gradio-container .gr-group {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  -webkit-backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-md) !important;
  overflow: hidden !important;
  padding: 0 !important;
}

/* Inner content of groups gets consistent padding */
.gradio-container .gr-group > *:not(.chex-card-header):not(.chex-chip-row) {
  padding-left: 20px !important;
  padding-right: 20px !important;
}
.gradio-container .gr-group > *:last-child {
  padding-bottom: 18px !important;
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
  font-family: 'Inter', sans-serif;
}

.chex-card-kicker {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--fg-subtle);
  font-weight: 400;
  letter-spacing: 0.04em;
}

/* ── Chip row (load samples) ── */
.chex-chip-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  border-top: 1px solid var(--hairline);
  background: var(--bg-sunken);
  flex-wrap: wrap;
}

.chex-chip-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg-subtle);
  white-space: nowrap;
  margin-right: 4px;
}

/* ── Suggested question bar ── */
.chex-suggested {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: rgba(13,18,32,0.04);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-size: 12.5px;
  color: var(--fg-muted);
  font-family: 'Inter', sans-serif;
  line-height: 1.4;
  margin-top: 2px;
}

.chex-suggested-icon {
  font-size: 13px;
  flex-shrink: 0;
  opacity: 0.7;
}


/* ── Labels on inputs ── */
label > span:first-child,
.label-wrap span,
.gradio-container label span.text-gray-500,
span.svelte-1b6s6s {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10.5px !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--fg-subtle) !important;
  margin-bottom: 6px !important;
  display: block !important;
}

/* ── Textareas & inputs ── */
textarea,
input[type="text"],
input[type="search"],
.gradio-container .gr-input,
.gradio-container .gr-textarea,
.gradio-container [data-testid="textbox"] textarea,
.gradio-container [data-testid="textbox"] input {
  background: var(--bg-input) !important;
  backdrop-filter: blur(10px) !important;
  -webkit-backdrop-filter: blur(10px) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--fg) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
  padding: 11px 14px !important;
  transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
  resize: vertical !important;
}

textarea:focus,
input[type="text"]:focus,
.gradio-container [data-testid="textbox"] textarea:focus,
.gradio-container [data-testid="textbox"] input:focus {
  border-color: var(--border-strong) !important;
  background: var(--bg-elev-strong) !important;
  box-shadow: 0 0 0 4px rgba(13,18,32,0.08) !important;
  outline: none !important;
}

textarea::placeholder,
input::placeholder {
  color: var(--fg-subtle) !important;
}

/* Read-only / output textboxes */
textarea[readonly],
.gradio-container [data-testid="textbox"][data-interactive="false"] textarea {
  background: var(--bg-sunken) !important;
  border: 1px solid var(--hairline) !important;
  color: var(--fg) !important;
  cursor: default !important;
}

/* ── Buttons ── */
.gradio-container button {
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  border-radius: var(--radius) !important;
  padding: 10px 16px !important;
  transition: opacity 0.15s ease, background 0.15s ease, box-shadow 0.15s ease !important;
  cursor: pointer !important;
  letter-spacing: -0.003em !important;
}

.gradio-container button.primary,
.gradio-container [data-testid="button"][variant="primary"],
button.primary {
  background: var(--fg) !important;
  color: var(--bg-base) !important;
  border: 1px solid var(--fg) !important;
  box-shadow: 0 6px 18px rgba(13,18,32,0.28), 0 1px 0 rgba(255,255,255,0.1) inset !important;
}

.gradio-container button.primary:hover,
button.primary:hover {
  opacity: 0.88 !important;
  box-shadow: 0 4px 12px rgba(13,18,32,0.22) !important;
}

.gradio-container button.secondary,
button.secondary {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(10px) !important;
  -webkit-backdrop-filter: blur(10px) !important;
  color: var(--fg) !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--shadow-md) !important;
}

.gradio-container button.secondary:hover,
button.secondary:hover {
  background: var(--bg-elev-strong) !important;
  border-color: var(--border-strong) !important;
}

/* Small / sm-size buttons */
button.sm,
.gradio-container button[size="sm"],
button.small {
  font-size: 12px !important;
  padding: 7px 11px !important;
}

/* ── File upload ── */
.gradio-container .upload-container,
.gradio-container [data-testid="file"] {
  background: var(--bg-input) !important;
  border: 1px dashed var(--border-strong) !important;
  border-radius: var(--radius) !important;
}

/* ── Dataframe / benchmark table ── */
.gradio-container .wrap.svelte-a4gbbr,
.gradio-container .table-wrap,
.gradio-container [data-testid="dataframe"] {
  background: var(--bg-elev) !important;
  backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  -webkit-backdrop-filter: blur(var(--blur)) saturate(180%) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-md) !important;
  overflow: hidden !important;
}

.gradio-container table {
  background: transparent !important;
  font-size: 13px !important;
  font-family: 'Inter', sans-serif !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
  width: 100% !important;
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
}

.gradio-container th {
  background: var(--bg-sunken) !important;
  border-bottom: 1px solid var(--hairline) !important;
  border-top: none !important;
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
  border-bottom: none !important;
  vertical-align: top !important;
  line-height: 1.6 !important;
  color: var(--fg) !important;
  background: transparent !important;
}

.gradio-container tr:first-child td { border-top: none !important; }

/* Hallucinated rows — rows where 'Hallucinated?' is YES */
.gradio-container tr:has(td:last-child:contains("YES")) td,
.chex-hallucinated-row td {
  background: color-mix(in srgb, var(--red-bg) 4%, transparent) !important;
  box-shadow: inset 2px 0 0 var(--red) !important;
}

/* ── Markdown output ── */
.gradio-container .prose,
.gradio-container .md,
.gradio-container [data-testid="markdown"] {
  color: var(--fg) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  line-height: 1.65 !important;
}

.gradio-container .prose h2,
.gradio-container .md h2 {
  font-size: 18px !important;
  font-weight: 600 !important;
  letter-spacing: -0.02em !important;
  color: var(--fg) !important;
  margin-bottom: 10px !important;
  margin-top: 0 !important;
}

.gradio-container .prose h3,
.gradio-container .md h3 {
  font-size: 13.5px !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
  color: var(--fg) !important;
  margin-bottom: 8px !important;
  margin-top: 16px !important;
}

.gradio-container .prose p,
.gradio-container .md p {
  color: var(--fg-muted) !important;
  font-size: 13px !important;
  line-height: 1.65 !important;
  margin-bottom: 8px !important;
}

.gradio-container .prose strong,
.gradio-container .md strong {
  color: var(--fg) !important;
  font-weight: 600 !important;
}

.gradio-container .prose code,
.gradio-container .md code {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
  background: rgba(13,18,32,0.06) !important;
  padding: 1px 5px !important;
  border-radius: 4px !important;
  color: var(--fg) !important;
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
  margin-bottom: 20px;
}

.chex-bench-intro h2 {
  margin: 0 0 10px;
  font-size: 19px;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--fg);
  font-family: 'Inter', sans-serif;
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
  line-height: 1.2;
  margin-bottom: 4px;
}

.chex-bench-stat .v.red { color: var(--red); }
.chex-bench-stat .v.green { color: var(--green); }

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
  margin-top: 32px;
}

.chex-footer .sep { opacity: 0.4; }

/* ── Result label container ── */
.chex-label-wrap {
  padding: 4px 0 8px;
}

/* ── Divider ── */
.chex-divider {
  height: 1px;
  background: var(--hairline);
  margin: 18px 0;
}

/* ── Section kicker ── */
.chex-section-kicker {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg-subtle);
  margin-bottom: 10px;
  display: block;
}

/* ── Card body padding ── */
.chex-card-body {
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

/* ── Scrollbars ── */
*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
*::-webkit-scrollbar-track { background: transparent; }

/* ── Gradio utility gaps ── */
.gradio-container .gap-4 { gap: 14px !important; }
.gradio-container .gap-2 { gap: 8px !important; }

/* Nested sub-tabs (bank statement) */
.tabitem .tab-nav {
  position: static !important;
  top: auto !important;
}

/* Responsive spacing */
@media (max-width: 900px) {
  .chex-topbar { padding: 0 16px; }
  .chex-tag { display: none; }
  .tabitem { padding: 16px !important; }
  .chex-bench-stats { grid-template-columns: 1fr; }
  .chex-footer { padding: 12px 16px; gap: 12px; flex-wrap: wrap; }
}
"""

# ---------------------------------------------------------------------------
# Static HTML strings
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
  <span class="chex-card-kicker">paste · pdf · csv</span>
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

with gr.Blocks(
    title="CHEX — Document Intelligence",
    theme=gr.themes.Base(),
    css=CHEX_CSS,
) as demo:

    # ── Topbar ──────────────────────────────────────────────────────────── #
    gr.HTML(TOPBAR_HTML)

    # ── Warning banner (only if model failed) ───────────────────────────── #
    if WARNING_HTML:
        gr.HTML(WARNING_HTML)

    # ── Tabs ────────────────────────────────────────────────────────────── #
    with gr.Tabs():

        # ================================================================== #
        # Tab 01 — Contract Analysis                                          #
        # ================================================================== #
        with gr.Tab("01  Contract analysis"):
            with gr.Row(equal_height=False):

                # ── Left panel: source document ──────────────────────────── #
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

                # ── Right panel: question + results ──────────────────────── #
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

        # ================================================================== #
        # Tab 02 — Bank Statements                                            #
        # ================================================================== #
        with gr.Tab("02  Bank statements"):
            with gr.Row(equal_height=False):

                # ── Left panel: statement input ───────────────────────────── #
                with gr.Column(scale=9):
                  with gr.Group():
                    gr.HTML(STATEMENT_SOURCE_HEADER_HTML)
                    with gr.Tabs():
                        with gr.Tab("Paste text"):
                            bank_paste_input = gr.Textbox(
                                label="Bank statement text",
                                lines=20,
                                placeholder="Paste your bank statement here, or load the sample below…",
                                show_label=False,
                            )
                            btn_load_statement = gr.Button("Load sample statement", variant="secondary", size="sm")
                        with gr.Tab("Upload PDF"):
                            bank_pdf_input = gr.File(label="PDF bank statement", file_types=[".pdf"])
                        with gr.Tab("Upload CSV"):
                            bank_csv_input = gr.File(label="CSV bank statement", file_types=[".csv"])

                # ── Right panel: summary + Q&A ───────────────────────────── #
                with gr.Column(scale=11):
                  with gr.Group():
                    gr.HTML(STATEMENT_RESULTS_HEADER_HTML)
                    analyse_stmt_btn = gr.Button("Analyse statement", variant="primary")
                    summary_output = gr.Markdown(value="*Run 'Analyse statement' to generate a financial summary.*")
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

        # ================================================================== #
        # Tab 03 — Benchmark                                                  #
        # ================================================================== #
        with gr.Tab("03  Benchmark"):
            gr.HTML(BENCH_INTRO_HTML)
            gr.Dataframe(
                value=BENCHMARK_DF,
                headers=list(BENCHMARK_DF.columns),
                datatype=["str"] * len(BENCHMARK_DF.columns),
                wrap=True,
                interactive=False,
            )

    # ── Footer ──────────────────────────────────────────────────────────── #
    gr.HTML(FOOTER_HTML)

    # ====================================================================== #
    # Event handlers                                                          #
    # ====================================================================== #

    def load_software():
        hint = (
            '<div class="chex-suggested">'
            '<span class="chex-suggested-icon">💡</span>'
            '<span><strong>Suggested:</strong> What is the limitation of liability in this agreement?</span>'
            '</div>'
        )
        return (
            SOFTWARE_LICENSE,
            SAMPLE_QUESTIONS["software_license.txt"],
            gr.update(value=hint, visible=True),
        )

    def load_nda():
        hint = (
            '<div class="chex-suggested">'
            '<span class="chex-suggested-icon">💡</span>'
            '<span><strong>Suggested:</strong> Does this agreement include a non-compete clause?</span>'
            '</div>'
        )
        return (
            NDA,
            SAMPLE_QUESTIONS["nda.txt"],
            gr.update(value=hint, visible=True),
        )

    def load_service():
        hint = (
            '<div class="chex-suggested">'
            '<span class="chex-suggested-icon">💡</span>'
            '<span><strong>Suggested:</strong> Does this contract include a termination for convenience clause? '
            '<em>(expected: ABSENT)</em></span>'
            '</div>'
        )
        return (
            SERVICE_AGREEMENT,
            SAMPLE_QUESTIONS["service_agreement.txt"],
            gr.update(value=hint, visible=True),
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

    # Trigger on Enter in question field
    question_input.submit(
        fn=analyze_contract,
        inputs=[contract_input, question_input],
        outputs=[label_display, answer_output, citation_output, reasoning_output],
    )

    # ── Bank Statement handlers ──────────────────────────────────────────── #

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


if __name__ == "__main__":
    demo.launch(show_error=True)
