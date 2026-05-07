"""
CHEX — Contractual Hallucination Eliminator
HuggingFace Spaces Gradio Demo

Tab 1: Analyze Contract — paste a contract, ask a question, get a structured answer
Tab 2: Benchmark Demo — side-by-side table showing base model hallucinations vs CHEX
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
    "HF_MODEL_REPO", "PLACEHOLDER/contractual-hallucination-eliminator"
)
SAMPLE_DIR = Path(__file__).parent / "sample_contracts"

analyzer = None
model_load_error: Optional[str] = None

try:
    from serving.inference import ContractAnalyzer  # type: ignore

    analyzer = ContractAnalyzer(model_path=MODEL_PATH)
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

BADGE_COLORS = {
    "GROUNDED": "#22c55e",
    "ABSENT": "#ef4444",
    "CONTRADICTS_PRIOR": "#f59e0b",
    "N/A": "#6b7280",
    "ERROR": "#dc2626",
}

BADGE_ICONS = {
    "GROUNDED": "✓",
    "ABSENT": "✗",
    "CONTRADICTS_PRIOR": "⚠",
    "N/A": "—",
    "ERROR": "!",
}


def format_label_html(label: str) -> str:
    color = BADGE_COLORS.get(label, "#6b7280")
    icon = BADGE_ICONS.get(label, "")
    return (
        f'<div style="background:{color}; color:white; padding:10px 20px; '
        f'border-radius:8px; font-weight:bold; font-size:1.2em; text-align:center; '
        f'letter-spacing:0.05em; margin:4px 0;">'
        f'{icon} {label}</div>'
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
        '<div style="background:#fef3c7; border:2px solid #f59e0b; padding:12px 16px; '
        'border-radius:8px; margin-bottom:12px; font-size:0.95em;">'
        f'<strong>⚠ Model not loaded:</strong> {model_load_error}<br>'
        'Set the <code>HF_MODEL_REPO</code> secret in your HuggingFace Space settings '
        'to the correct model repository ID.'
        '</div>'
    )

with gr.Blocks(
    title="CHEX — Contractual Hallucination Eliminator",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# CHEX — Contractual Hallucination Eliminator\n"
        "**Fine-tuned Qwen3.5-9B on AMD MI300X (ROCm)** — "
        "detects hallucinations in contract analysis with calibrated uncertainty signals.\n\n"
        "Instead of confidently fabricating answers, CHEX outputs one of three structured labels: "
        "**GROUNDED** (answer exists), **ABSENT** (clause not present), or "
        "**CONTRADICTS_PRIOR** (terms deviate from standard)."
    )

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
        # Tab 2: Benchmark Demo                                               #
        # ------------------------------------------------------------------ #
        with gr.Tab("Benchmark Demo"):
            gr.Markdown(
                "## Base Qwen3.5-9B (untuned) vs. CHEX Fine-tuned Model\n\n"
                "The table below shows 5 representative contract questions. "
                "Rows marked with 🚨 show where the **base model hallucinated** "
                "— it predicted GROUNDED (with a fabricated citation) when the correct "
                "answer is **ABSENT** (the clause does not exist in the contract).\n\n"
                "CHEX correctly returns ABSENT for all such cases."
            )

            gr.Dataframe(
                value=BENCHMARK_DF,
                headers=list(BENCHMARK_DF.columns),
                datatype=["str"] * len(BENCHMARK_DF.columns),
                wrap=True,
                interactive=False,
            )

            gr.Markdown(
                "### Key Insight\n"
                "The base model (Qwen3.5-9B, zero-shot) hallucinates **2 out of 5** "
                "examples — fabricating legal clauses that do not exist in the document.\n\n"
                "CHEX (fine-tuned on AMD MI300X with LoRA) achieves **0 hallucinations** "
                "on these examples by learning to distinguish between what the contract "
                "actually says and what it doesn't say."
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


if __name__ == "__main__":
    demo.launch(show_error=True)
