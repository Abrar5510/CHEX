"""
Prompt template for CHEX.

Provides a system prompt with 3 hardcoded few-shot examples (one per label class)
and utility functions for formatting training examples and inference prompts.

Qwen3.5-9B uses the ChatML format (<|im_start|>/<|im_end|>).  At inference time
the tokenizer's apply_chat_template() is used so the model sees the exact same
token sequence it was trained on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import Label, LabeledQAExample, ModelOutput


# ---------------------------------------------------------------------------
# System prompt — establishes task framing and provides 3 few-shot examples
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_inference_prompt(contract_text: str, question: str) -> str:
    """
    Build the full inference prompt string (system + user content as plain text).
    Use this for quick testing; for proper Qwen ChatML formatting at inference
    time, use build_chat_messages() + tokenizer.apply_chat_template().
    """
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"[CONTRACT]\n{contract_text}\n[/CONTRACT]\n\n"
        f"Question: {question}"
    )


def build_chat_messages(contract_text: str, question: str) -> list[dict[str, str]]:
    """
    Build ChatML-format messages for Qwen3.5-9B.
    Pass to: tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"[CONTRACT]\n{contract_text}\n[/CONTRACT]\n\nQuestion: {question}",
        },
    ]


def format_training_example(example: LabeledQAExample) -> dict[str, str]:
    """
    Returns {"prompt": str, "completion": str} suitable for SFTTrainer.

    The prompt is the ChatML-formatted input (system + user turn with generation
    prompt appended).  The completion is the raw JSON output — no additional tokens.

    Note: the tokenizer must be passed when building the actual training text so
    that apply_chat_template() can insert the correct special tokens.  This function
    returns the logical parts; train.py combines them via the tokenizer.
    """
    output = ModelOutput(
        question=example.question,
        label=example.label,
        answer=example.answer,
        citation=example.citation,
        reasoning=example.reasoning,
    )
    # Return logical parts; concatenation with tokenizer happens in train.py
    return {
        "contract_text": example.contract_text,
        "question": example.question,
        "completion": output.model_dump_json(),
    }


def build_full_training_text(
    example: LabeledQAExample,
    tokenizer,  # HuggingFace tokenizer with apply_chat_template support
    eos_token: str = "<|im_end|>",
) -> str:
    """
    Build the complete training string using the tokenizer's chat template.
    Returns: <ChatML prompt> + <JSON completion> + <EOS>
    """
    messages = build_chat_messages(example.contract_text, example.question)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    output = ModelOutput(
        question=example.question,
        label=example.label,
        answer=example.answer,
        citation=example.citation,
        reasoning=example.reasoning,
    )
    return prompt + output.model_dump_json() + eos_token
