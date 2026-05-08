"""
ContractAnalyzer — inference wrapper for the CHEX - Document Intelligence fine-tuned model.

Uses HuggingFace pipeline for ROCm compatibility (not vLLM).
Validates output JSON against ModelOutput schema; retries once with a stricter
prompt on parse failure; falls back gracefully to ABSENT if both attempts fail.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import Label, LabeledQAExample, ModelOutput
from training.prompt_template import build_chat_messages, format_inference_prompt


# ---------------------------------------------------------------------------
# ROCm detection (same logic as train.py)
# ---------------------------------------------------------------------------

def _is_rocm() -> bool:
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0)
    return "AMD" in name or "gfx" in name.lower()


# ---------------------------------------------------------------------------
# ContractAnalyzer
# ---------------------------------------------------------------------------

class ContractAnalyzer:
    """
    Load a CHEX fine-tuned (or base) Qwen model and answer contract questions.

    Args:
        model_path:  Local checkpoint / HF repo for the merged model *or* a LoRA
                     adapter repo (when base_model is also provided).
        base_model:  When set, model_path is treated as a LoRA adapter and loaded
                     on top of this base model via PEFT. The adapter's weight shapes
                     must match the base — a 9B-trained adapter cannot be loaded on
                     a 0.8B base without re-training (shape mismatch will raise).
        device:      "auto" lets HF choose; pass "cuda" or "cpu" to override.
    """

    MAX_CONTRACT_TOKENS = 8192
    STRICT_SUFFIX = (
        "\n\nIMPORTANT: You must output ONLY a valid JSON object. "
        "Do not include any text before or after the JSON."
    )

    def __init__(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        self._model_path = model_path
        self._base_model = base_model
        self._device = device
        self._pipe, self._tokenizer = self._load_pipeline(model_path, base_model, device)
        label = f"base={base_model} + adapter={model_path}" if base_model else model_path
        print(f"ContractAnalyzer ready. {label}")

    def _load_pipeline(self, model_path: str, base_model: Optional[str], device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

        rocm = _is_rocm()
        bnb_available = importlib.util.find_spec("bitsandbytes") is not None

        # When a base model is given, load it first then apply the LoRA adapter.
        # Otherwise treat model_path as a fully merged checkpoint.
        weights_source = base_model if base_model else model_path

        print(f"Loading tokenizer from: {weights_source}")
        tokenizer = AutoTokenizer.from_pretrained(weights_source, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading base model from: {weights_source}")
        if bnb_available:
            from transformers import BitsAndBytesConfig  # type: ignore

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                weights_source,
                quantization_config=bnb_config,
                device_map="auto" if device == "auto" else device,
                trust_remote_code=True,
            )
            print("  Loaded with 4-bit NF4 quantization")
        else:
            dtype = torch.float16 if rocm else torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(
                weights_source,
                torch_dtype=dtype,
                device_map="auto" if device == "auto" else device,
                trust_remote_code=True,
            )
            print(f"  Loaded in {'fp16' if rocm else 'bf16'} (no quantization)")

        if base_model:
            # NOTE: adapter must have been trained on a model with the same
            # architecture as base_model. Loading a 9B-trained adapter onto a
            # 0.8B base will raise a shape-mismatch error here.
            try:
                from peft import PeftModel  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "peft is required to load a LoRA adapter. "
                    "Install it with: pip install peft"
                ) from exc
            print(f"Applying LoRA adapter from: {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("  Adapter applied.")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=False,
            return_full_text=False,  # return only newly generated tokens
            pad_token_id=tokenizer.eos_token_id,
        )
        return pipe, tokenizer

    def _truncate_contract(self, contract_text: str) -> str:
        """Truncate contract to MAX_CONTRACT_TOKENS tokens, printing a warning."""
        tokens = self._tokenizer.encode(contract_text, add_special_tokens=False)
        if len(tokens) > self.MAX_CONTRACT_TOKENS:
            print(
                f"WARNING: Contract truncated from {len(tokens)} to "
                f"{self.MAX_CONTRACT_TOKENS} tokens."
            )
            tokens = tokens[: self.MAX_CONTRACT_TOKENS]
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        return contract_text

    def _parse_output(self, raw_text: str) -> ModelOutput:
        """
        Extract and validate the JSON object from the model's raw text output.
        Raises ValueError if no valid JSON is found.
        """
        # Try to find the outermost JSON object
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw_text, re.DOTALL)
        if not match:
            # Fallback: try the whole text
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output: {raw_text[:300]!r}")

        json_str = match.group()
        return ModelOutput.model_validate_json(json_str)

    def _build_prompt(self, contract_text: str, question: str, strict: bool = False) -> str:
        """Build the full prompt string using the tokenizer's chat template."""
        messages = build_chat_messages(contract_text, question)
        if strict:
            messages[-1]["content"] += self.STRICT_SUFFIX
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback for tokenizers that don't support chat templates
            return format_inference_prompt(contract_text, question) + (
                self.STRICT_SUFFIX if strict else ""
            )

    def analyze(self, contract_text: str, question: str) -> ModelOutput:
        """
        Analyze a single contract/question pair.

        Returns ModelOutput with the classified label, answer, citation, and reasoning.
        Retries once with a stricter prompt on JSON parse failure.
        Falls back to ABSENT if both attempts fail.
        """
        contract_text = self._truncate_contract(contract_text)

        for attempt in range(2):
            strict = attempt == 1
            prompt = self._build_prompt(contract_text, question, strict=strict)
            try:
                result = self._pipe(prompt)
                raw = result[0]["generated_text"]
                return self._parse_output(raw)
            except Exception as e:
                if attempt == 0:
                    print(f"  Parse attempt 1 failed ({e}). Retrying with stricter prompt...")
                else:
                    print(f"  Parse attempt 2 failed ({e}). Returning safe fallback.")

        return ModelOutput(
            question=question,
            label=Label.ABSENT,
            answer=None,
            citation=None,
            reasoning="Model output could not be parsed as valid JSON after two attempts.",
        )

    def analyze_batch(self, examples: list[dict]) -> list[ModelOutput]:
        """
        Analyze a batch of contract/question pairs.

        Each dict must have keys: "contract_text" (str), "question" (str).
        """
        results: list[ModelOutput] = []
        for i, ex in enumerate(examples):
            print(f"  Batch [{i+1}/{len(examples)}]: {ex.get('question', '')[:60]}...")
            output = self.analyze(ex["contract_text"], ex["question"])
            results.append(output)
        return results


# ---------------------------------------------------------------------------
# CLI (for quick ad-hoc testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CHEX Document Intelligence inference on a document file.")
    parser.add_argument("--model_path", required=True, help="Merged model path/repo, or LoRA adapter repo when --base_model is set")
    parser.add_argument("--base_model", default=None, help="Base model to load before applying the LoRA adapter")
    parser.add_argument("--contract_file", type=Path, required=True, help="Contract .txt file")
    parser.add_argument("--question", required=True, help="Question about the contract")
    args = parser.parse_args()

    analyzer = ContractAnalyzer(model_path=args.model_path, base_model=args.base_model)
    contract = Path(args.contract_file).read_text(encoding="utf-8")
    result = analyzer.analyze(contract, args.question)

    print("\n--- Result ---")
    print(result.model_dump_json(indent=2))
