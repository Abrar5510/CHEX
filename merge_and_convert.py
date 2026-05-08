"""
CHEX — Merge LoRA + Convert to GGUF Q4_K_M

Run this locally:
    pip install torch transformers peft huggingface_hub
    python merge_and_convert.py

Requirements:
    - ~20 GB free disk space
    - llama.cpp cloned alongside this script OR installed via brew/pip
"""

import os
import subprocess
import sys
from pathlib import Path

ADAPTER_REPO = "Abrar55/contractual-hallucination-eliminator"
BASE_MODEL   = "Qwen/Qwen3-9B"
MERGED_DIR   = Path("./merged-chex")
GGUF_FILE    = Path("./chex-q4_k_m.gguf")
LLAMA_CPP_DIR = Path("./llama.cpp")

# ---------------------------------------------------------------------------
# Step 1 — merge LoRA into base model
# ---------------------------------------------------------------------------

def merge():
    print("\n=== Step 1: Merging LoRA adapter into base model ===")
    print(f"  Base : {BASE_MODEL}")
    print(f"  LoRA : {ADAPTER_REPO}")
    print(f"  Out  : {MERGED_DIR}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("  Loading base model (fp16)... this takes a few minutes")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("  Applying LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_REPO)

    print("  Merging weights...")
    merged = model.merge_and_unload()

    print(f"  Saving to {MERGED_DIR} ...")
    MERGED_DIR.mkdir(exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    AutoTokenizer.from_pretrained(ADAPTER_REPO, trust_remote_code=True).save_pretrained(str(MERGED_DIR))
    print("  Merge complete.")


# ---------------------------------------------------------------------------
# Step 2 — clone llama.cpp if needed and convert to GGUF
# ---------------------------------------------------------------------------

def convert():
    print("\n=== Step 2: Converting to GGUF Q4_K_M ===")

    if not LLAMA_CPP_DIR.exists():
        print("  Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP_DIR)],
            check=True,
        )

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # older llama.cpp used convert.py
        convert_script = LLAMA_CPP_DIR / "convert.py"

    print("  Installing llama.cpp Python deps...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(LLAMA_CPP_DIR / "requirements.txt")],
        check=True,
    )

    print(f"  Converting {MERGED_DIR} → {GGUF_FILE} (Q4_K_M)...")
    subprocess.run(
        [
            sys.executable, str(convert_script),
            str(MERGED_DIR),
            "--outfile", str(GGUF_FILE),
            "--outtype", "q4_k_m",
        ],
        check=True,
    )
    print(f"  GGUF written: {GGUF_FILE} ({GGUF_FILE.stat().st_size / 1e9:.1f} GB)")


# ---------------------------------------------------------------------------
# Step 3 — upload GGUF to HF
# ---------------------------------------------------------------------------

def upload():
    print("\n=== Step 3: Uploading GGUF to HuggingFace ===")
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("  Paste your HF token (write access): ").strip()

    api = HfApi(token=token)
    repo_id = "Abrar55/chex-gguf"

    print(f"  Creating repo {repo_id} (if not exists)...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"  Uploading {GGUF_FILE} ... (this may take a while)")
    api.upload_file(
        path_or_fileobj=str(GGUF_FILE),
        path_in_repo=GGUF_FILE.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  Done! Model available at: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    merge()
    convert()
    upload()
    print("\n=== All done! ===")
    print("Next: update your Space to use llama-cpp-python with Abrar55/chex-gguf")
