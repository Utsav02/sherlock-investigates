"""
Thinking-shift check — Kaggle T4.  Paste into a Kaggle notebook cell.

WHAT THIS ANSWERS: does the fine-tune shift the model's REASONING, not just its
prose? Runs the committed probe set (10 deduction-inviting + 10 reasoning + 10
neutral) through the BASE model and the low-rank step-50 adapter and puts their
think blocks side by side. NEUTRAL is the control — a real shift should move the
deduction prompts more than the small-talk. Perplexity could not answer this
(most of its drop was generic prose recovery); reading the think blocks can.

NO TRAINING — just load + generate. ~15–25 min. COST: $0.

SETUP: GPU T4 x2, Internet ON, HF_TOKEN secret ATTACHED (needed to pull the
private step-50 adapter). Optional GITHUB_TOKEN for --push-results.
"""

# =============================================================================
# CELL 1 — Environment check.
# =============================================================================
import subprocess
import sys

import torch

if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then restart.")
print("GPU:", torch.cuda.get_device_properties(0).name)


# =============================================================================
# CELL 2 — Dependencies.  RESTART THE KERNEL after this cell.
# =============================================================================
DEPS = [
    "transformers==4.57.1", "peft==0.18.0", "bitsandbytes==0.48.2",
    "accelerate==1.11.0", "datasets==4.8.5", "huggingface_hub>=0.24", "pyyaml",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DEPS])  # no -q trap: check output
print("deps done — RESTART THE KERNEL, then go to CELL 3")


# =============================================================================
# CELL 3 — Clone repo, CHECK OUT THE FEATURE BRANCH, load token.
# =============================================================================
# The thinking_shift script lives on the feature branch (not yet on main), so we
# check it out explicitly — this is why the low-rank run first hit a
# FileNotFoundError on a plain clone-of-main.
import os
from pathlib import Path

BRANCH = "claude/mystifying-hofstadter-bc999b"
REPO_DIR = Path("/kaggle/working/si")

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded (needed to pull the private step-50 adapter).")
except Exception as e:
    raise SystemExit(f"No HF_TOKEN secret ({e}) — required to pull the adapter. "
                     "Add it in Add-ons -> Secrets and ATTACH it.")

_gh = None
try:
    _gh = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
url = f"https://{_gh}@github.com/Utsav02/sherlock-investigates.git" if _gh \
    else "https://github.com/Utsav02/sherlock-investigates.git"

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", url, str(REPO_DIR)], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "checkout", BRANCH], check=True)
os.chdir(REPO_DIR)
assert Path("scripts/eval/thinking_shift.py").exists(), "on the wrong branch"
print("repo ready on", BRANCH)


# =============================================================================
# CELL 4 — Run the comparison.
# =============================================================================
# Greedy decoding so base-vs-fine-tuned differences are from the weights, not
# sampling. Pulls the step-50 adapter (closure 8/8, the sweet spot) from HF.
# Add --limit 6 for a fast first look; drop it for the full 30-prompt set.
ADAPTER = "utsvsngh/sherlock-r1distill-7b-lowrank-r8"
subprocess.run([
    sys.executable, "-u", "scripts/eval/thinking_shift.py",
    "--adapter", ADAPTER, "--subfolder", "checkpoint-50",
    "--hf-results-repo", "utsvsngh/sherlock-dosecurve-results",
] + (["--push-results"] if _gh else []))

print("\nDONE. The transcript markdown is the thing to read:")
print("  results/analysis/thinking_shift_*_transcript.md")
print("Paste a few deduction-prompt pairs (base vs fine-tuned think blocks) and")
print("the REGISTER PROFILE table back into the chat.")
