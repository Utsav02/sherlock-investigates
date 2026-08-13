"""
Low-rank MITIGATION driver — Kaggle T4.  Paste into a Kaggle notebook cell.

WHAT THIS ANSWERS (and why it decides whether rehearsal is needed):
    The confound separator (2026-08-12) showed the <think> format collapse is
    driven by optimizer STEPS / weight movement, not corpus breadth. The lever is
    therefore a CONSTRAINED SUBSPACE — lower LoRA rank. This run trains rank 8
    (vs 32) on the full canon and measures BOTH:
      (a) closure per checkpoint  -> dose_curve.py     (does the format survive?)
      (b) held-out Speckled Band PPL per checkpoint -> effect_curve.py (did it learn?)
    mitigation_analysis.py overlays them:
      window exists (closure intact AND PPL drop >= 5%) -> RESCUED, rehearsal NOT needed
      effect only after closure collapses               -> COUPLED, rehearsal needed
      effect never appears                              -> TOO WEAK, rehearsal needed

    A closure curve ALONE is insufficient: low rank could "preserve closure" by
    learning too little to matter. The effect curve catches that.

KAGGLE SETUP (identical to the other drivers):
    1. Settings -> Accelerator -> GPU T4 x2   2. Internet -> ON
    3. Add-ons -> Secrets -> HF_TOKEN (WRITE), ATTACHED. Without it CELL 4 ABORTS.
    4. Optional GITHUB_TOKEN for --push-results.

RUNTIME: ~4.5 h train + ~40 min closure + ~20 min effect. COST: $0.
"""

# =============================================================================
# CELL 1 — Environment check.
# =============================================================================
import subprocess
import sys

import torch

print("=" * 70)
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then restart.")
props = torch.cuda.get_device_properties(0)
print(f"  GPU: {props.name}  VRAM: {props.total_memory/1024**3:.1f} GB  "
      f"SM {props.major}.{props.minor}  devices {torch.cuda.device_count()}")


# =============================================================================
# CELL 2 — Dependencies.  RESTART THE KERNEL after this cell.
# =============================================================================
DEPS = [
    "transformers==4.57.1", "peft==0.18.0", "bitsandbytes==0.48.2",
    "accelerate==1.11.0", "datasets==4.8.5", "huggingface_hub>=0.24", "pyyaml",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DEPS], check=True)
print("dependencies installed — RESTART THE KERNEL now, then skip to CELL 3")


# =============================================================================
# CELL 3 — Clone repo + secrets.
# =============================================================================
import os
from pathlib import Path

REPO_URL = "https://github.com/Utsav02/sherlock-investigates.git"
REPO_DIR = Path("/kaggle/working/si")

try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = _secrets.get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets — checkpoints WILL be persisted.")
except Exception as e:
    print(f"WARNING: no HF_TOKEN secret ({e}). CELL 4 will ABORT (fail-closed).")

_gh = None
try:
    _gh = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
clone_url = f"https://{_gh}@github.com/Utsav02/sherlock-investigates.git" if _gh else REPO_URL

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", clone_url, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
subprocess.run(["git", "config", "user.email", "kaggle-runner@sherlock.local"])
subprocess.run(["git", "config", "user.name", "kaggle-lowrank-runner"])
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
print(f"repo at {REPO_DIR}")


# =============================================================================
# CELL 4 — Preflight. FAIL CLOSED on a missing token.
# =============================================================================
import json

import yaml

CONFIG = Path("configs/kaggle_t4_lowrank_r8.yaml")
cfg = yaml.safe_load(CONFIG.read_text())

corpus = Path(cfg["training_corpus_path"])
heldout = Path(cfg["heldout_corpus_path"])
assert corpus.exists(), f"missing corpus: {corpus}"
assert heldout.exists(), f"missing held-out corpus: {heldout}"

print("=" * 70)
print("  PREFLIGHT — low-rank mitigation (rank 8, full canon)")
print("=" * 70)
print(f"  corpus (full canon): {corpus}")
print(f"  held-out (effect)  : {heldout}")
print(f"  lora_rank / alpha  : {cfg['lora_rank']} / {cfg['lora_alpha']}  "
      f"(dose curve was 32 / 64)")
print(f"  save_total_limit   : {cfg.get('save_total_limit')}")
print(f"  hf_repo_id         : {cfg.get('hf_repo_id')}")

assert cfg["lora_rank"] == 8, "this run is the rank-8 arm."
assert cfg.get("save_total_limit") is None, "keep every checkpoint for both curves."
assert cfg.get("hf_repo_id"), "hf_repo_id unset — persistence would be off."
assert os.environ.get("HF_TOKEN"), (
    "HF_TOKEN is NOT set — this ~4.5h run would be LOST at session end. Add "
    "HF_TOKEN in Add-ons -> Secrets, ATTACH it, then re-run CELL 3.")
print("\n  preflight OK — persistence configured")


# =============================================================================
# CELL 5 — CHAINED train -> closure -> effect -> mitigation verdict.
# =============================================================================
RESULTS_REPO = "utsvsngh/sherlock-dosecurve-results"
PUSH = ["--push-results"] if _gh is not None else []
CKPT = cfg["output_dir"]

print("=" * 70); print("  TRAIN rank-8 (each checkpoint uploads to the Hub)"); print("=" * 70)
train = subprocess.run(
    [sys.executable, "-u", "scripts/training/train_lora.py", "--config", str(CONFIG)])
print(f"\n  train exit code: {train.returncode}")

print("\n" + "=" * 70); print("  (a) CLOSURE per checkpoint"); print("=" * 70)
subprocess.run([sys.executable, "-u", "scripts/eval/dose_curve.py",
                "--checkpoint-dir", CKPT, "--n-prompts", "8",
                "--seed", str(cfg["seed"]), "--hf-results-repo", RESULTS_REPO, *PUSH])

print("\n" + "=" * 70); print("  (b) EFFECT per checkpoint (held-out PPL)"); print("=" * 70)
subprocess.run([sys.executable, "-u", "scripts/eval/effect_curve.py",
                "--checkpoint-dir", CKPT, "--with-wikitext",
                "--hf-results-repo", RESULTS_REPO, *PUSH])

# newest closure + effect JSONs this session wrote
def _newest(glob):
    c = sorted(Path("results/analysis").glob(glob), key=lambda p: p.stat().st_mtime,
               reverse=True)
    return c[0] if c else None
closure_json = _newest("dose_curve_*.json")
effect_json = _newest("effect_curve_*.json")

print("\n" + "=" * 70); print("  MITIGATION VERDICT — closure x effect"); print("=" * 70)
if closure_json and effect_json:
    subprocess.run([sys.executable, "-u", "scripts/eval/mitigation_analysis.py",
                    "--closure", str(closure_json), "--effect", str(effect_json),
                    "--out", "results/analysis/mitigation_lowrank_r8.json"])
    if os.environ.get("HF_TOKEN"):
        sys.path.insert(0, "scripts/training")
        import hf_persist
        hf_persist.upload_path("results/analysis/mitigation_lowrank_r8.json",
                               RESULTS_REPO, os.environ["HF_TOKEN"],
                               path_in_repo="mitigation_lowrank_r8.json",
                               repo_type="dataset", private=True, label="mitigation verdict")
else:
    print("  missing a curve JSON — check the eval steps above.")

print("\n" + "=" * 70)
if os.environ.get("HF_TOKEN"):
    print("  DONE. Durable copies:")
    print(f"    checkpoints : https://huggingface.co/{cfg['hf_repo_id']}")
    print(f"    results     : https://huggingface.co/datasets/{RESULTS_REPO}")
    print("  Paste the CLOSURE table, the EFFECT table, and the MITIGATION VERDICT")
    print("  back into the chat — that triple decides rehearsal-or-writeup.")
else:
    print("  DONE — but NOTHING persisted (no token). Local only; will be lost.")
print("=" * 70)
