"""
Confound separator driver — Kaggle T4.  Paste into a Kaggle notebook cell.

WHAT THIS ANSWERS:
    Is the think-block format collapse driven by OPTIMIZER STEPS (weight
    movement) or by UNIQUE-TOKEN BREADTH? The full-canon dose curve moved both
    at once. This run trains the PILOT corpus (311K unique) to ~103 steps by
    re-reading it 11x — same step count as full canon, 1/11th the breadth — so
    overlaying its closure curve on the full-canon curve isolates breadth as the
    only difference. See configs/kaggle_t4_confound_pilot103.yaml for the 2x2.

READS:
    pilot stays HIGH while canon decays -> BREADTH drives it (rehearsal needed)
    pilot ALSO decays with steps        -> WEIGHT MOVEMENT contributes
                                           (low-LR / low-rank worth trying)

PREREQUISITE: the full-canon dose-curve JSON, committed at
    results/analysis/dose_curve_20260808_204827.json
It ships with the repo clone, so the final analysis cell can overlay the two
curves without re-running full canon.

KAGGLE SETUP (identical to kaggle_t4_dosecurve.py):
    1. Settings -> Accelerator -> GPU T4 x2
    2. Settings -> Internet -> ON
    3. Add-ons -> Secrets -> add HF_TOKEN (WRITE), and ATTACH it to this
       notebook. Without it the run ABORTS in CELL 4 (fail-closed) — by design,
       after 2026-08-08.
    4. Optional GITHUB_TOKEN for --push-results.

RUNTIME: ~4.5 h train + ~40 min eval, chained. COST: $0.
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
# CELL 3 — Clone the repo into /kaggle/working/si and set the environment.
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
subprocess.run(["git", "config", "user.name", "kaggle-confound-runner"])
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
print(f"repo at {REPO_DIR}")


# =============================================================================
# CELL 4 — Preflight. Verify invariants; FAIL CLOSED on a missing token.
# =============================================================================
import json

import yaml

CONFIG = Path("configs/kaggle_t4_confound_pilot103.yaml")
cfg = yaml.safe_load(CONFIG.read_text())

corpus = Path(cfg["training_corpus_path"])
assert corpus.exists(), f"missing PILOT corpus: {corpus}"
n_examples = sum(1 for line in corpus.open() if line.strip())
n_chars = sum(len(json.loads(line)["text"]) for line in corpus.open() if line.strip())
est_tokens = int(n_chars / 4.555)
seq = cfg["max_seq_length"]
eff_batch = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
est_blocks = est_tokens // seq
est_steps = max(1, est_blocks * cfg["num_epochs"] // eff_batch)

print("=" * 70)
print("  PREFLIGHT — confound separator (pilot @ ~103 steps)")
print("=" * 70)
print(f"  corpus (PILOT)    : {corpus}")
print(f"  examples          : {n_examples:,}")
print(f"  est. unique tokens: {est_tokens:,}   (~311K, vs full canon 3.36M)")
print(f"  epochs            : {cfg['num_epochs']}  (re-reads the pilot corpus)")
print(f"  EST. OPTIM STEPS  : {est_steps}   (target ~103, matching full canon)")
print(f"  save_total_limit  : {cfg.get('save_total_limit')}")
print(f"  hf_repo_id        : {cfg.get('hf_repo_id')}")

assert cfg.get("save_total_limit") is None, "keep every checkpoint for the curve."
assert cfg.get("hf_repo_id"), "hf_repo_id unset — persistence would be off."
# The full-canon curve to overlay against must be in the clone.
FULLCANON_JSON = Path("results/analysis/dose_curve_20260808_204827.json")
assert FULLCANON_JSON.exists(), (
    f"missing {FULLCANON_JSON} — the confound analysis needs the full-canon "
    "curve to overlay. It ships with the repo; check the clone.")
# FAIL CLOSED: no token -> do not spend 4.5 h on a run that cannot persist.
assert os.environ.get("HF_TOKEN"), (
    "HF_TOKEN is NOT set — this ~4.5h run would be LOST at session end "
    "(as on 2026-08-08). Add HF_TOKEN in Add-ons -> Secrets, ATTACH it to this "
    "notebook, then re-run CELL 3.")
print("\n  preflight OK — persistence configured, full-canon curve present")


# =============================================================================
# CELL 5 — CHAINED train -> eval -> confound analysis, in ONE cell.
# =============================================================================
RESULTS_REPO = "utsvsngh/sherlock-dosecurve-results"
PUSH_RESULTS = _gh is not None

print("=" * 70)
print("  TRAIN pilot @ ~103 steps (each checkpoint uploads to the Hub)")
print("=" * 70)
train = subprocess.run(
    [sys.executable, "-u", "scripts/training/train_lora.py", "--config", str(CONFIG)])
print(f"\n  train exit code: {train.returncode}")

print("\n" + "=" * 70)
print("  EVAL closure per checkpoint")
print("=" * 70)
eval_cmd = [
    sys.executable, "-u", "scripts/eval/dose_curve.py",
    "--checkpoint-dir", cfg["output_dir"],
    "--n-prompts", "8", "--seed", str(cfg["seed"]),
    "--hf-results-repo", RESULTS_REPO,
]
if PUSH_RESULTS:
    eval_cmd.append("--push-results")
ev = subprocess.run(eval_cmd)
print(f"\n  eval exit code: {ev.returncode}")

# Find the pilot curve JSON this eval just wrote (newest dose_curve_*.json that
# is NOT the committed full-canon one).
cands = sorted(Path("results/analysis").glob("dose_curve_*.json"),
               key=lambda p: p.stat().st_mtime, reverse=True)
pilot_json = next((p for p in cands if p.name != FULLCANON_JSON.name), None)

print("\n" + "=" * 70)
print("  CONFOUND ANALYSIS — steps vs unique-token breadth")
print("=" * 70)
if pilot_json:
    subprocess.run([
        sys.executable, "-u", "scripts/eval/confound_analysis.py",
        "--fullcanon", str(FULLCANON_JSON),
        "--pilot", str(pilot_json),
        "--out", "results/analysis/confound_pilot103_vs_fullcanon.json",
    ])
    # Persist the small analysis JSON off-machine too.
    if os.environ.get("HF_TOKEN"):
        sys.path.insert(0, "scripts/training")
        import hf_persist
        hf_persist.upload_path(
            "results/analysis/confound_pilot103_vs_fullcanon.json",
            RESULTS_REPO, os.environ["HF_TOKEN"],
            path_in_repo="confound_pilot103_vs_fullcanon.json",
            repo_type="dataset", private=True, label="confound analysis")
else:
    print("  no pilot curve JSON found — check the eval step above.")

print("\n" + "=" * 70)
if os.environ.get("HF_TOKEN"):
    print("  DONE. Durable copies:")
    print(f"    checkpoints : https://huggingface.co/{cfg['hf_repo_id']}")
    print(f"    results     : https://huggingface.co/datasets/{RESULTS_REPO}")
    print("  Verify the HF repos populated before closing the tab.")
else:
    print("  DONE — but NOTHING persisted (no token). Local only; will be lost.")
print("=" * 70)
