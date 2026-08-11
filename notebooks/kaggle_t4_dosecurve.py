"""
Dose-response curve driver — Kaggle T4.  Paste into a Kaggle notebook cell.

WHAT THIS ANSWERS:
    At what training dose does DeepSeek-R1-Distill's think-block closure begin to
    degrade? One full-canon run (save_total_limit=null keeps EVERY checkpoint),
    then closure scored at each checkpoint = the whole curve from a single run.

WHY THIS NOTEBOOK EXISTS IN GIT:
    The 2026-08-07 run was driven from cells typed live into a Kaggle session and
    was not reproducible from the repo. Worse, its 12 checkpoints, final adapter,
    and results JSON were all lost when the interactive session ended — the third
    time /kaggle/working being wiped at session end has destroyed a run. This
    notebook is the run path, committed so it reproduces from git, and it relies
    on two persistence mechanisms that are now part of the code, not manual
    afterthoughts:
      - train_lora.py uploads each checkpoint to the HF Hub the instant it is
        written (HFCheckpointUploader, gated on the config's hf_repo_id).
      - dose_curve.py uploads the results JSON + partial jsonl as rows are
        written, and --push-results git-commits the small JSON to the repo.

    CLAUDE.md, "Experiment durability": an experiment you cannot read mid-flight
    is not running, it is gambling. Everything below streams and persists.

KAGGLE SETUP BEFORE RUNNING:
    1. Notebook Settings -> Accelerator -> "GPU T4 x2"  (only GPU 0 is used)
    2. Notebook Settings -> Internet -> ON  (pull weights from HF; push to Hub)
    3. Add-ons -> Secrets -> add HF_TOKEN (WRITE scope). Required, or the
       checkpoints are NOT persisted and this failure repeats a fourth time.
    4. Optional: add GITHUB_TOKEN as a Secret if you want --push-results to push
       the small results JSON back to the repo (owner-requested git-push option).
    5. Session type: "Save & Run All (Commit)" so the run survives a browser
       close — but note even that wipes /kaggle/working at the end, which is
       precisely why persistence is off-machine.

RUNTIME: ~4.6 h train + ~40 min eval = well inside the 12 h cap, chained so
nothing durable has to cross a session boundary.
COST: $0.
"""

# =============================================================================
# CELL 1 — Environment check. Run FIRST and read the output.
# =============================================================================
import subprocess
import sys

import torch

print("=" * 70)
print("  ENVIRONMENT")
print("=" * 70)
if not torch.cuda.is_available():
    raise SystemExit("No GPU. Settings -> Accelerator -> GPU T4 x2, then restart.")

props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
# T4 = Turing = SM 7.5 -> fp16 only, NO native bfloat16. train_lora.native_bf16()
# uses compute capability, NOT is_bf16_supported() (which counts T4 emulation).
print(f"  GPU               : {props.name}")
print(f"  VRAM              : {vram_gb:.1f} GB")
print(f"  Compute capability: SM {props.major}.{props.minor}")
print(f"  CUDA devices      : {torch.cuda.device_count()}")
if vram_gb < 14:
    print(f"\n  WARNING: {vram_gb:.1f}GB is tight for 7B QLoRA at seq 2048.")


# =============================================================================
# CELL 2 — Dependencies.  RESTART THE KERNEL after this cell.
# =============================================================================
# Same pins as the validation notebook, plus huggingface_hub for the
# per-checkpoint uploads. unsloth is intentionally NOT installed — its install
# fights Kaggle's torch pin and the transformers+PEFT fallback is the path that
# runs on RunPod, so it is the path worth validating.
DEPS = [
    "transformers==4.57.1",
    "peft==0.18.0",
    "bitsandbytes==0.48.2",
    "accelerate==1.11.0",
    "datasets==4.8.5",
    "huggingface_hub>=0.24",   # HfApi.upload_folder for per-checkpoint persistence
    "pyyaml",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DEPS], check=True)
print("dependencies installed — RESTART THE KERNEL now, then skip to CELL 3")


# =============================================================================
# CELL 3 — Clone the repo into /kaggle/working/si and set the environment.
# =============================================================================
# The corpus (data/augmented/full_canon_train.jsonl) is tracked in git, so the
# clone brings the training data with it. No separate upload.
import os
from pathlib import Path

REPO_URL = "https://github.com/Utsav02/sherlock-investigates.git"
REPO_DIR = Path("/kaggle/working/si")

# The HF token persists every checkpoint off-machine. Set it in the ENV so
# train_lora.py (via hf_persist.find_hf_token) and dose_curve.py both see it.
# NEVER inline the token — Kaggle saves notebook output publicly.
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = _secrets.get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets — checkpoints WILL be persisted.")
except Exception as e:
    print(f"WARNING: no HF_TOKEN secret ({e}). Checkpoints will NOT be persisted "
          "off-machine and will be LOST at session end. Add the secret.")

# Optional GitHub token for --push-results (small results JSON back to the repo)
# and for cloning if the repo is ever made private. Held only in the clone URL
# and git remote config, never printed.
_gh = None
try:
    _gh = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass

clone_url = REPO_URL
if _gh:
    clone_url = f"https://{_gh}@github.com/Utsav02/sherlock-investigates.git"

if not REPO_DIR.exists():
    # Full clone (not --depth 1): --push-results needs real history to commit
    # against and push. A shallow clone can refuse to push on some setups.
    subprocess.run(["git", "clone", clone_url, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)

# Identify the committer for --push-results commits.
subprocess.run(["git", "config", "user.email", "kaggle-runner@sherlock.local"])
subprocess.run(["git", "config", "user.name", "kaggle-dosecurve-runner"])

# W&B off — it prompts for a login and blocks the run otherwise.
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
print(f"repo at {REPO_DIR}")


# =============================================================================
# CELL 4 — Preflight. Verify the invariants BEFORE burning ~5 hours.
# =============================================================================
import json

import yaml

CONFIG = Path("configs/kaggle_t4_dosecurve.yaml")
cfg = yaml.safe_load(CONFIG.read_text())

corpus = Path(cfg["training_corpus_path"])
assert corpus.exists(), f"missing training corpus: {corpus}"

n_examples = sum(1 for line in corpus.open() if line.strip())
n_chars = sum(len(json.loads(line)["text"]) for line in corpus.open() if line.strip())
# Measured chars/token on this tokenizer is 4.555 (STATUS.md), not 4 — the /4
# heuristic overestimates tokens ~12%. Use the measured ratio for the estimate.
est_tokens = int(n_chars / 4.555)
seq = cfg["max_seq_length"]
eff_batch = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
est_blocks = est_tokens // seq
est_steps = max(1, est_blocks * cfg["num_epochs"] // eff_batch)

print("=" * 70)
print("  PREFLIGHT")
print("=" * 70)
print(f"  corpus            : {corpus}")
print(f"  examples          : {n_examples:,}")
print(f"  est. tokens       : {est_tokens:,}  (measured 3,352,033 last run)")
print(f"  effective batch   : {eff_batch}")
print(f"  epochs            : {cfg['num_epochs']}")
print(f"  EST. OPTIM STEPS  : {est_steps}  (measured 103 last run)")
print(f"  save_steps        : {cfg['save_steps']}")
print(f"  save_total_limit  : {cfg.get('save_total_limit')}")
print(f"  hf_repo_id        : {cfg.get('hf_repo_id')}")
print(f"  base model        : {cfg['base_model']}")

# The whole point of this run: keep every checkpoint. If this is not null the
# curve is destroyed exactly as it was on 2026-08-06.
assert cfg.get("save_total_limit") is None, (
    "save_total_limit MUST be null for the dose curve — the checkpoints ARE the "
    "experiment. A finite limit discards the curve this run exists to produce.")
# save_steps must divide into the run or checkpoints are sparse.
assert cfg["save_steps"] <= est_steps, "save_steps > est steps: too few checkpoints."
# Persistence must be configured or the run repeats the 2026-08-07 loss.
assert cfg.get("hf_repo_id"), (
    "hf_repo_id is unset — checkpoints would not be persisted off-machine. "
    "Set it in the config so train_lora.py uploads each checkpoint as written.")
# HARD STOP, not a warning. On 2026-08-08 this was a warning, the run trained 5h
# with no token, and the session wipe destroyed all 22 checkpoints. Fail closed.
assert os.environ.get("HF_TOKEN"), (
    "HF_TOKEN is NOT in the environment — persistence would be disabled and this "
    "~5h run would be LOST at session end (as it was on 2026-08-08). Add HF_TOKEN "
    "in Add-ons -> Secrets, ATTACH it to this notebook, then re-run CELL 3. "
    "Deliberate local-only run only: set os.environ['ALLOW_UNPERSISTED']='1' and "
    "delete this assert.")
print("\n  preflight OK — persistence is configured (HF_TOKEN present)")


# =============================================================================
# CELL 5 — CHAINED train -> eval, in ONE cell.
# =============================================================================
# The 2026-08-06 post-mortem: training and evaluation were separate cells hours
# apart, so the checkpoints had to survive a session boundary they never needed
# to cross. 4.6 h train + ~40 min eval fits inside one 12 h session. Chain them.
#
# Semantics: run train, then run eval EVEN IF train exited non-zero (a partial
# set of checkpoints is still worth scoring, and a failed run still needs
# diagnosing). This mirrors the house rule "use ; not && so the consumer still
# runs when the producer exits non-zero".
#
# Streaming: train_lora.py and dose_curve.py both print unbuffered (flush=True
# on the hot paths); run them with -u so nothing block-buffers. Do NOT pipe
# through tail/head/sort — that silences a live job until it finishes.

# Persist results to a HF dataset repo as they are written; optionally git-push
# the small JSON back to the repo. Drop --push-results if you did not set
# GITHUB_TOKEN in CELL 3.
RESULTS_REPO = "utsvsngh/sherlock-dosecurve-results"
PUSH_RESULTS = _gh is not None

print("=" * 70)
print("  TRAIN  (each checkpoint uploads to the Hub as it is written)")
print("=" * 70)
train = subprocess.run(
    [sys.executable, "-u", "scripts/training/train_lora.py", "--config", str(CONFIG)],
)
print(f"\n  train exit code: {train.returncode} "
      f"({'ok' if train.returncode == 0 else 'NON-ZERO — scoring partial curve'})")

print("\n" + "=" * 70)
print("  EVAL  (closure per checkpoint; results persisted as rows are written)")
print("=" * 70)
eval_cmd = [
    sys.executable, "-u", "scripts/eval/dose_curve.py",
    "--checkpoint-dir", cfg["output_dir"],
    "--n-prompts", "8",
    "--seed", str(cfg["seed"]),
    "--hf-results-repo", RESULTS_REPO,
]
if PUSH_RESULTS:
    eval_cmd.append("--push-results")
ev = subprocess.run(eval_cmd)
print(f"\n  eval exit code: {ev.returncode}")

print("\n" + "=" * 70)
if os.environ.get("HF_TOKEN"):
    print("  DONE. Durable copies (survive session end):")
    print(f"    checkpoints + final adapter : https://huggingface.co/{cfg['hf_repo_id']}")
    print(f"    results JSON + partial jsonl: https://huggingface.co/datasets/{RESULTS_REPO}")
    if PUSH_RESULTS:
        print("    results JSON also pushed to the git repo (results/analysis/).")
    print("  Verify the HF repos above actually populated before closing the tab.")
else:
    print("  DONE — but HF_TOKEN was NOT set, so NOTHING was persisted off-machine.")
    print("  Everything is on /kaggle/working ONLY and will be LOST at session end.")
    print("  Set the token now and re-upload outputs/ + results/analysis/ before")
    print("  closing (see the recovery snippet in the PR discussion).")
print("=" * 70)
