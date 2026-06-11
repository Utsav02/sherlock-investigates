# RunPod training runbook

How to take the committed pilot corpus to two trained LoRA adapters on a
rented GPU. Assembled from `EXPERIMENT_DESIGN.md` ("Compute and cost"
section), the `requirements.txt` header comment, `scripts/training/train_lora.py`'s
docstring, and the two pilot configs.

> **Status: pre-flight plan.** Training has not run yet (see CLAUDE.md
> "Current state"). Commands below are taken from the training script's own
> usage docs and RunPod's standard workflow — none have been executed on a
> pod yet. Update this file with corrections after the first real run.

## Pod spec

| Item | Choice | Why |
|---|---|---|
| GPU | RTX 4090 (24 GB), Community Cloud | ~$0.34/hr as of May 2026; a third of Secure-tier price. Preemption risk is acceptable with 30-min checkpointing. |
| Template | RunPod official PyTorch template | Ships CUDA 12.4 + PyTorch 2.4 with transformers, peft, bitsandbytes, accelerate pre-installed. |
| Storage | 50 GB Network Volume (~$3.50/mo) | Persists datasets and adapter checkpoints across pod sessions; Community pods can be reclaimed at any time. |
| Time budget | ~3 hr per variant; 2 variants × 1 seed ≈ 6 GPU-hours (~$2) | Pilot training budget is $5–8 total including a contingency margin. |

## Mounts and layout on the pod

Network Volumes mount at `/workspace` on RunPod. Keep everything the run
needs (and everything it produces) on the volume, not the pod's ephemeral
container disk:

```
/workspace/sherlock-investigates/      # git clone of this repo
  data/augmented/train.jsonl           # training corpus (committed in git)
  data/processed/heldout/speckled_band.txt
  configs/pilot_qwen.yaml
  configs/pilot_mistral.yaml
  outputs/                             # adapters + checkpoints land here
    pilot_qwen_seed42/
    pilot_mistral_seed42/
```

The whole corpus is tracked in git (~8 MB), so `git clone` is the entire
data transfer — no separate upload step.

## Deferred pip installs

`requirements.txt` deliberately contains only the data-prep deps. The
training-phase deps (per its header comment) are installed on the pod, where
most come pre-installed with the PyTorch template:

```bash
# Pre-installed on the RunPod PyTorch template — verify, don't reinstall:
#   torch, transformers, peft, bitsandbytes, accelerate
pip install datasets pyyaml tqdm   # needed by train_lora.py, not on template
pip install unsloth                # optional but strongly recommended (~30% less memory, faster)
pip install wandb                  # optional — only if logging to W&B
```

Pin the versions that actually resolve on the pod and record them here after
the first run, so the second run is reproducible.

## Environment variables

| Var | Required | Purpose |
|---|---|---|
| `WANDB_API_KEY` | No | Only if `wandb` is installed and you want hosted run tracking. The script runs fine without it. |
| `HF_HOME` | No | Set to `/workspace/hf-cache` so model downloads survive pod preemption. |

## Run

```bash
cd /workspace/sherlock-investigates
python3 scripts/training/train_lora.py --config configs/pilot_qwen.yaml
python3 scripts/training/train_lora.py --config configs/pilot_mistral.yaml

# Resume after preemption:
python3 scripts/training/train_lora.py --config configs/pilot_qwen.yaml \
    --resume-from outputs/pilot_qwen_seed42/checkpoint-200
```

(`make train-qwen` / `make train-mistral` wrap the same commands but assume
the repo venv at `venv/` — on a pod it's simpler to use the system python.)

## Pre-launch checklist (gotchas from the design doc)

- `logging_steps` and `save_steps` are `null` in both configs — **the script
  exits with an error until they're set.** Recommended: `save_steps: 50`
  (≈ every 30 min on a 4090) for preemption resilience.
- Qwen: config already points at `unsloth/Qwen2.5-7B-Instruct`, the patched
  checkpoint. Do **not** swap in the original HF checkpoint — its
  `pad_token == eos_token` causes infinite generation after fine-tuning.
- Mistral-7B-v0.3 uses the extended 32,768-token v3 tokenizer; keep the
  canonical instruction format exactly.
- Run a small smoke test (a few steps) before committing to the full 3-hour
  run, per the design doc's risk section.
- Checkpoint adapters live under `outputs/` which is gitignored
  (`*.safetensors`, `checkpoints/`) — copy final adapters to the Network
  Volume; don't rely on the container disk.

## Fallbacks if RunPod has no 4090 availability

- Kaggle Notebooks: 30 free GPU-hrs/week (P100 16 GB — enough for 7B QLoRA),
  ~9–12 hr per run, 9-hr session cap requires checkpointing.
- Modal Labs: $30/mo recurring credit, A100-40GB at $2.10/hr — covers the
  pilot for free.
