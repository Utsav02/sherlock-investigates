"""
Stage 0 — Kaggle T4 pipeline validation.  Paste into a Kaggle notebook cell.

WHAT THIS ANSWERS (the only question):
    Does the QLoRA fine-tuning path run end to end on free hardware, and does
    the resulting adapter STILL EMIT <think> BLOCKS that agent.py can extract?

WHAT IT DOES NOT ANSWER:
    Whether fine-tuning on Holmes shifts a reasoning prior. This run is ~29
    optimizer steps. Reading the loss curve as evidence about the hypothesis
    would be wrong. The eval gates (perplexity / WikiText / MMLU / probe
    separation) belong to the real training run on better hardware.

WHY THINK BLOCKS ARE THE TEST:
    The entire three-level commitment gap depends on <think> content surviving
    fine-tuning. If QLoRA on raw prose degrades the reasoning format, the
    project's novel measurement disappears and no amount of GPU budget fixes
    it. This is the cheapest possible place to find that out — $0, ~30 min.

KAGGLE SETUP BEFORE RUNNING:
    1. Notebook Settings -> Accelerator -> "GPU T4 x2"  (only GPU 0 is used;
       device_map is pinned to a single device below so the model is not split)
    2. Notebook Settings -> Internet -> ON  (needed to pull weights from HF)
    3. Session type: "Save & Run All" if you want it to survive a browser close

RUNTIME: ~20-40 min including a ~5GB model download.
COST: $0.
"""

# =============================================================================
# CELL 1 — Environment check. Run this FIRST and read the output.
# =============================================================================
# Fail fast on the wrong accelerator rather than 20 minutes into a download.

import subprocess
import sys

import torch

print("=" * 70)
print("  ENVIRONMENT")
print("=" * 70)

if not torch.cuda.is_available():
    raise SystemExit(
        "No GPU. Notebook Settings -> Accelerator -> GPU T4 x2, then restart."
    )

props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
# Compute capability determines which dtypes the tensor cores support.
# T4 = Turing = SM 7.5 -> fp16 only, NO bfloat16.
# A100/L4/4090 = Ampere+ = SM 8.0+ -> bfloat16 available.
sm = f"{props.major}.{props.minor}"
bf16_ok = torch.cuda.is_bf16_supported()

print(f"  GPU              : {props.name}")
print(f"  VRAM             : {vram_gb:.1f} GB")
print(f"  Compute capability: SM {sm}")
print(f"  bfloat16 supported: {bf16_ok}   -> training will use "
      f"{'bf16' if bf16_ok else 'fp16'}")
print(f"  CUDA devices     : {torch.cuda.device_count()}")

# 7B in 4-bit is ~4.5GB of weights. Add LoRA gradients, 8-bit Adam state and
# 2048-token activations and the run needs roughly 9-12GB. Below ~14GB free the
# backward pass is the thing that OOMs, usually several minutes in.
if vram_gb < 14:
    print(f"\n  WARNING: {vram_gb:.1f}GB is tight for 7B QLoRA at seq len 2048.")
    print("  If it OOMs, drop max_seq_length to 1024 in the config.")


# =============================================================================
# CELL 2 — Dependencies.
# =============================================================================
# Kaggle images ship torch + transformers, but usually a transformers too old
# for DeepSeek-R1-Distill's chat template and a bitsandbytes too old for
# paged_adamw_8bit. Pin explicitly rather than trusting the image.
#
# unsloth is NOT installed here. It is faster and lighter, but its install
# frequently conflicts with Kaggle's pinned torch build and burns session time.
# train_lora.py falls back to transformers+PEFT automatically when unsloth is
# absent, and that path is what we want to validate anyway — it is what will
# run on RunPod.

DEPS = [
    "transformers==4.57.1",   # DeepSeek-R1-Distill chat template support
    "peft==0.18.0",           # LoRA / QLoRA adapters
    "bitsandbytes==0.48.2",   # 4-bit NF4 quantization + paged 8-bit Adam
    "accelerate==1.11.0",     # device placement, used internally by Trainer
    "datasets==4.8.5",        # matches requirements.txt so tokenisation matches
    "pyyaml",                 # config loading
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DEPS], check=True)
print("dependencies installed — RESTART THE KERNEL now, then skip to CELL 3")


# =============================================================================
# CELL 3 — Get the repo onto the machine.
# =============================================================================
# The corpus (data/augmented/train.jsonl) is tracked in git and is only ~1.5MB,
# so cloning brings the training data with it. No separate upload needed.
#
# If the repo has been made PRIVATE, replace the clone URL with:
#   https://<YOUR_GITHUB_TOKEN>@github.com/Utsav02/sherlock-investigates.git
# using a Kaggle Secret, NOT a literal token in the notebook:
#   from kaggle_secrets import UserSecretsClient
#   tok = UserSecretsClient().get_secret("GITHUB_TOKEN")

import os
from pathlib import Path

REPO_URL = "https://github.com/Utsav02/sherlock-investigates.git"
REPO_DIR = Path("/kaggle/working/sherlock-investigates")

if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
                   check=True)
os.chdir(REPO_DIR)
print(f"repo at {REPO_DIR}")

# Weights & Biases off. It prompts for a login and blocks the run otherwise.
# train_lora.py honours these and sets report_to='none'.
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"


# =============================================================================
# CELL 4 — Preflight. Verify the invariants BEFORE burning session time.
# =============================================================================
# House rule: check flags with arithmetic, confirm the data is real, and
# dry-run the shape before launching anything long.

import json

import yaml

CONFIG = Path("configs/kaggle_t4_validation.yaml")
cfg = yaml.safe_load(CONFIG.read_text())

corpus = Path(cfg["training_corpus_path"])
assert corpus.exists(), f"missing training corpus: {corpus}"

n_examples = sum(1 for line in corpus.open() if line.strip())
# Rough token estimate: ~4 chars per token for English prose. Good enough for a
# sanity check; the real count is printed by train_lora.py after tokenisation.
n_chars = sum(len(json.loads(line)["text"]) for line in corpus.open() if line.strip())
est_tokens = n_chars // 4

seq = cfg["max_seq_length"]
eff_batch = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
est_blocks = est_tokens // seq
est_steps = max(1, est_blocks * cfg["num_epochs"] // eff_batch)

print("=" * 70)
print("  PREFLIGHT")
print("=" * 70)
print(f"  corpus            : {corpus}")
print(f"  examples          : {n_examples:,}")
print(f"  est. tokens       : {est_tokens:,}")
print(f"  seq length        : {seq}")
print(f"  est. blocks       : {est_blocks:,}")
print(f"  effective batch   : {eff_batch}  "
      f"({cfg['per_device_train_batch_size']} x {cfg['gradient_accumulation_steps']})")
print(f"  epochs            : {cfg['num_epochs']}")
print(f"  EST. OPTIM STEPS  : {est_steps}")
print(f"  save_steps        : {cfg['save_steps']}")
print(f"  base model        : {cfg['base_model']}")

# save_steps must be <= total steps or no checkpoint is ever written and a
# session timeout loses the entire run. This is why the main configs' save_steps
# of 50 is wrong for a ~29-step job.
if cfg["save_steps"] > est_steps:
    print(f"\n  WARNING: save_steps ({cfg['save_steps']}) > est. steps ({est_steps})")
    print("  No checkpoint would be written. Lower save_steps.")

# The full canon would turn a plumbing check into a multi-hour job.
if "full_canon" in str(corpus):
    print("\n  WARNING: this is the FULL CANON (~3.44M tokens, ~314 steps).")
    print("  Stage 0 should use data/augmented/train.jsonl (~29 steps).")

assert "modules_to_save" not in cfg, (
    "modules_to_save trains embed_tokens + lm_head (~1.1B params in fp32) and "
    "will OOM on a 16GB T4. Remove it for this run."
)
print("\n  preflight OK")


# =============================================================================
# CELL 5 — Train.
# =============================================================================
# Uses the repo's own training script rather than reimplementing it inline:
# whatever runs here is exactly what will run on RunPod, so a bug found here is
# a bug fixed for the real run. Reimplementing in the notebook would validate
# the notebook, not the pipeline.
#
# Expect the first ~5 minutes to be the model download (~5GB), during which
# nothing prints. That is normal.

subprocess.run(
    [sys.executable, "scripts/training/train_lora.py",
     "--config", str(CONFIG)],
    check=True,
)


# =============================================================================
# CELL 6 — THE ACTUAL TEST: do think blocks survive fine-tuning?
# =============================================================================
# Everything above is setup. This cell is the reason the run exists.
#
# Loads the trained adapter, generates on prompts that should elicit reasoning,
# and checks for <think> content using THE SAME extraction agent.py uses — not
# a reimplementation, because a divergence between the two would make this test
# pass while the real orchestrator fails.

import sys as _sys

_sys.path.insert(0, str(REPO_DIR / "scripts" / "conversation"))
from agent import _resolve_think_block  # the production extractor

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER = Path(cfg["output_dir"]) / "final_adapter"
assert ADAPTER.exists(), f"no adapter at {ADAPTER} — did CELL 5 finish?"

# Same quantization as training. Loading the adapter onto a differently-
# quantized base would test a configuration that will never be run for real.
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # Must match the GPU: fp16 on T4, bf16 on Ampere+. See CELL 1.
    bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
)

tok = AutoTokenizer.from_pretrained(str(ADAPTER))
base = AutoModelForCausalLM.from_pretrained(
    cfg["base_model"],
    quantization_config=bnb,
    device_map={"": 0},   # pin to GPU 0; "auto" would shard across both T4s
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, str(ADAPTER))
model.eval()

# Deduction-inviting prompts. Holmes-flavoured because a fine-tune that took
# should reason in that register; the format check works regardless of content.
PROMPTS = [
    "You meet a stranger whose left cuff is frayed and whose right shoe is "
    "newly resoled. What do you conclude, and why?",
    "A man claims he walked here from the station, but his umbrella is dry on "
    "a rainy day. What follows?",
    "Someone in conversation uses a phrase that sounds slightly too formal. "
    "How would you decide whether that means anything?",
]

print("=" * 70)
print("  THINK-BLOCK SURVIVAL TEST")
print("=" * 70)

n_with_think = 0
for i, prompt in enumerate(PROMPTS, 1):
    # The chat template is what triggers R1-distill's thinking format. Applying
    # it wrongly is the single most common cause of think blocks vanishing —
    # which is precisely the failure this cell exists to catch.
    messages = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=400,   # think blocks in the pilot averaged ~1.4-1.8K
                                  # chars; 400 tokens is enough to see whether a
                                  # block STARTS, which is all we are testing.
            temperature=0.7,      # same as the orchestrator (agent.py), so this
                                  # exercises the real sampling configuration.
            do_sample=True,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=False)

    # Production extractor: handles inline <think> tags AND the separate
    # reasoning field some servers return. {} because HF generation has no
    # message_extra — inline tags are the only transport here.
    think, remainder = _resolve_think_block(text, {})
    ok = bool(think and think.strip())
    n_with_think += ok

    print(f"\n  [{i}/{len(PROMPTS)}] think block: {'YES' if ok else 'NO'}"
          f"  ({len(think) if think else 0} chars)")
    print(f"      prompt: {prompt[:70]}...")
    if ok:
        print(f"      think : {think[:200].strip()}...")
    else:
        print(f"      raw   : {text[:200].strip()}...")

print("\n" + "=" * 70)
print(f"  RESULT: {n_with_think}/{len(PROMPTS)} generations contained a think block")
if n_with_think == len(PROMPTS):
    print("  PASS — the reasoning format survived fine-tuning.")
    print("  Stage 0 complete. Proceed to the 14B run on RunPod.")
elif n_with_think > 0:
    print("  PARTIAL — intermittent. Check the chat template and whether")
    print("  training data leaked stray '<think>' strings into the corpus.")
else:
    print("  FAIL — fine-tuning destroyed the reasoning format.")
    print("  STOP. Do not spend GPU budget. The three-level commitment gap")
    print("  depends on this, and no amount of compute recovers it.")
    print("  First things to check: chat template in the Unsloth checkpoint,")
    print("  and whether modules_to_save accidentally trained the embeddings.")
print("=" * 70)


# =============================================================================
# CELL 7 — Persist the adapter. Kaggle deletes /kaggle/working on session end.
# =============================================================================
# The adapter is ~150MB at rank 32. Two options:
#
# (a) Download it from the Kaggle file browser (right pane -> /kaggle/working).
# (b) Push to HuggingFace, which is what RunPod will pull from later.
#     Store the token as a Kaggle Secret named HF_TOKEN — never inline it.
#
# Uncomment (b) when you have the secret set:
#
# from kaggle_secrets import UserSecretsClient
# from huggingface_hub import HfApi
# hf_token = UserSecretsClient().get_secret("HF_TOKEN")
# HfApi().upload_folder(
#     folder_path=str(ADAPTER),
#     repo_id="Utsav02/sherlock-r1distill-7b-validation",
#     repo_type="model",
#     token=hf_token,
#     private=True,   # matches the repo's IP posture; flip if you decide public
# )

print(f"adapter at: {ADAPTER.resolve()}")
print(f"size: {sum(f.stat().st_size for f in ADAPTER.rglob('*') if f.is_file())/1e6:.0f} MB")
print("\nKaggle deletes /kaggle/working when the session ends — download it or")
print("push to HuggingFace before closing the tab.")
