"""
QLoRA fine-tuning script — Sherlock Investigates pilot.

Loads a YAML config (configs/pilot_qwen.yaml or configs/pilot_mistral.yaml),
trains a LoRA adapter on data/augmented/train.jsonl using causal language
modeling with sequence packing, and saves the adapter to output_dir.

Prefers Unsloth if installed (faster, ~30% less memory); falls back to
standard transformers + PEFT + bitsandbytes if Unsloth is not available.

Usage (from repo root, venv active, RunPod or local GPU):
    pip install unsloth              # optional but strongly recommended
    pip install wandb                # optional — set WANDB_API_KEY to enable
    python3 scripts/training/train_lora.py --config configs/pilot_qwen.yaml
    python3 scripts/training/train_lora.py --config configs/pilot_mistral.yaml

To resume from a checkpoint:
    python3 scripts/training/train_lora.py --config configs/pilot_qwen.yaml \\
        --resume-from outputs/pilot_qwen_seed42/checkpoint-200

Required on RunPod (most are pre-installed on the PyTorch template):
    transformers, peft, bitsandbytes, accelerate, datasets, pyyaml, tqdm
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    from unsloth import FastLanguageModel
    _HAS_UNSLOTH = True
except ImportError:
    _HAS_UNSLOTH = False

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

try:
    import wandb  # noqa: F401
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = [
    "base_model", "lora_rank", "lora_alpha", "lora_dropout", "target_modules",
    "learning_rate", "lr_scheduler", "warmup_ratio",
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "num_epochs", "max_seq_length", "seed",
    "training_corpus_path", "heldout_corpus_path", "output_dir", "run_name",
]


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in _REQUIRED_KEYS if cfg.get(k) is None]
    if missing:
        sys.exit(
            f"ERROR: config has missing or null values: {', '.join(missing)}\n"
            f"       Edit {path} and fill in all required fields before training."
        )
    return cfg


# ---------------------------------------------------------------------------
# Data: load → tokenise → pack into fixed-length blocks
# ---------------------------------------------------------------------------

def load_texts(jsonl_path: Path) -> list[str]:
    texts = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
    return texts


def pack_into_blocks(texts: list[str], tokenizer, max_seq_length: int) -> Dataset:
    """
    Append EOS after each document, concatenate all tokens, then chunk into
    non-overlapping blocks of exactly max_seq_length. The trailing partial
    block is discarded. Returns a Dataset with a single 'input_ids' column.
    """
    eos = tokenizer.eos_token_id
    pool: list[int] = []
    for text in texts:
        pool.extend(tokenizer.encode(text, add_special_tokens=False))
        pool.append(eos)

    n_blocks = len(pool) // max_seq_length
    if n_blocks == 0:
        sys.exit(
            f"ERROR: {len(pool)} total tokens < max_seq_length {max_seq_length}. "
            "Not enough data to form a single block — check your training corpus."
        )

    blocks = [
        {"input_ids": pool[i * max_seq_length:(i + 1) * max_seq_length]}
        for i in range(n_blocks)
    ]
    discarded = len(pool) % max_seq_length
    print(
        f"  Packed {len(texts)} documents → {n_blocks} blocks of {max_seq_length} tokens "
        f"({discarded} trailing tokens discarded)"
    )
    return Dataset.from_list(blocks)


# ---------------------------------------------------------------------------
# Model loading: Unsloth path
# ---------------------------------------------------------------------------

def _load_unsloth(cfg: dict):
    print("Backend: Unsloth")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,  # auto: bfloat16 on Ampere+, float16 otherwise
    )
    modules_to_save = cfg.get("modules_to_save") or []
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        modules_to_save=modules_to_save if modules_to_save else None,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Model loading: standard transformers + PEFT path
# ---------------------------------------------------------------------------

def _load_standard(cfg: dict):
    print("Backend: transformers + PEFT + bitsandbytes  (install unsloth for faster training)")
    if not _HAS_PEFT:
        sys.exit("ERROR: peft not installed. Run: pip install peft bitsandbytes")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)

    # Qwen base models have pad_token == eos_token which causes infinite generation.
    # Add a distinct pad token. Unsloth handles this automatically on its path.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        print("  Added <|pad|> as pad token (pad==eos issue)")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Resize embeddings if we added a new pad token
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    modules_to_save = cfg.get("modules_to_save") or None
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        modules_to_save=modules_to_save,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Sherlock Investigates."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config (e.g. configs/pilot_qwen.yaml)",
    )
    parser.add_argument(
        "--resume-from", default=None, dest="resume_from",
        help="Resume training from a checkpoint directory",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        sys.exit(f"ERROR: config not found: {config_path}")

    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    print(f"\n{'='*60}")
    print(f"  Sherlock Investigates — LoRA fine-tuning")
    print(f"  Config : {config_path.name}")
    print(f"  Run    : {cfg['run_name']}")
    print(f"  Base   : {cfg['base_model']}")
    print(f"  Rank/α : {cfg['lora_rank']} / {cfg['lora_alpha']}")
    print(f"  Seed   : {cfg['seed']}")
    print(f"{'='*60}\n")

    # Load model
    if _HAS_UNSLOTH:
        model, tokenizer = _load_unsloth(cfg)
    else:
        model, tokenizer = _load_standard(cfg)

    model.print_trainable_parameters()

    # Load and pack training data
    corpus_path = ROOT / cfg["training_corpus_path"]
    if corpus_path.is_dir():
        corpus_path = corpus_path / "train.jsonl"
    if not corpus_path.exists():
        sys.exit(f"ERROR: training corpus not found: {corpus_path}")

    print(f"\nLoading training data: {corpus_path.relative_to(ROOT)}")
    texts = load_texts(corpus_path)
    print(f"  {len(texts)} examples")
    dataset = pack_into_blocks(texts, tokenizer, cfg["max_seq_length"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    output_dir   = ROOT / cfg["output_dir"]
    logging_steps = cfg.get("logging_steps") or 10
    save_steps    = cfg.get("save_steps") or 50
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=bf16,
        fp16=fp16,
        optim="paged_adamw_8bit",
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        report_to="wandb" if _HAS_WANDB else "none",
        run_name=cfg["run_name"] if _HAS_WANDB else None,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    eff_batch = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    print(f"\nTraining: {len(dataset)} blocks × {cfg['num_epochs']} epochs")
    print(f"Effective batch size: {eff_batch}  "
          f"(per_device={cfg['per_device_train_batch_size']} × "
          f"grad_accum={cfg['gradient_accumulation_steps']})")
    print(f"Checkpoints → {output_dir.relative_to(ROOT)}")
    if not _HAS_WANDB:
        print("W&B not installed — logging to stdout only")
    print()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)

    # Save final adapter
    adapter_path = output_dir / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nFinal adapter saved → {adapter_path.relative_to(ROOT)}")
    print(f"Upload to HuggingFace Hub:")
    print(f"  huggingface-cli upload <username>/sherlock-investigates-{cfg['run_name']} "
          f"{adapter_path}")


if __name__ == "__main__":
    main()
