"""
Perplexity evaluation — pilot gates H1 and H2.

H1 (confirmatory): fine-tuned model achieves ≥5% lower perplexity on the
held-out Speckled Band text vs the base model.

H2 (confirmatory): perplexity on WikiText-2 changes by ≤±5% vs the base
model (no catastrophic forgetting of general language).

Loads the base model once, evaluates it, then loads the LoRA adapter on top
and evaluates again — no double download of base weights.

Usage (from repo root):
    python scripts/eval/perplexity.py \\
        --config  configs/pilot_qwen.yaml \\
        --adapter outputs/pilot_qwen_seed42/final_adapter \\
        --output  results/pilot/

    # Or pull adapter from HF Hub:
    python scripts/eval/perplexity.py \\
        --config  configs/pilot_qwen.yaml \\
        --adapter utsvsngh/sherlock-qwen25-7b-pilot-seed42 \\
        --output  results/pilot/

Exit code 0 = both gates pass. Exit code 1 = at least one gate fails.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Gate thresholds (from EXPERIMENT_DESIGN.md)
H1_MIN_DROP  = 0.05   # ≥5% perplexity reduction on Speckled Band
H2_MAX_DRIFT = 0.05   # ≤5% change on WikiText-2

# Cap WikiText tokens for speed; 8k tokens is enough to be stable
WIKITEXT_TOKEN_CAP = 8_192


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_base(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def attach_adapter(model, adapter_path: str):
    from peft import PeftModel
    print(f"  Attaching adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Perplexity with sliding window (handles texts longer than max_length)
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, text: str,
                       max_length: int = 1024, stride: int = 512) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids.to(model.device)
    seq_len = input_ids.size(1)

    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end   # only new tokens count toward loss

        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        labels[:, :-target_len] = -100  # mask context tokens

        with torch.inference_mode():
            loss = model(chunk, labels=labels).loss.item()

        total_nll   += loss * target_len
        total_tokens += target_len
        prev_end = end

        if end == seq_len:
            break

    return math.exp(total_nll / total_tokens)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Perplexity evaluation (H1 + H2)")
    parser.add_argument("--config",  required=True, help="YAML config path")
    parser.add_argument("--adapter", required=True, help="local adapter dir or HF Hub repo")
    parser.add_argument("--output",  default="results/pilot", help="output directory")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    base_model   = cfg["base_model"]
    heldout_path = ROOT / cfg["heldout_corpus_path"]
    out_dir      = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Perplexity evaluation")
    print(f"  Config:    {args.config}")
    print(f"  Base:      {base_model}")
    print(f"  Adapter:   {args.adapter}")
    print(f"  Held-out:  {heldout_path}")
    print("=" * 60)

    # Load texts
    heldout_text = heldout_path.read_text()
    print(f"\nHeld-out text: {len(heldout_text):,} chars")

    from datasets import load_dataset
    wiki_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_text = "\n".join(t for t in wiki_raw["text"] if t.strip())
    print(f"WikiText-2:    {len(wiki_text):,} chars (will cap at {WIKITEXT_TOKEN_CAP} tokens)")

    # ---- Base model ----
    print("\n[1/2] Base model")
    model, tokenizer = load_base(base_model)

    # Truncate WikiText to token cap once (reuse same slice for finetuned)
    wiki_ids = tokenizer(wiki_text, return_tensors="pt").input_ids[0, :WIKITEXT_TOKEN_CAP]
    wiki_text_capped = tokenizer.decode(wiki_ids, skip_special_tokens=True)
    print(f"  WikiText capped to {wiki_ids.size(0)} tokens")

    print("  PPL on Speckled Band ...", flush=True)
    base_heldout_ppl = compute_perplexity(model, tokenizer, heldout_text)
    print(f"  → {base_heldout_ppl:.4f}")

    print("  PPL on WikiText-2 ...", flush=True)
    base_wiki_ppl = compute_perplexity(model, tokenizer, wiki_text_capped)
    print(f"  → {base_wiki_ppl:.4f}")

    # ---- Fine-tuned model (adapter on top of base — no re-download) ----
    print("\n[2/2] Fine-tuned model")
    model = attach_adapter(model, args.adapter)

    print("  PPL on Speckled Band ...", flush=True)
    ft_heldout_ppl = compute_perplexity(model, tokenizer, heldout_text)
    print(f"  → {ft_heldout_ppl:.4f}")

    print("  PPL on WikiText-2 ...", flush=True)
    ft_wiki_ppl = compute_perplexity(model, tokenizer, wiki_text_capped)
    print(f"  → {ft_wiki_ppl:.4f}")

    # ---- Gate evaluation ----
    heldout_drop  = (base_heldout_ppl - ft_heldout_ppl) / base_heldout_ppl
    wiki_drift    = (ft_wiki_ppl - base_wiki_ppl) / base_wiki_ppl

    h1_pass = heldout_drop >= H1_MIN_DROP
    h2_pass = abs(wiki_drift) <= H2_MAX_DRIFT

    print("\n" + "=" * 60)
    print("GATE RESULTS")
    print("=" * 60)
    print(f"  Base PPL  — Speckled Band: {base_heldout_ppl:.4f}")
    print(f"  FT PPL    — Speckled Band: {ft_heldout_ppl:.4f}")
    print(f"  Drop: {heldout_drop*100:+.1f}%   (gate: ≥+5%)   {'✓ PASS' if h1_pass else '✗ FAIL'}")
    print()
    print(f"  Base PPL  — WikiText-2:    {base_wiki_ppl:.4f}")
    print(f"  FT PPL    — WikiText-2:    {ft_wiki_ppl:.4f}")
    print(f"  Drift: {wiki_drift*100:+.1f}%   (gate: ≤±5%)   {'✓ PASS' if h2_pass else '✗ FAIL'}")
    print("=" * 60)

    output = {
        "run_name":  cfg.get("run_name", "unknown"),
        "base_model": base_model,
        "adapter":   args.adapter,
        "base": {
            "speckled_band_ppl": base_heldout_ppl,
            "wikitext_ppl":      base_wiki_ppl,
        },
        "finetuned": {
            "speckled_band_ppl": ft_heldout_ppl,
            "wikitext_ppl":      ft_wiki_ppl,
        },
        "gates": {
            "H1_speckled_band_drop_pct": round(heldout_drop * 100, 2),
            "H1_pass":                   h1_pass,
            "H2_wikitext_drift_pct":     round(wiki_drift * 100, 2),
            "H2_pass":                   h2_pass,
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"perplexity_{cfg.get('run_name', 'run')}_{ts}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults → {out_path.relative_to(ROOT)}")

    if not (h1_pass and h2_pass):
        sys.exit(1)


if __name__ == "__main__":
    main()
