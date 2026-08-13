#!/usr/bin/env python3
"""Measure the BEHAVIOURAL EFFECT at every checkpoint — held-out perplexity.

The companion to dose_curve.py. Where that measures whether the <think> format
SURVIVES (closure), this measures whether the fine-tune LEARNED anything:
perplexity on the held-out Speckled Band (unseen Holmes) at each checkpoint. A
drop vs base = the model absorbed Holmes distribution = the effect is present.

Together the two curves answer the question a closure curve alone cannot: is
there a dose where the format is intact AND an effect exists? (mitigation_
analysis.py overlays them.) This matters because low rank could "preserve
closure" simply by learning too little to matter — perplexity catches that.

METHOD. Identical to dose_curve.py: load the base ONCE, swap exactly one adapter
at a time (constant VRAM), measure every checkpoint against identical base
weights in one session. Reuses perplexity.compute_perplexity (sliding window)
so the numbers match the H1 gate.

    python scripts/eval/effect_curve.py \
        --checkpoint-dir outputs/kaggle_t4_lowrank_r8_seed42 \
        --heldout data/processed/heldout/speckled_band.txt

Writes results/analysis/effect_curve_<timestamp>.json + a printed table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import hf_persist  # noqa: E402
from dose_curve import checkpoint_steps  # noqa: E402 — reuse the ckpt lister (torch-free)
# perplexity.compute_perplexity is imported inside main() — perplexity.py imports
# torch at module top, and keeping it out of module scope means effect_curve
# stays importable in a torch-free env (mirrors dose_curve.py's pattern).

WIKITEXT_TOKEN_CAP = 8_192


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--base", default="unsloth/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--heldout", default="data/processed/heldout/speckled_band.txt")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every Nth checkpoint (the last is always kept)")
    ap.add_argument("--with-wikitext", action="store_true",
                    help="also track WikiText-2 PPL (H2 guardrail). Needs "
                         "internet; failures are caught, not fatal.")
    ap.add_argument("--hf-results-repo", default=None)
    ap.add_argument("--push-results", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from perplexity import compute_perplexity  # torch already imported above

    hf_results_repo = os.environ.get("HF_RESULTS_REPO") or args.hf_results_repo
    hf_token = hf_persist.find_hf_token() if hf_results_repo else None
    if hf_results_repo and not hf_token:
        print("  WARNING: --hf-results-repo set but NO HF token — results will "
              "NOT be persisted to the Hub.", flush=True)

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    ckpts = checkpoint_steps(ckpt_dir)
    final = ckpt_dir / "final_adapter"
    if final.exists():
        ckpts.append((10**6, final))
    if not ckpts:
        sys.exit(f"ERROR: no checkpoint-N dirs in {ckpt_dir}.")
    if args.stride > 1:
        kept = ckpts[::args.stride]
        if ckpts[-1] not in kept:
            kept.append(ckpts[-1])
        ckpts = kept

    heldout_path = Path(args.heldout)
    if not heldout_path.is_absolute():
        heldout_path = ROOT / heldout_path
    heldout_text = heldout_path.read_text()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    partial = out_dir / f"effect_curve_{stamp}.partial.jsonl"
    partial_f = partial.open("w", encoding="utf-8")

    print(f"  live log    : {partial}")
    print(f"  checkpoints : {len(ckpts)}")
    print(f"  held-out    : {heldout_path.name} ({len(heldout_text):,} chars)")

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True)

    wiki_text = None
    if args.with_wikitext:
        try:
            from datasets import load_dataset
            wiki_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            joined = "\n".join(t for t in wiki_raw["text"] if t.strip())
            wiki_ids = tok(joined, return_tensors="pt").input_ids[0, :WIKITEXT_TOKEN_CAP]
            wiki_text = tok.decode(wiki_ids, skip_special_tokens=True)
            print(f"  wikitext    : capped to {wiki_ids.size(0)} tokens")
        except Exception as e:  # noqa: BLE001 — guardrail, not the primary signal
            print(f"  wikitext    : unavailable ({type(e).__name__}: {e}) — skipping")

    def persist_partial() -> None:
        if hf_results_repo and hf_token:
            hf_persist.upload_path(partial, hf_results_repo, hf_token,
                                   path_in_repo=partial.name, repo_type="dataset",
                                   private=True, label=partial.name)

    def measure(model, label) -> dict:
        row = {"label": label,
               "heldout_ppl": compute_perplexity(model, tok, heldout_text)}
        if wiki_text is not None:
            row["wikitext_ppl"] = compute_perplexity(model, tok, wiki_text)
        partial_f.write(json.dumps(row) + "\n")
        partial_f.flush()
        os.fsync(partial_f.fileno())
        persist_partial()
        return row

    ADAPTER = "cur"
    model = PeftModel.from_pretrained(base, str(ckpts[0][1]), adapter_name=ADAPTER)
    model.eval()

    rows = []
    # Base FIRST (the reference for every drop).
    with model.disable_adapter():
        r = measure(model, "base")
    base_ppl = r["heldout_ppl"]
    r["heldout_drop_pct"] = 0.0
    rows.append(r)
    print(f"\n  {'base':<12} heldout PPL {r['heldout_ppl']:.3f}")

    for i, (step, path) in enumerate(ckpts):
        if i > 0:
            model.delete_adapter(ADAPTER)
            import torch as _t
            _t.cuda.empty_cache()
            model.load_adapter(str(path), adapter_name=ADAPTER)
        model.set_adapter(ADAPTER)
        label = "final" if step == 10**6 else f"step-{step}"
        r = measure(model, label)
        r["step"] = None if step == 10**6 else step
        r["heldout_drop_pct"] = round((base_ppl - r["heldout_ppl"]) / base_ppl * 100, 2)
        rows.append(r)
        print(f"  {label:<12} heldout PPL {r['heldout_ppl']:.3f}  "
              f"drop {r['heldout_drop_pct']:+.1f}%", flush=True)

    partial_f.close()
    out = out_dir / f"effect_curve_{stamp}.json"
    out.write_text(json.dumps({
        "checkpoint_dir": str(ckpt_dir), "base": args.base,
        "heldout": str(heldout_path), "base_heldout_ppl": base_ppl,
        "h1_gate_pct": 5.0, "rows": rows,
    }, indent=2))

    print(f"\n  wrote {out}")
    persisted = False
    if hf_results_repo and hf_token:
        persisted = hf_persist.upload_path(out, hf_results_repo, hf_token,
                                           path_in_repo=out.name, repo_type="dataset",
                                           private=True, label=out.name)
    if args.push_results:
        from dose_curve import git_push_results
        persisted = git_push_results([out, partial],
                                     f"results: effect curve {stamp}") or persisted
    if not persisted:
        print("  NOTE: effect curve is LOCAL ONLY — download it before session end "
              "if on ephemeral compute.")


if __name__ == "__main__":
    main()
