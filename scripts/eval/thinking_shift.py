#!/usr/bin/env python3
"""Behavioural effect check: how does the THINK BLOCK shift, base vs fine-tuned?

This is the verification perplexity could not give. The low-rank run showed the
format survives and perplexity drops — but ~34 of the 44 PPL points were generic
prose-LM recovery (WikiText dropped too), so perplexity does NOT establish that
the model *reasons* differently. This runs the SAME deduction-inviting prompts
through the base model and one fine-tuned adapter and puts their think blocks
side by side, so a human can read whether the reasoning shifted.

DESIGN:
  - Load base ONCE, generate with base (disable_adapter), then attach the
    adapter and generate on the identical prompts (swap-adapter pattern).
  - GREEDY decoding (do_sample=False) so any base-vs-fine-tuned difference is
    from the weights, not the sampling dice.
  - Prompt set is the committed probe set: DEDUCTION_INVITING + REASONING_REQUIRED
    + NEUTRAL. NEUTRAL is the built-in control — a genuine reasoning shift should
    move the deduction prompts MORE than the neutral small-talk.

OUTPUT (primary is the transcript, NOT the numbers):
  - <out>_transcript.md : paired base/fine-tuned think blocks, for reading.
  - <out>.json          : structured pairs + a descriptive register profile.

The register profile (deduction/hedging marker rates) is DESCRIPTIVE ONLY. The
project has already established this task is NOT lexical (stance-detector
precision 0.185; markers saturate R1 traces — see probe_eval.py and the
2026-08-07 Decision Log). Read the transcripts; treat the markers as a rough
compass, never a verdict.

    python scripts/eval/thinking_shift.py \
        --adapter utsvsngh/sherlock-r1distill-7b-lowrank-r8 --subfolder checkpoint-50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "conversation"))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import hf_persist  # noqa: E402

# Copied from probe_eval.py (kept local so this module imports without torch).
# These SATURATE R1 think blocks — hence descriptive only.
DEDUCTION_MARKERS = [
    r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bclearly\b",
    r"\bevidently\b", r"\bmust be\b", r"\bcan only\b", r"\bit follows\b",
    r"\bI deduce\b", r"\bthis indicates\b", r"\bthis suggests\b",
    r"\bI conclude\b", r"\bthis means\b", r"\bit is clear\b",
    r"\bobviously\b", r"\bwithout doubt\b", r"\bwithout question\b",
    r"\bwe can see\b", r"\bone can see\b",
]
HEDGING_MARKERS = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bmight\b", r"\bcould be\b",
    r"\bpossibly\b", r"\bI'm not sure\b", r"\bI am not sure\b",
    r"\bI think\b", r"\bI guess\b", r"\buncertain\b", r"\bunclear\b",
    r"\bhard to say\b", r"\bdifficult to (say|determine|know)\b",
    r"\bcan't be sure\b", r"\bcannot be sure\b",
]
_DED = re.compile("|".join(DEDUCTION_MARKERS), re.IGNORECASE)
_HED = re.compile("|".join(HEDGING_MARKERS), re.IGNORECASE)


def register_profile(think: str | None) -> dict:
    """Descriptive marker rates for one think block. Per-1000-words so a longer
    trace does not automatically score higher. NOT a Holmes-ness score."""
    if not think:
        return {"words": 0, "deduction_per_1k": 0.0, "hedging_per_1k": 0.0,
                "deduction_hits": 0, "hedging_hits": 0}
    words = max(1, len(think.split()))
    d = len(_DED.findall(think))
    h = len(_HED.findall(think))
    return {"words": words,
            "deduction_hits": d, "hedging_hits": h,
            "deduction_per_1k": round(d / words * 1000, 2),
            "hedging_per_1k": round(h / words * 1000, 2)}


def aggregate_by_category(pairs: list[dict]) -> dict:
    """Base-vs-fine-tuned register means per prompt category + deltas. The
    NEUTRAL category is the control: the deduction categories should move more."""
    cats: dict[str, dict] = {}
    for p in pairs:
        c = cats.setdefault(p["category"], {
            "n": 0, "base_ded": 0.0, "ft_ded": 0.0,
            "base_hed": 0.0, "ft_hed": 0.0,
            "base_words": 0.0, "ft_words": 0.0})
        c["n"] += 1
        c["base_ded"] += p["base_profile"]["deduction_per_1k"]
        c["ft_ded"] += p["ft_profile"]["deduction_per_1k"]
        c["base_hed"] += p["base_profile"]["hedging_per_1k"]
        c["ft_hed"] += p["ft_profile"]["hedging_per_1k"]
        c["base_words"] += p["base_profile"]["words"]
        c["ft_words"] += p["ft_profile"]["words"]
    out = {}
    for cat, c in cats.items():
        n = c["n"]
        out[cat] = {
            "n": n,
            "base_deduction_per_1k": round(c["base_ded"] / n, 2),
            "ft_deduction_per_1k": round(c["ft_ded"] / n, 2),
            "delta_deduction_per_1k": round((c["ft_ded"] - c["base_ded"]) / n, 2),
            "base_hedging_per_1k": round(c["base_hed"] / n, 2),
            "ft_hedging_per_1k": round(c["ft_hed"] / n, 2),
            "delta_hedging_per_1k": round((c["ft_hed"] - c["base_hed"]) / n, 2),
            "base_mean_words": round(c["base_words"] / n, 1),
            "ft_mean_words": round(c["ft_words"] / n, 1),
        }
    return out


def write_transcript(pairs: list[dict], path: Path, meta: dict) -> None:
    lines = ["# Thinking-shift transcript — base vs fine-tuned",
             "",
             f"adapter: `{meta['adapter']}`"
             + (f" (subfolder `{meta['subfolder']}`)" if meta.get("subfolder") else ""),
             f"base: `{meta['base']}`  |  decoding: greedy  |  "
             f"max_new_tokens: {meta['max_new_tokens']}",
             "",
             "> The register numbers in the JSON are descriptive only — this task "
             "is not lexical. Read the reasoning below.", ""]
    for p in pairs:
        lines += [f"## [{p['id']}] {p['category']}", "",
                  f"**Prompt:** {p['prompt']}", "",
                  "**BASE think:**", "", "```", (p["base_think"] or "(no think block)").strip(), "```", "",
                  "**FINE-TUNED think:**", "", "```", (p["ft_think"] or "(no think block)").strip(), "```", "",
                  f"_markers/1k — base: ded {p['base_profile']['deduction_per_1k']} "
                  f"hed {p['base_profile']['hedging_per_1k']} | "
                  f"fine-tuned: ded {p['ft_profile']['deduction_per_1k']} "
                  f"hed {p['ft_profile']['hedging_per_1k']}_", "", "---", ""]
    path.write_text("\n".join(lines))


def load_prompts(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True, help="HF repo id or local dir")
    ap.add_argument("--subfolder", default=None, help="e.g. checkpoint-50")
    ap.add_argument("--base", default="unsloth/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--prompts", default="data/probes/probe_set_v1.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (default; clean paired diff). >0 samples.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--hf-results-repo", default=None)
    ap.add_argument("--push-results", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed)

    from agent import _resolve_think_block

    set_seed(args.seed)
    prompts_path = ROOT / args.prompts if not Path(args.prompts).is_absolute() else Path(args.prompts)
    prompts = load_prompts(prompts_path, args.limit)

    hf_repo = os.environ.get("HF_RESULTS_REPO") or args.hf_results_repo
    hf_token = hf_persist.find_hf_token() if hf_repo else None

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map={"": 0}, trust_remote_code=True)
    model = PeftModel.from_pretrained(
        base, args.adapter, subfolder=args.subfolder, adapter_name="ft")
    model.eval()
    print(f"  base   : {args.base}")
    print(f"  adapter: {args.adapter}" + (f" / {args.subfolder}" if args.subfolder else ""))
    print(f"  prompts: {len(prompts)}  decoding: "
          f"{'greedy' if args.temperature == 0 else f'temp {args.temperature}'}")

    def gen(prompt: str) -> tuple[str | None, str]:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt").to(model.device)
        kw = dict(max_new_tokens=args.max_new_tokens,
                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if args.temperature and args.temperature > 0:
            kw.update(do_sample=True, temperature=args.temperature)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(ids, **kw)
        txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=False)
        return _resolve_think_block(txt, {})

    pairs = []
    for p in prompts:
        with model.disable_adapter():
            b_think, b_ans = gen(p["prompt"])
        model.set_adapter("ft")
        f_think, f_ans = gen(p["prompt"])
        pairs.append({
            "id": p["id"], "category": p["category"], "prompt": p["prompt"],
            "expected_direction": p.get("expected_direction"),
            "base_think": b_think, "base_answer": b_ans,
            "ft_think": f_think, "ft_answer": f_ans,
            "base_profile": register_profile(b_think),
            "ft_profile": register_profile(f_think)})
        bp, fp = pairs[-1]["base_profile"], pairs[-1]["ft_profile"]
        print(f"  [{p['id']:>2}] {p['category']:<18} "
              f"ded {bp['deduction_per_1k']:>5}->{fp['deduction_per_1k']:<5} "
              f"hed {bp['hedging_per_1k']:>5}->{fp['hedging_per_1k']:<5}", flush=True)

    agg = aggregate_by_category(pairs)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"base": args.base, "adapter": args.adapter, "subfolder": args.subfolder,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "seed": args.seed}
    out = out_dir / f"thinking_shift_{stamp}.json"
    out.write_text(json.dumps({**meta, "aggregate_by_category": agg, "pairs": pairs},
                              indent=2))
    md = out_dir / f"thinking_shift_{stamp}_transcript.md"
    write_transcript(pairs, md, meta)

    print("\n  REGISTER PROFILE (descriptive — read the transcript, don't gate on this)")
    print(f"  {'category':<20}{'ded base->ft (Δ)':<22}{'hed base->ft (Δ)':<22}")
    for cat, c in agg.items():
        print(f"  {cat:<20}"
              f"{c['base_deduction_per_1k']:>5}->{c['ft_deduction_per_1k']:<5}"
              f"({c['delta_deduction_per_1k']:>+5})     "
              f"{c['base_hedging_per_1k']:>5}->{c['ft_hedging_per_1k']:<5}"
              f"({c['delta_hedging_per_1k']:>+5})")
    print(f"\n  wrote {out}")
    print(f"  wrote {md}  <-- READ THIS")

    for f in (out, md):
        if hf_repo and hf_token:
            hf_persist.upload_path(f, hf_repo, hf_token, path_in_repo=f.name,
                                   repo_type="dataset", private=True, label=f.name)
    if args.push_results:
        from dose_curve import git_push_results
        git_push_results([out, md], f"results: thinking-shift {stamp}")


if __name__ == "__main__":
    main()
