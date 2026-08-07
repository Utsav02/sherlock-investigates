#!/usr/bin/env python3
"""Measure think-block closure at every training checkpoint — the dose curve.

WHY. On 2026-08-06, n=8 per arm on identical prompts:
    base, no adapter              closure 8/8
    stage-0    (30 steps,  311K)  closure 8/8
    full canon (103 steps, 3.36M) closure 1/8
Something between 30 and 103 steps destroys the reasoning format. Two points
cannot distinguish a threshold from a slope, nor say whether a usable dose
window exists below the collapse. This script answers both by scoring every
checkpoint from one run.

METHOD. Loads the base model ONCE, then attaches exactly ONE checkpoint at a
time (load_adapter -> measure -> delete_adapter). Every arm is measured against
identical base weights in a single session, which is the validity requirement;
reloading the base per checkpoint would let fragmentation and load-order
differences leak into a curve whose whole purpose is isolating one variable.

Holding all checkpoints resident at once ALSO satisfies that requirement but
does not fit: 22 adapters x ~161MB is ~3.5GB on top of a 4.5GB 4-bit base, and
it OOM'd on a 14.6GB T4 on 2026-08-06. One-at-a-time gives the same guarantee
at constant memory.

RUN THIS IN A FRESH KERNEL. Any model still resident from an earlier cell holds
several GB and will OOM this script before it starts.

    python scripts/eval/dose_curve.py \
        --checkpoint-dir outputs/kaggle_t4_dosecurve_seed42 \
        --n-prompts 8

Writes results/analysis/dose_curve_<timestamp>.json plus a printed table.
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

# Conversational openers matching the REAL task, not open-ended puzzles.
# Open-ended prompts have a different and much higher-variance think length
# (837 tokens on one sample, >1200 on another for the SAME riddle), which is
# what produced two false FAILs on 2026-07-28. The real task sits at ~450.
OPENERS = [
    "Hey, how's your week going?",
    "What did you get up to this weekend?",
    "Long day here. You?",
    "Do you follow any sports?",
    "That's a very tidy way of putting it. Do you always phrase things that way?",
    "Where are you based?",
    "Any plans for the evening?",
    "How's the weather there?",
    "Did you see the news this morning?",
    "I've been meaning to fix my bike. You handy with that sort of thing?",
    "Coffee or tea person?",
    "What's the last thing that made you laugh?",
]


def checkpoint_steps(d: Path) -> list[tuple[int, Path]]:
    """(step, path) for every checkpoint-N dir, ascending."""
    out = []
    for p in d.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--base", default="unsloth/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--n-prompts", type=int, default=8,
                    help="prompts per checkpoint. 8 gives a closure rate with a "
                         "~+/-0.17 binomial SE at p=0.5 — enough to see a "
                         "collapse, not enough to resolve 7/8 from 8/8.")
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every Nth checkpoint. 2 halves the runtime "
                         "and still localises a collapse to within 10 steps; "
                         "use 1 once you know roughly where it is.")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    import prompts as sherlock_prompts
    from agent import _resolve_think_block

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = ROOT / ckpt_dir
    ckpts = checkpoint_steps(ckpt_dir)
    final = ckpt_dir / "final_adapter"
    if final.exists():
        ckpts.append((10**6, final))     # sorts last; relabelled "final" below
    if not ckpts:
        sys.exit(f"ERROR: no checkpoint-N dirs in {ckpt_dir}. Was "
                 "save_total_limit set to null?")

    if args.stride > 1:
        # Always keep the last checkpoint: the collapse is known to exist by
        # the end, so dropping it would remove the one certain data point.
        kept = ckpts[::args.stride]
        if ckpts[-1] not in kept:
            kept.append(ckpts[-1])
        ckpts = kept

    openers = OPENERS[:args.n_prompts]

    # Open the incremental log BEFORE any GPU work. Every row is appended and
    # flushed as it is produced, so a crash or session cap at 80% still leaves
    # 80% of the curve on disk. A single json.dump at the end loses everything —
    # which is exactly what happened on 2026-08-06.
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    partial = out_dir / f"dose_curve_{stamp}.partial.jsonl"
    partial_f = partial.open("w", encoding="utf-8")
    print(f"  live log    : {partial}")

    print(f"  checkpoints : {len(ckpts)}")
    print(f"  prompts     : {len(openers)} per checkpoint")
    print(f"  generations : {len(ckpts) * len(openers)}")

    # NATIVE bf16 only — is_bf16_supported() counts Turing emulation.
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb,
        device_map={"": 0}, trust_remote_code=True,
    )

    # Attach the first checkpoint so a PeftModel exists; every subsequent one
    # replaces it under the same name. Constant VRAM: base + exactly one adapter.
    ADAPTER = "cur"
    model = PeftModel.from_pretrained(base, str(ckpts[0][1]), adapter_name=ADAPTER)
    model.eval()

    def measure(label) -> dict:
        ok = trunc = 0
        lens = []
        for o in openers:
            ids = tok.apply_chat_template(
                [{"role": "system", "content": sherlock_prompts.INITIATOR_SYSTEM_THINKING},
                 {"role": "user", "content": o}],
                add_generation_prompt=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, do_sample=True,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            n = out.shape[-1] - ids.shape[-1]
            txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=False)
            think, _ = _resolve_think_block(txt, {})
            ok += bool(think and think.strip())
            trunc += n >= args.max_new_tokens
            lens.append(n)
        row = {"label": label, "closure": ok, "n": len(openers),
               "closure_rate": ok / len(openers), "truncated": trunc,
               "mean_tokens": sum(lens) // len(lens)}
        # Append + flush + fsync: survives the process being killed outright,
        # not merely exiting cleanly.
        partial_f.write(json.dumps(row) + "\n")
        partial_f.flush()
        os.fsync(partial_f.fileno())
        return row

    rows = []
    # Base FIRST: it is the control, and if it is not ~1.0 the instrument is
    # broken and every downstream number is meaningless.
    with model.disable_adapter():
        r = measure("base")
        rows.append(r)
        print(f"\n  {'base':<12} closure {r['closure']}/{r['n']}  "
              f"trunc {r['truncated']}  mean {r['mean_tokens']}")
    if rows[0]["closure_rate"] < 0.75:
        print("\n  WARNING: base closure is low. The instrument, not the "
              "adapters, is the first thing to check — every row below "
              "inherits this.")

    for i, (step, path) in enumerate(ckpts):
        if i > 0:
            # Swap in place. delete BEFORE load so peak memory is base + 1,
            # never base + 2.
            model.delete_adapter(ADAPTER)
            torch.cuda.empty_cache()
            model.load_adapter(str(path), adapter_name=ADAPTER)
        model.set_adapter(ADAPTER)
        label = "final" if step == 10**6 else f"step-{step}"
        r = measure(label)
        r["step"] = None if step == 10**6 else step
        rows.append(r)
        print(f"  {label:<12} closure {r['closure']}/{r['n']}  "
              f"trunc {r['truncated']}  mean {r['mean_tokens']}", flush=True)

    partial_f.close()
    out = out_dir / f"dose_curve_{stamp}.json"
    out.write_text(json.dumps({
        "checkpoint_dir": str(ckpt_dir), "base": args.base,
        "n_prompts": len(openers), "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "rows": rows,
    }, indent=2))

    # --- where did it break? -------------------------------------------------
    graded = [r for r in rows if r.get("step") is not None]
    collapse = next((r for r in graded if r["closure_rate"] < 0.5), None)
    print(f"\n{'='*66}")
    if collapse:
        prior = [r for r in graded if r["step"] < collapse["step"]
                 and r["closure_rate"] >= 0.75]
        last_good = prior[-1]["step"] if prior else None
        print(f"  COLLAPSE at step {collapse['step']} "
              f"(closure {collapse['closure']}/{collapse['n']})")
        if last_good is not None:
            print(f"  Last healthy checkpoint: step {last_good}")
            print(f"  -> A usable dose window exists: train to ~{last_good} steps.")
            print("     Whether that dose ALSO produces a behavioural effect is a "
                  "separate question — run the eval gates on that checkpoint.")
        else:
            print("  No healthy checkpoint before the collapse. The format breaks "
                  "before any plausible effect dose;")
            print("  rehearsal data or a different method is needed, not a "
                  "smaller learning rate.")
    else:
        print("  NO COLLAPSE observed across the retained checkpoints.")
        print("  If the final adapter is nonetheless broken, the cause is not "
              "dose — check the final save path.")
    print(f"{'='*66}")
    print(f"  wrote {out}")
    print(f"  (live log kept at {partial} — the two agree unless the run died)\n")


if __name__ == "__main__":
    main()
