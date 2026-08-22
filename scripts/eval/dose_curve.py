#!/usr/bin/env python3
"""Measure think-block closure at every training checkpoint — the dose curve.

WHY. On 2026-08-06, n=8 per arm on identical prompts:
    base, no adapter              closure 8/8
    stage-0    (30 steps,  311K)  closure 8/8
    full canon (103 steps, 3.36M) closure 1/8
Something between 30 and 103 steps degrades the reasoning format. Two points
cannot distinguish a threshold from a slope, nor say whether the format survives
at a dose large enough to move a reasoning prior. This script scores every
checkpoint from one run and reports the degradation with confidence intervals.

METHOD. Loads the base model ONCE, then attaches exactly ONE checkpoint at a
time (load_adapter -> measure -> delete_adapter). Every arm is measured against
identical base weights in a single session, which is the validity requirement;
reloading the base per checkpoint would let fragmentation and load-order
differences leak into a curve whose whole purpose is isolating one variable.

Holding all checkpoints resident at once ALSO satisfies that requirement but
does not fit: 22 adapters x ~161MB is ~3.5GB on top of a 4.5GB 4-bit base, and
it OOM'd on a 14.6GB T4 on 2026-08-06. One-at-a-time gives the same guarantee
at constant memory.

PERSISTENCE. The results JSON and the incremental .partial.jsonl are pushed
off-machine as they are written — HF Hub upload after every row, plus an
optional --push-results git commit+push of the small JSON. On 2026-08-07 the
entire results file was lost when the interactive Kaggle session ended and
/kaggle/working was wiped; this is the third occurrence of that failure class.
A results file that exists only on the machine producing it is a bet, not a
result.

RUN THIS IN A FRESH KERNEL. Any model still resident from an earlier cell holds
several GB and will OOM this script before it starts.

    python scripts/eval/dose_curve.py \
        --checkpoint-dir outputs/kaggle_t4_dosecurve_seed42 \
        --n-prompts 8 --seed 42 \
        --hf-results-repo utsvsngh/sherlock-dosecurve-results --push-results

Writes results/analysis/dose_curve_<timestamp>.json plus a printed table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "conversation"))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import hf_persist  # noqa: E402 — needs the path insert above

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


# ---------------------------------------------------------------------------
# Statistics — Wilson intervals and a dependency-free Fisher exact test.
# ---------------------------------------------------------------------------
# Neither needs scipy (not in requirements.txt), so the eval keeps its light
# footprint on Kaggle. Both are exact/closed-form, not approximations.

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion.

    The right tool at n=8: the normal approximation gives [1.0, 1.0] for 8/8
    (zero width, obviously wrong) and can dip below 0 for small k. Wilson never
    leaves [0, 1] and stays informative at the extremes, which is exactly the
    regime this curve lives in.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _hypergeom_pmf(x: int, row1: int, col1: int, n: int) -> float:
    col2 = n - col1
    return (math.comb(col1, x) * math.comb(col2, row1 - x)) / math.comb(n, row1)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]].

    Sums the hypergeometric probability of every table with the same margins
    whose probability is <= that of the observed table. Verified against the
    textbook lady-tasting-tea value: [[3,1],[1,3]] -> 0.4857.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1 = a + b
    col1 = a + c
    p_obs = _hypergeom_pmf(a, row1, col1, n)
    lo = max(0, row1 - (n - col1))
    hi = min(row1, col1)
    total = 0.0
    for x in range(lo, hi + 1):
        px = _hypergeom_pmf(x, row1, col1, n)
        if px <= p_obs * (1 + 1e-9):
            total += px
    return min(1.0, total)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[pos]


def _prompt_blocked_summary(early: list[dict], late: list[dict],
                            seed: int = 20260822,
                            n_boot: int = 5000) -> dict | None:
    """Conditional paired summary when prompt-level outcomes are available.

    Resampling prompts handles reuse of the same prompt across checkpoints. It
    does not turn one training trajectory into replicated training evidence.
    """
    rows = early + late
    if not rows or any("prompt_outcomes" not in row for row in rows):
        return None
    prompt_ids = [p["prompt_id"] for p in rows[0]["prompt_outcomes"]]
    if any([p["prompt_id"] for p in row["prompt_outcomes"]] != prompt_ids
           for row in rows):
        return None

    def rates(group: list[dict]) -> dict[str, float]:
        values = {pid: [] for pid in prompt_ids}
        for row in group:
            for outcome in row["prompt_outcomes"]:
                values[outcome["prompt_id"]].append(float(outcome["closed"]))
        return {pid: sum(v) / len(v) for pid, v in values.items()}

    e, l = rates(early), rates(late)
    diffs = {pid: l[pid] - e[pid] for pid in prompt_ids}
    point = sum(diffs.values()) / len(diffs)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [prompt_ids[rng.randrange(len(prompt_ids))]
                  for _ in prompt_ids]
        draws.append(sum(diffs[p] for p in sample) / len(sample))
    return {
        "n_prompts": len(prompt_ids),
        "late_minus_early_rate": point,
        "prompt_bootstrap_95": [_percentile(draws, 0.025),
                                _percentile(draws, 0.975)],
        "per_prompt_late_minus_early": diffs,
        "interpretation": ("conditional on this fixed training trajectory and "
                           "prompt bank; not a training-seed replication or a "
                           "causal test of dose"),
    }


def analyze_rows(rows: list[dict], early_max: int = 35, late_min: int = 45,
                 z: float = 1.96) -> dict:
    """Descriptive checkpoint trajectory plus prompt-blocked future analysis.

    Historical artifacts contain checkpoint totals only. Pooling those totals
    into a binomial interval or Fisher test treats repeated prompts and adjacent
    checkpoints as independent, so the former inferential output is withdrawn.
    """
    graded = [r for r in rows if r.get("step") is not None]
    per = []
    for r in sorted(graded, key=lambda x: x["step"]):
        per.append({
            "step": r["step"], "closure": r["closure"], "n": r["n"],
            "rate": r["closure"] / r["n"] if r["n"] else 0.0,
            "interval": None,
            "warning": ("fixed heterogeneous prompt bank; checkpoint rate is "
                        "descriptive, not an iid binomial estimate"),
        })

    early = [r for r in graded if r["step"] <= early_max]
    late = [r for r in graded if r["step"] >= late_min]
    descriptive = None
    prompt_blocked = None
    if early and late:
        e_ok = sum(r["closure"] for r in early)
        e_n = sum(r["n"] for r in early)
        l_ok = sum(r["closure"] for r in late)
        l_n = sum(r["n"] for r in late)
        descriptive = {
            "early_max": early_max, "late_min": late_min,
            "early_ok": e_ok, "early_n": e_n,
            "early_rate": e_ok / e_n if e_n else 0.0,
            "late_ok": l_ok, "late_n": l_n,
            "late_rate": l_ok / l_n if l_n else 0.0,
            "late_minus_early_rate": ((l_ok / l_n) - (e_ok / e_n)
                                      if e_n and l_n else float("nan")),
            "warning": ("descriptive pooled counts only; no binomial interval "
                        "or Fisher test because prompts and checkpoints repeat"),
        }
        prompt_blocked = _prompt_blocked_summary(early, late)
    return {
        "per_checkpoint": per,
        "descriptive_early_late": descriptive,
        "prompt_blocked": prompt_blocked,
        "inference_status": (
            "prompt-blocked conditional interval available; training-seed "
            "replication still required for a causal mechanism claim"
            if prompt_blocked else
            "historical aggregate only: inferential early/late comparison withdrawn"
        ),
    }


def print_analysis(analysis: dict) -> None:
    """Print the descriptive table and corrected inferential status."""
    per = analysis["per_checkpoint"]
    descriptive = analysis["descriptive_early_late"]
    print(f"\n{'='*72}")
    print("  THINK-BLOCK CLOSURE BY CHECKPOINT  (DESCRIPTIVE)")
    print(f"{'='*72}")
    print(f"  {'step':>6}  {'closure':>9}  {'rate':>5}")
    for r in per:
        print(f"  {r['step']:>6}  {r['closure']:>4}/{r['n']:<4}  "
              f"{r['rate']:>5.2f}")

    print(f"\n{'-'*72}")
    if not descriptive:
        print("  Not enough checkpoints on both sides of the split to pool. Report "
              "the per-checkpoint intervals above and collect more before "
              "claiming a difference.")
        print(f"{'='*72}")
        return

    print(f"  DESCRIPTIVE early (step <= {descriptive['early_max']}) vs "
          f"late (step >= {descriptive['late_min']})")
    print(f"    early : {descriptive['early_ok']}/{descriptive['early_n']}  "
          f"= {descriptive['early_rate']:.2f}")
    print(f"    late  : {descriptive['late_ok']}/{descriptive['late_n']}  "
          f"= {descriptive['late_rate']:.2f}")
    print("    NO p-value or pooled binomial CI: repeated prompts and adjacent "
          "checkpoints are dependent.")

    print(f"\n{'-'*72}")
    print("  CONCLUSION")
    print("  On this recorded training trajectory, closure was generally lower at "
          "later checkpoints. The historical aggregate cannot support a p-value, "
          "a safe threshold, or a causal claim about optimizer dose.")
    if analysis["prompt_blocked"]:
        pb = analysis["prompt_blocked"]
        print(f"  Prompt-blocked late-minus-early difference: "
              f"{pb['late_minus_early_rate']:+.3f} "
              f"[{pb['prompt_bootstrap_95'][0]:+.3f}, "
              f"{pb['prompt_bootstrap_95'][1]:+.3f}], conditional on one "
              "training trajectory.")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Persistence of results — off-machine as they are written.
# ---------------------------------------------------------------------------

def git_push_results(files: list[Path], message: str) -> bool:
    """git add + commit + push the small result files. Never raises.

    The owner explicitly asked for a git-push option for results. Only the small
    JSON / JSONL files are staged by exact path — never `git add .` on a dirty
    tree (house rule). Runs against whatever origin the driver notebook cloned
    (on Kaggle, a tokened URL held only in the remote config, never in argv or a
    committed file).
    """
    rels = []
    for f in files:
        if f.exists():
            try:
                rels.append(str(f.relative_to(ROOT)))
            except ValueError:
                rels.append(str(f))
    if not rels:
        print("  [push-results] nothing to push — no result files exist yet.",
              flush=True)
        return False
    try:
        subprocess.run(["git", "-C", str(ROOT), "add", *rels], check=True)
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "commit", "-m", message],
            capture_output=True, text=True)
        blob = commit.stdout + commit.stderr
        if commit.returncode != 0 and "nothing to commit" not in blob:
            print(f"  [push-results] commit failed: {blob.strip()}", flush=True)
            return False
        push = subprocess.run(["git", "-C", str(ROOT), "push"],
                              capture_output=True, text=True)
        if push.returncode != 0:
            print(f"  [push-results] push FAILED: "
                  f"{(push.stdout + push.stderr).strip()}", flush=True)
            return False
        print(f"  [push-results] pushed {', '.join(rels)}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001 — persistence must not kill the eval
        print(f"  [push-results] git error ({type(e).__name__}: {e})", flush=True)
        return False


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
    ap.add_argument("--seed", type=int, default=42,
                    help="Seeds torch/transformers before generation. The eval "
                         "was previously UNSEEDED, so identical commands gave "
                         "different closure numbers. Logged into the output JSON.")
    ap.add_argument("--out-dir", default="results/analysis")
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every Nth checkpoint. 2 halves the runtime "
                         "and still localises a collapse to within 10 steps; "
                         "use 1 once you know roughly where it is.")
    ap.add_argument("--early-max", type=int, default=35,
                    help="pooled 'early' arm = checkpoints with step <= this.")
    ap.add_argument("--late-min", type=int, default=45,
                    help="pooled 'late' arm = checkpoints with step >= this.")
    ap.add_argument("--hf-results-repo", default=None,
                    help="HF dataset repo id for the results JSON + partial "
                         "jsonl. Env HF_RESULTS_REPO overrides. Uploaded as each "
                         "row is written so a wiped session keeps the partial.")
    ap.add_argument("--push-results", action="store_true",
                    help="Also git-commit and push the small result JSON to the "
                         "repo. Owner-requested; opt-in.")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed,
    )

    import prompts as sherlock_prompts
    from agent import _resolve_think_block

    # Seed everything BEFORE any generation so identical commands reproduce.
    set_seed(args.seed)
    SYSTEM_PROMPT_NAME = "INITIATOR_SYSTEM_THINKING"

    # Results persistence config resolved up front so a misconfiguration is
    # visible before 5 hours of GPU work, not after.
    hf_results_repo = os.environ.get("HF_RESULTS_REPO") or args.hf_results_repo
    hf_token = hf_persist.find_hf_token() if hf_results_repo else None
    if hf_results_repo and not hf_token:
        print("  WARNING: --hf-results-repo set but NO HF token found. The "
              "results will NOT be persisted to the Hub — set env HF_TOKEN or "
              "the Kaggle Secret HF_TOKEN.", flush=True)

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

    def persist_partial() -> None:
        """Push the partial jsonl off-machine after every row. Cheap (a few KB)
        and it is the artifact that survives a mid-run session kill."""
        if hf_results_repo and hf_token:
            hf_persist.upload_path(
                partial, hf_results_repo, hf_token,
                path_in_repo=partial.name, repo_type="dataset",
                private=True, label=partial.name)

    print(f"  checkpoints : {len(ckpts)}")
    print(f"  prompts     : {len(openers)} per checkpoint")
    print(f"  generations : {len(ckpts) * len(openers)}")
    print(f"  seed        : {args.seed}")
    print(f"  sampling    : temp={args.temperature} "
          f"max_new_tokens={args.max_new_tokens} system={SYSTEM_PROMPT_NAME}")
    if hf_results_repo:
        print(f"  results repo: {hf_results_repo} "
              f"({'token OK' if hf_token else 'NO TOKEN'})")

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
        outcomes = []
        for prompt_idx, o in enumerate(openers):
            # Common random numbers across checkpoints: a prompt gets the same
            # generation seed in every arm. This is logged per observation.
            generation_seed = args.seed * 1_000_003 + prompt_idx
            set_seed(generation_seed)
            ids = tok.apply_chat_template(
                [{"role": "system", "content": getattr(sherlock_prompts, SYSTEM_PROMPT_NAME)},
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
            closed = bool(think and think.strip())
            was_truncated = n >= args.max_new_tokens
            ok += closed
            trunc += was_truncated
            lens.append(n)
            outcomes.append({
                "prompt_id": f"opener-{prompt_idx:02d}",
                "prompt_sha256": hashlib.sha256(o.encode()).hexdigest(),
                "closed": closed,
                "truncated": was_truncated,
                "n_tokens": n,
                "generation_seed": generation_seed,
                "output_sha256": hashlib.sha256(txt.encode()).hexdigest(),
            })
        row = {"label": label, "closure": ok, "n": len(openers),
               "closure_rate": ok / len(openers), "truncated": trunc,
               "mean_tokens": sum(lens) // len(lens),
               "prompt_outcomes": outcomes}
        # Append + flush + fsync: survives the process being killed outright,
        # not merely exiting cleanly. Then push off-machine.
        partial_f.write(json.dumps(row) + "\n")
        partial_f.flush()
        os.fsync(partial_f.fileno())
        persist_partial()
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

    analysis = analyze_rows(rows, early_max=args.early_max, late_min=args.late_min)
    out = out_dir / f"dose_curve_{stamp}.json"
    out.write_text(json.dumps({
        "checkpoint_dir": str(ckpt_dir), "base": args.base,
        "n_prompts": len(openers), "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT_NAME,
        "rows": rows, "analysis": analysis,
    }, indent=2))

    print_analysis(analysis)
    print(f"\n  wrote {out}")
    print(f"  (live log kept at {partial} — the two agree unless the run died)")

    # Persist the final JSON off-machine immediately: HF Hub, then optional git.
    persisted = False
    if hf_results_repo and hf_token:
        persisted = hf_persist.upload_path(
            out, hf_results_repo, hf_token,
            path_in_repo=out.name, repo_type="dataset",
            private=True, label=out.name)
    if args.push_results:
        persisted = git_push_results(
            [out, partial],
            f"results: dose curve {stamp} (Wilson CIs + early/late Fisher)") or persisted

    # HONEST close. Do NOT print "safe to close" when nothing left the machine —
    # that false reassurance is exactly what preceded the 2026-08-08 loss.
    if persisted:
        print("\n  results persisted off-machine — safe to close the session.")
    else:
        print("\n  " + "!" * 68)
        print("  RESULTS ARE LOCAL ONLY — NOT persisted off-machine.")
        print(f"  {out}")
        print(f"  {partial}")
        print("  On ephemeral compute (Kaggle/Colab) these are LOST at session "
              "end. Download them now, or set HF_TOKEN and re-upload before "
              "closing.")
        print("  " + "!" * 68)


def checkpoint_steps(d: Path) -> list[tuple[int, Path]]:
    """(step, path) for every checkpoint-N dir, ascending."""
    out = []
    for p in d.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return sorted(out)


if __name__ == "__main__":
    main()
