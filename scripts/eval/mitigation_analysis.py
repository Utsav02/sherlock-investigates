#!/usr/bin/env python3
"""Overlay closure and perplexity curves to screen candidate checkpoints.

The dual measurement that decides whether rehearsal is needed at all:
  closure curve (dose_curve.py)  -> does the <think> format SURVIVE?  (a)
  effect curve  (effect_curve.py) -> did held-out perplexity move?     (b)

A checkpoint is only a candidate for behavioral evaluation when both thresholds
hold. Perplexity is a proxy, not evidence that Holmes-like reasoning was learned;
the 2026-08-14 behavioral check was null. This screen must not declare a rescue.

  window exists (closure ok AND proxy)    -> CANDIDATE. run behavior checks.
  effect only where closure collapsed     -> COUPLED. rehearsal needed
                                             (low rank cannot decouple them).
  effect never appears at any checkpoint  -> TOO WEAK. low rank learned nothing;
                                             more capacity is needed, which
                                             breaks closure -> rehearsal needed.

Usage:
    python scripts/eval/mitigation_analysis.py \
        --closure results/analysis/dose_curve_<lowrank_stamp>.json \
        --effect  results/analysis/effect_curve_<lowrank_stamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CLOSURE_OK = 0.75    # closure rate at/above which the format is "intact"
EFFECT_MIN = 5.0     # held-out PPL drop % at/above which the effect is "present"
                     # (the pre-registered H1 gate)


def _closure_by_step(closure_rows: list[dict]) -> dict[int, dict]:
    out = {}
    for r in closure_rows:
        step = r.get("step")
        if step is not None:
            out[step] = {"closure_rate": r["closure"] / r["n"],
                         "closure": r["closure"], "n": r["n"]}
    return out


def _effect_by_step(effect_rows: list[dict]) -> dict[int, dict]:
    out = {}
    for r in effect_rows:
        step = r.get("step")
        if step is not None:
            out[step] = {"heldout_ppl": r.get("heldout_ppl"),
                         "drop_pct": r.get("heldout_drop_pct")}
    return out


def analyze_mitigation(closure_rows: list[dict], effect_rows: list[dict],
                       closure_ok: float = CLOSURE_OK,
                       effect_min: float = EFFECT_MIN) -> dict:
    """Pure overlay. Returns per-step joins and a non-inferential screen."""
    cbys = _closure_by_step(closure_rows)
    ebys = _effect_by_step(effect_rows)
    steps = sorted(set(cbys) & set(ebys))

    joined = []
    for s in steps:
        c = cbys[s]["closure_rate"]
        d = ebys[s]["drop_pct"]
        joined.append({
            "step": s, "closure_rate": c, "drop_pct": d,
            "closure_ok": c >= closure_ok,
            "effect_present": d is not None and d >= effect_min,
        })

    window = [j for j in joined if j["closure_ok"] and j["effect_present"]]
    any_effect = [j for j in joined if j["effect_present"]]
    max_drop_intact = max(
        (j["drop_pct"] for j in joined
         if j["closure_ok"] and j["drop_pct"] is not None), default=None)

    if window:
        best = max(window, key=lambda j: j["drop_pct"])
        verdict_key = "CANDIDATE_WINDOW"
        verdict = (
            f"CANDIDATE WINDOW at step {best['step']}: closure "
            f"{best['closure_rate']:.2f} (>= {closure_ok}) AND held-out PPL drop "
            f"{best['drop_pct']:+.1f}% (>= {effect_min}% proxy gate). This only "
            f"selects step-{best['step']} for behavioral evaluation; it does not "
            f"show a reasoning shift, establish rescue, or decide whether "
            f"rehearsal is needed.")
    elif any_effect:
        verdict_key = "PROXY_COUPLED"
        best_effect = max(any_effect, key=lambda j: j["drop_pct"])
        verdict = (
            f"PROXY COUPLED. The perplexity effect appears (max PPL drop "
            f"{best_effect['drop_pct']:+.1f}% at step {best_effect['step']}) but "
            f"ONLY where closure has already collapsed (closure "
            f"{best_effect['closure_rate']:.2f} there); the best drop while "
            f"closure is intact is "
            f"{max_drop_intact if max_drop_intact is not None else float('nan'):+.1f}%, "
            f"below the {effect_min}% gate. Low rank does not decouple format from "
            f"proxy effect. This screen alone cannot prescribe rehearsal.")
    else:
        verdict_key = "NO_PROXY_WINDOW"
        verdict = (
            f"NO PROXY WINDOW. No checkpoint reaches the {effect_min}% "
            f"held-out PPL drop at all (best while closure intact: "
            f"{max_drop_intact if max_drop_intact is not None else float('nan'):+.1f}%). "
            f"Rank 8 preserved closure by learning too little to matter. More "
            f"This screen cannot infer what behavioral intervention is needed.")

    return {
        "closure_ok_threshold": closure_ok, "effect_min_pct": effect_min,
        "joined": joined, "window": window, "verdict_key": verdict_key,
        "verdict": verdict,
    }


def print_mitigation(a: dict) -> None:
    print(f"\n{'='*72}")
    print("  MITIGATION SCREEN — closure vs held-out PPL proxy")
    print(f"  intact = closure >= {a['closure_ok_threshold']}, "
          f"effect = PPL drop >= {a['effect_min_pct']}%")
    print(f"{'='*72}")
    print(f"  {'step':>6}  {'closure':>7}  {'PPL drop':>9}  flags")
    for j in a["joined"]:
        flags = []
        if j["closure_ok"]:
            flags.append("format-intact")
        if j["effect_present"]:
            flags.append("effect")
        d = f"{j['drop_pct']:+.1f}%" if j["drop_pct"] is not None else "   n/a"
        star = " <-- WINDOW" if (j["closure_ok"] and j["effect_present"]) else ""
        print(f"  {j['step']:>6}  {j['closure_rate']:>7.2f}  {d:>9}  "
              f"{', '.join(flags):<28}{star}")
    print(f"\n{'-'*72}")
    print("  SCREEN RESULT")
    for line in _wrap(a["verdict"], 70):
        print(f"  {line}")
    print(f"{'='*72}")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--closure", required=True, help="dose_curve JSON (closure)")
    ap.add_argument("--effect", required=True, help="effect_curve JSON (PPL)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cp = Path(args.closure)
    ep = Path(args.effect)
    cp = cp if cp.is_absolute() else ROOT / cp
    ep = ep if ep.is_absolute() else ROOT / ep
    for p in (cp, ep):
        if not p.exists():
            sys.exit(f"ERROR: no such JSON: {p}")

    analysis = analyze_mitigation(_rows(cp), _rows(ep))
    print_mitigation(analysis)

    if args.out:
        out = Path(args.out)
        out = out if out.is_absolute() else ROOT / out
        out.write_text(json.dumps({
            "closure": str(cp), "effect": str(ep), "analysis": analysis,
        }, indent=2))
        print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
