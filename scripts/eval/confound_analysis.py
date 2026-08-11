#!/usr/bin/env python3
"""Separate STEPS from UNIQUE-TOKEN BREADTH in the think-block format collapse.

Overlays two dose-curve JSONs produced by dose_curve.py:
  - full canon  (3.36M unique tokens, ~103 steps)   — already collected
  - pilot @ 103 (311K unique, re-read 11x, ~103 steps) — configs/kaggle_t4_confound_pilot103.yaml

Both are ~103-step cosine runs on identical model/LoRA/optimizer settings; the
ONLY difference is corpus breadth. That makes a clean 2x2:

                 early (<=35 steps)   late (>=45 steps)
  pilot  (311K)  [a]                  [b]
  canon  (3.36M) [c]                  [d]

Four contrasts, each Fisher exact + Wilson CIs:
  STEPS effect at low breadth : a vs b   (pilot early vs late)
  STEPS effect at high breadth: c vs d   (canon early vs late)   -- known sig.
  BREADTH effect at low steps : a vs c   (early: pilot vs canon)
  BREADTH effect at high steps: b vs d   (late:  pilot vs canon)

Verdict:
  breadth significant, steps-at-low-breadth NOT  -> BREADTH drives it
                                                    (rehearsal mandatory)
  steps-at-low-breadth significant               -> WEIGHT MOVEMENT contributes
                                                    (low-LR / low-rank worth trying)

Usage:
    python scripts/eval/confound_analysis.py \
        --fullcanon results/analysis/dose_curve_20260808_204827.json \
        --pilot     results/analysis/dose_curve_<pilot103_stamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from dose_curve import fisher_exact_two_sided, wilson_interval

SIG = 0.05


def _pool(rows: list[dict], lo: int | None = None, hi: int | None = None) -> tuple[int, int]:
    """Sum (closure, n) over graded checkpoints whose step is in [lo, hi]."""
    ok = n = 0
    for r in rows:
        step = r.get("step")
        if step is None:
            continue
        if lo is not None and step < lo:
            continue
        if hi is not None and step > hi:
            continue
        ok += r["closure"]
        n += r["n"]
    return ok, n


def _contrast(name: str, ok1: int, n1: int, ok2: int, n2: int) -> dict:
    return {
        "name": name,
        "a": {"ok": ok1, "n": n1, "rate": ok1 / n1 if n1 else 0.0,
              "wilson": list(wilson_interval(ok1, n1))},
        "b": {"ok": ok2, "n": n2, "rate": ok2 / n2 if n2 else 0.0,
              "wilson": list(wilson_interval(ok2, n2))},
        "fisher_p": fisher_exact_two_sided(ok1, n1 - ok1, ok2, n2 - ok2)
        if n1 and n2 else 1.0,
    }


def analyze_confound(fullcanon_rows: list[dict], pilot_rows: list[dict],
                     early_max: int = 35, late_min: int = 45) -> dict:
    """Pure 2x2 analysis. Returns the four contrasts and a verdict string."""
    p_e_ok, p_e_n = _pool(pilot_rows, hi=early_max)
    p_l_ok, p_l_n = _pool(pilot_rows, lo=late_min)
    c_e_ok, c_e_n = _pool(fullcanon_rows, hi=early_max)
    c_l_ok, c_l_n = _pool(fullcanon_rows, lo=late_min)

    steps_pilot = _contrast("STEPS @ low breadth (pilot early vs late)",
                            p_e_ok, p_e_n, p_l_ok, p_l_n)
    steps_canon = _contrast("STEPS @ high breadth (canon early vs late)",
                            c_e_ok, c_e_n, c_l_ok, c_l_n)
    breadth_early = _contrast("BREADTH @ low steps (early: pilot vs canon)",
                             p_e_ok, p_e_n, c_e_ok, c_e_n)
    breadth_late = _contrast("BREADTH @ high steps (late: pilot vs canon)",
                            p_l_ok, p_l_n, c_l_ok, c_l_n)

    # Direction matters, not just significance. "steps hurt" = pilot closure
    # FALLS from early to late. "breadth hurts" = pilot (fewer unique) closes
    # BETTER than canon.
    steps_hurt = (steps_pilot["fisher_p"] < SIG
                  and steps_pilot["a"]["rate"] > steps_pilot["b"]["rate"])
    breadth_hurt = (
        (breadth_early["fisher_p"] < SIG
         and breadth_early["a"]["rate"] > breadth_early["b"]["rate"])
        or (breadth_late["fisher_p"] < SIG
            and breadth_late["a"]["rate"] > breadth_late["b"]["rate"]))

    if breadth_hurt and not steps_hurt:
        verdict = ("UNIQUE-TOKEN BREADTH drives the collapse. Steps alone, at low "
                   "breadth, do not break closure (pilot stays healthy across "
                   "steps) while canon decays. The effect dose (~1M+ unique "
                   "tokens) cannot be reached without the breadth that breaks the "
                   "format -> low-LR / low-rank / early-stop CANNOT open a window; "
                   "REHEARSAL (base-model-generated think blocks) is the only "
                   "mitigation with a mechanism.")
    elif steps_hurt and not breadth_hurt:
        verdict = ("OPTIMIZER STEPS / weight movement drive the collapse, "
                   "independent of breadth (pilot decays with steps even at 1/11th "
                   "the unique tokens). Fewer/smaller updates should preserve the "
                   "format -> low-LR, low-rank, and early-stop are worth trying "
                   "before the more involved rehearsal pipeline.")
    elif steps_hurt and breadth_hurt:
        verdict = ("BOTH contribute: closure falls with steps even at low breadth, "
                   "AND at matched steps the broader corpus is worse. Rehearsal is "
                   "the robust mitigation; low-LR/rank may buy headroom but will "
                   "not fully protect the format at an effect dose.")
    else:
        verdict = ("INCONCLUSIVE at this n — no contrast reaches p<0.05 in the "
                   "damaging direction. Increase --n-prompts on both curves or add "
                   "checkpoints before drawing a mechanism.")

    return {
        "early_max": early_max, "late_min": late_min,
        "contrasts": {
            "steps_at_low_breadth": steps_pilot,
            "steps_at_high_breadth": steps_canon,
            "breadth_at_low_steps": breadth_early,
            "breadth_at_high_steps": breadth_late,
        },
        "steps_hurt": steps_hurt, "breadth_hurt": breadth_hurt,
        "verdict": verdict,
    }


def print_confound(a: dict) -> None:
    print(f"\n{'='*72}")
    print("  CONFOUND SEPARATOR — steps vs unique-token breadth")
    print(f"  (early = step <= {a['early_max']}, late = step >= {a['late_min']})")
    print(f"{'='*72}")
    for c in a["contrasts"].values():
        aa, bb = c["a"], c["b"]
        print(f"\n  {c['name']}")
        print(f"    {aa['ok']:>3}/{aa['n']:<3} = {aa['rate']:.2f} "
              f"[{aa['wilson'][0]:.2f}, {aa['wilson'][1]:.2f}]   vs   "
              f"{bb['ok']:>3}/{bb['n']:<3} = {bb['rate']:.2f} "
              f"[{bb['wilson'][0]:.2f}, {bb['wilson'][1]:.2f}]   "
              f"Fisher p={c['fisher_p']:.4g}")
    print(f"\n{'-'*72}")
    print("  VERDICT")
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


def _load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["rows"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fullcanon", required=True,
                    help="dose_curve JSON for the full-canon run")
    ap.add_argument("--pilot", required=True,
                    help="dose_curve JSON for the pilot@103 run")
    ap.add_argument("--early-max", type=int, default=35)
    ap.add_argument("--late-min", type=int, default=45)
    ap.add_argument("--out", default=None,
                    help="optional path to write the analysis JSON")
    args = ap.parse_args()

    fc = Path(args.fullcanon)
    pl = Path(args.pilot)
    for p in (fc, pl):
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"ERROR: no such dose-curve JSON: {p}")

    analysis = analyze_confound(
        _load_rows(fc if fc.is_absolute() else ROOT / fc),
        _load_rows(pl if pl.is_absolute() else ROOT / pl),
        early_max=args.early_max, late_min=args.late_min)
    print_confound(analysis)

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.write_text(json.dumps({
            "fullcanon": str(fc), "pilot": str(pl), "analysis": analysis,
        }, indent=2))
        print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
