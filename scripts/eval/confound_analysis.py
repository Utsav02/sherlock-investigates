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

  Four descriptive contrasts:
  STEPS effect at low breadth : a vs b   (pilot early vs late)
  STEPS effect at high breadth: c vs d   (canon early vs late)   -- known sig.
  BREADTH effect at low steps : a vs c   (early: pilot vs canon)
  BREADTH effect at high steps: b vs d   (late:  pilot vs canon)

Historical limitation:
  Each cell pools the same prompts over adjacent checkpoints from one training
  trajectory. Those are repeated, correlated observations. The former Fisher
  tests and mechanism verdict are withdrawn; this script now reports effect
  patterns only. Independent training seeds are required for causal attribution.

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
    rate1 = ok1 / n1 if n1 else 0.0
    rate2 = ok2 / n2 if n2 else 0.0
    return {
        "name": name,
        "a": {"ok": ok1, "n": n1, "rate": rate1},
        "b": {"ok": ok2, "n": n2, "rate": rate2},
        "b_minus_a_rate": rate2 - rate1,
        "warning": ("descriptive only; pooled rows repeat prompts and adjacent "
                    "checkpoints from one training trajectory"),
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

    observed_steps_decline = (
        steps_pilot["b"]["rate"] < steps_pilot["a"]["rate"])
    observed_breadth_gap = (
        breadth_early["b"]["rate"] < breadth_early["a"]["rate"] or
        breadth_late["b"]["rate"] < breadth_late["a"]["rate"])
    verdict = (
        "NO CAUSAL MECHANISM VERDICT. These recorded trajectories can describe "
        "whether closure was lower at later checkpoints or under broader data, "
        "but repeated prompts, serially related checkpoints, and one training "
        "trajectory per condition make the old Fisher tests invalid. Replicate "
        "independent training seeds and retain prompt-level outcomes before "
        "attributing the pattern to steps or unique-token breadth."
    )

    return {
        "early_max": early_max, "late_min": late_min,
        "contrasts": {
            "steps_at_low_breadth": steps_pilot,
            "steps_at_high_breadth": steps_canon,
            "breadth_at_low_steps": breadth_early,
            "breadth_at_high_steps": breadth_late,
        },
        "observed_steps_decline": observed_steps_decline,
        "observed_breadth_gap": observed_breadth_gap,
        "inference_status": "historical inferential verdict withdrawn",
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
              f"vs {bb['ok']:>3}/{bb['n']:<3} = {bb['rate']:.2f}   "
              f"delta={c['b_minus_a_rate']:+.2f}")
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
