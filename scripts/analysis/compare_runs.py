#!/usr/bin/env python3
"""Compare conversation runs on degeneracy and the commitment-gap levels.

Built for the 2026-07-26 shakedown, where three prompt configurations were run
over identical seeds. Matched seeds are what make the comparison meaningful:
the only thing differing between runs is the intervention.

    python scripts/analysis/compare_runs.py \
        results/pilot/shakedown_20260726 \
        results/pilot/shakedown_20260726_antiecho \
        results/pilot/shakedown_20260726_reminder

Reports per run:
  degeneracy rate      -- the Gate 2 criterion (<=20%)
  unique-reply ratio   -- distinct replies / usable turns
  mirrored pairs       -- consecutive identical replies, the raw failure
  t_think fire rates   -- directed (headline) vs topic (legacy baseline)
  censoring            -- how often no public accusation ever arrives
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE_MAX_DEGENERATE = 0.20


def load_run(run_dir: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    conv_files = sorted(glob.glob(str(run_dir / "conversations_*.jsonl")))
    turn_files = sorted(glob.glob(str(run_dir / "turns_*.jsonl")))
    if not conv_files:
        raise SystemExit(f"no conversations_*.jsonl in {run_dir}")
    convs = [json.loads(line) for f in conv_files for line in open(f) if line.strip()]
    turns: dict[str, list[dict]] = collections.defaultdict(list)
    for f in turn_files:
        for line in open(f):
            if line.strip():
                rec = json.loads(line)
                turns[rec["conv_id"]].append(rec)
    for v in turns.values():
        v.sort(key=lambda t: t["turn_idx"])
    return convs, turns


def mirrored_pairs(turn_list: list[dict]) -> tuple[int, int]:
    usable = [t for t in turn_list
              if t.get("parse_mode") not in ("api_error", "parse_failed")]
    replies = [(t["reply"] or "").strip().lower() for t in usable]
    pairs = sum(1 for a, b in zip(replies, replies[1:]) if a and a == b)
    return pairs, max(len(replies) - 1, 0)


def summarise(name: str, convs: list[dict], turns: dict[str, list[dict]]) -> dict:
    n = len(convs)
    degen = sum(1 for c in convs if c.get("is_degenerate"))
    ratios = [c.get("unique_reply_ratio", 1.0) for c in convs]
    mp = mt = 0
    for c in convs:
        a, b = mirrored_pairs(turns.get(c["conv_id"], []))
        mp += a
        mt += b
    return {
        "name": name,
        "n": n,
        "degenerate": degen,
        "degenerate_rate": degen / n if n else 0.0,
        "mean_unique_ratio": statistics.mean(ratios) if ratios else 0.0,
        "mirrored_pairs": mp,
        "adjacent_pairs": mt,
        "mirror_rate": mp / mt if mt else 0.0,
        "mean_turns": statistics.mean([c["n_turns"] for c in convs]) if n else 0.0,
        "t_think_fired": sum(1 for c in convs if c.get("t_think_07") is not None),
        "t_topic_fired": sum(1 for c in convs if c.get("t_think_topic") is not None),
        "accusations": sum(1 for c in convs if c.get("t_public") is not None),
        "censored": sum(1 for c in convs if c.get("t_public") is None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()

    rows = []
    for d in args.run_dirs:
        path = Path(d) if Path(d).is_absolute() else ROOT / d
        convs, turns = load_run(path)
        rows.append(summarise(path.name, convs, turns))

    w = max(len(r["name"]) for r in rows) + 2
    print(f"\n{'='*(w+68)}")
    print("  Conversation run comparison — matched seeds")
    print(f"{'='*(w+68)}")
    hdr = (f"  {'run':<{w}}{'n':>3}{'degen':>10}{'uniq':>7}{'mirror':>9}"
           f"{'turns':>7}{'t_think':>9}{'topic':>7}{'accuse':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        degen = "{}/{}".format(r["degenerate"], r["n"])
        think = "{}/{}".format(r["t_think_fired"], r["n"])
        topic = "{}/{}".format(r["t_topic_fired"], r["n"])
        accus = "{}/{}".format(r["accusations"], r["n"])
        print(f"  {r['name']:<{w}}{r['n']:>3}{degen:>10}"
              f"{r['mean_unique_ratio']:>7.2f}"
              f"{r['mirror_rate']:>9.0%}"
              f"{r['mean_turns']:>7.1f}"
              f"{think:>9}{topic:>7}{accus:>8}")

    print(f"\n  Gate 2 (degeneracy <= {GATE_MAX_DEGENERATE:.0%}):")
    for r in rows:
        verdict = "PASS" if r["degenerate_rate"] <= GATE_MAX_DEGENERATE else "FAIL"
        print(f"    {r['name']:<{w}} {r['degenerate_rate']:>5.0%}  {verdict}")

    print("\n  Notes:")
    print("    'topic' is the superseded topic-mention measure, kept as a baseline.")
    print("    Firing on n/n conversations is the saturation this project fixed.")
    print("    'accuse' counts conversations reaching a public accusation; the")
    print("    remainder are right-censored and need survival methods, not means.\n")


if __name__ == "__main__":
    main()
