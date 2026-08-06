#!/usr/bin/env python3
"""Emit the think-stance labelling task for independent annotators.

Deliberately minimal: id, sentence, and the neighbouring sentences for pronoun
resolution. NOTHING about how the detector works — no regex, no vetoes, no
predictions. An annotator who can see the pattern is labelling with the answer
key, and the disagreements (the whole point of the exercise) would vanish.

    python scripts/eval/prepare_label_task.py
    -> results/analysis/think_stance_task.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from build_think_label_tool import collect_sentences  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--turns-glob", nargs="+",
                    default=["results/pilot/**/turns_*.jsonl"])
    ap.add_argument("--out", default="results/analysis/think_stance_task.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=300,
                    help="random sample size (0 = all). The pool grows with "
                         "every run; a sample keeps the annotation task bounded "
                         "and is drawn AFTER the shuffle, so it stays random.")
    args = ap.parse_args()

    patterns = [str(ROOT / p) for p in args.turns_glob]
    sentences = collect_sentences(patterns)
    random.Random(args.seed).shuffle(sentences)   # same order as the GUI
    pool = len(sentences)
    if args.limit:
        sentences = sentences[:args.limit]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps({
                "id": s["id"],
                "text": s["text"],
                "prev": s["prev"],
                "next": s["next"],
            }, ensure_ascii=False) + "\n")

    print(f"{len(sentences)} sentences (sampled from {pool}) -> {out}")


if __name__ == "__main__":
    main()
