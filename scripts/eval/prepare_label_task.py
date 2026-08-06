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

sys.path.insert(0, str(ROOT / "scripts" / "conversation"))
import conv_logging  # noqa: E402


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
    ap.add_argument("--stratified", action="store_true",
                    help="Rare-positive design. Random sampling cannot measure a "
                         "detector whose positive class is ~0.5%% of sentences: a "
                         "300-sentence sample yielded 1-2 positives. Instead take "
                         "EVERY detector fire (exact precision, no sampling error) "
                         "plus a sample of sentences that mention AI but do not "
                         "fire (where false negatives must live). Sentences with no "
                         "AI term at all are skipped - an assumption, not a proof: "
                         "a conclusion phrased entirely outside the keyword list "
                         "would hide there.")
    ap.add_argument("--recall-sample", type=int, default=150)
    ap.add_argument("--strata-out", default="results/analysis/think_stance_strata.json",
                    help="Stratum membership, written SEPARATELY. It must never "
                         "reach an annotator: knowing the stratum reveals the "
                         "detector's prediction.")
    args = ap.parse_args()

    patterns = [str(ROOT / p) for p in args.turns_glob]
    sentences = collect_sentences(patterns)
    random.Random(args.seed).shuffle(sentences)   # same order as the GUI
    pool = len(sentences)
    strata = {}
    if args.stratified:
        fires = [x for x in sentences
                 if conv_logging._think_block_suspicious(x["text"])]
        mentions = [x for x in sentences
                    if conv_logging._think_block_mentions_ai(x["text"])
                    and not conv_logging._think_block_suspicious(x["text"])]
        recall = mentions[:args.recall_sample]
        for x in fires:
            strata[x["id"]] = "fire"
        for x in recall:
            strata[x["id"]] = "mention_only"
        sentences = fires + recall
        # Reshuffle so stratum is not inferable from position in the file.
        random.Random(args.seed + 1).shuffle(sentences)
        print(f"  precision stratum (all detector fires): {len(fires)}")
        print(f"  recall stratum (mentions AI, no fire) : {len(recall)} "
              f"sampled from {len(mentions)}")
    elif args.limit:
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

    if strata:
        sp = ROOT / args.strata_out
        sp.write_text(json.dumps(strata, indent=1), encoding="utf-8")
        print(f"  strata (NOT for annotators) -> {sp}")
    print(f"{len(sentences)} sentences (from a pool of {pool}) -> {out}")


if __name__ == "__main__":
    main()
