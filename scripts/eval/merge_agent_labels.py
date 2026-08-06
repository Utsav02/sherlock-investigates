#!/usr/bin/env python3
"""Merge independent annotator passes, measure agreement, surface the clashes.

Workflow this belongs to:
    prepare_label_task.py      -> the task (no detector information in it)
    N independent annotators   -> agent_labels_*.jsonl
    THIS SCRIPT                -> agreement, majority labels, clash list
    build_think_label_tool.py --ids-file  -> adjudicate only the clashes
    score_think_detector.py    -> detector precision/recall vs the merged set

IMPORTANT CAVEAT, carried in the output on purpose:
These annotators are not independent of the detector's author. The detector's
notion of "conclusion" and theirs come from the same place, so agreement between
them measures self-consistency, not correctness. Use this to FIND errors, not to
certify accuracy — do not report the resulting precision as a validation figure.
A human-labelled anchor set is what would upgrade it to evidence.

    python scripts/eval/merge_agent_labels.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "conversation"))

import conv_logging  # noqa: E402

LABELS = ("conclusion", "task_restatement", "neither")
POSITIVE = "conclusion"


def fleiss_kappa(rows: list[list[str]]) -> float:
    """rows[i] = the labels assigned to item i by all raters."""
    n_items = len(rows)
    if not n_items:
        return 0.0
    n_raters = len(rows[0])
    if n_raters < 2:
        return 0.0

    unknown = {l for r in rows for l in r} - set(LABELS)
    if unknown:
        # Silently ignoring these makes every per-item sum zero and returns a
        # plausible-looking negative kappa. Fail loudly instead.
        raise ValueError(f"labels outside the schema: {sorted(unknown)}")

    counts = [collections.Counter(r) for r in rows]
    # P_i: agreement among rater pairs for item i
    p_i = [
        (sum(c[l] ** 2 for l in LABELS) - n_raters) / (n_raters * (n_raters - 1))
        for c in counts
    ]
    p_bar = sum(p_i) / n_items
    # p_j: marginal proportion for each category
    p_j = [sum(c[l] for c in counts) / (n_items * n_raters) for l in LABELS]
    p_e = sum(p * p for p in p_j)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else 1.0


def interpret(k: float) -> str:
    for lo, word in ((0.81, "almost perfect"), (0.61, "substantial"),
                     (0.41, "moderate"), (0.21, "fair"), (0.0, "slight")):
        if k >= lo:
            return word
    return "poor (worse than chance)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="results/analysis/think_stance_task.jsonl")
    ap.add_argument("--labels-glob", default="results/analysis/agent_labels_*.jsonl")
    ap.add_argument("--out-merged", default="data/probes/think_stance_labels_v1.jsonl")
    ap.add_argument("--out-clashes", default="results/analysis/think_stance_clashes.json")
    ap.add_argument("--strata",
                    help="think_stance_strata.json from prepare_label_task.py "
                         "--stratified. Enables exact precision (every fire is "
                         "labelled) and a reweighted recall estimate.")
    ap.add_argument("--recall-population", type=int, default=654,
                    help="size of the mention-but-no-fire population the recall "
                         "stratum was sampled from")
    args = ap.parse_args()

    task = {}
    order = []
    for line in open(ROOT / args.task, encoding="utf-8"):
        rec = json.loads(line)
        task[rec["id"]] = rec
        order.append(rec["id"])

    files = sorted(glob.glob(str(ROOT / args.labels_glob)))
    if len(files) < 2:
        raise SystemExit(f"need >=2 annotator files, found {len(files)}")

    annotators: dict[str, dict[str, str]] = {}
    for f in files:
        name = Path(f).stem.replace("agent_labels_", "")
        got = {}
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("label") in LABELS:
                got[rec["id"]] = rec["label"]
        annotators[name] = got
        missing = len(task) - len(got)
        print(f"  annotator {name}: {len(got)}/{len(task)} labelled"
              + (f"  ({missing} MISSING)" if missing else ""))

    names = sorted(annotators)
    common = [i for i in order if all(i in annotators[n] for n in names)]
    print(f"\n  {len(common)}/{len(task)} sentences labelled by all {len(names)} annotators")
    if not common:
        raise SystemExit("no overlap between annotators")

    rows = [[annotators[n][i] for n in names] for i in common]

    # --- agreement --------------------------------------------------------
    kappa = fleiss_kappa(rows)
    unanimous = sum(1 for r in rows if len(set(r)) == 1)
    print(f"\n{'='*68}")
    print("  Inter-annotator agreement")
    print(f"{'='*68}")
    print(f"  unanimous:      {unanimous}/{len(common)} ({unanimous/len(common):.0%})")
    print(f"  Fleiss' kappa:  {kappa:.3f}  ({interpret(kappa)})")
    print("  pairwise:")
    for a, b in itertools.combinations(names, 2):
        agree = sum(1 for i in common if annotators[a][i] == annotators[b][i])
        print(f"    {a} vs {b}: {agree/len(common):.0%}")

    print("\n  label distribution per annotator:")
    for n in names:
        c = collections.Counter(annotators[n][i] for i in common)
        print(f"    {n}: " + "  ".join(f"{l}={c[l]}" for l in LABELS))

    # --- majority + clashes ----------------------------------------------
    merged, clashes = [], []
    for i, r in zip(common, rows):
        counts = collections.Counter(r)
        top, n_top = counts.most_common(1)[0]
        tied = [l for l, c in counts.items() if c == n_top]
        rec = {
            "id": i,
            "text": task[i]["text"],
            "label": top,
            "votes": {n: annotators[n][i] for n in names},
            "unanimous": len(counts) == 1,
            "tied": len(tied) > 1,
        }
        merged.append(rec)
        if len(counts) > 1:
            clashes.append(rec)

    out_merged = ROOT / args.out_merged
    out_merged.parent.mkdir(parents=True, exist_ok=True)
    with out_merged.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps({
                "id": rec["id"], "text": rec["text"], "label": rec["label"],
                "provenance": "annotator-majority",
                "unanimous": rec["unanimous"], "tied": rec["tied"],
                "votes": rec["votes"],
            }, ensure_ascii=False) + "\n")

    out_clashes = ROOT / args.out_clashes
    out_clashes.write_text(
        json.dumps([c["id"] for c in clashes], indent=1), encoding="utf-8")

    tied = [c for c in clashes if c["tied"]]
    print(f"\n{'='*68}")
    print("  Clashes to adjudicate")
    print(f"{'='*68}")
    print(f"  disagreements:   {len(clashes)}/{len(common)} ({len(clashes)/len(common):.0%})")
    print(f"  three-way ties:  {len(tied)}  (no majority — these need you most)")
    print(f"\n  merged labels -> {out_merged}")
    print(f"  clash ids     -> {out_clashes}")

    # --- where the detector sits relative to the merged set ---------------
    tp = fp = fn = 0
    disagree_with_detector = []
    for rec in merged:
        gold = rec["label"] == POSITIVE
        pred = bool(conv_logging._think_block_suspicious(rec["text"]))
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
            disagree_with_detector.append(("FP", rec))
        elif gold and not pred:
            fn += 1
            disagree_with_detector.append(("FN", rec))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec_ = tp / (tp + fn) if (tp + fn) else 0.0

    print(f"\n{'='*68}")
    print("  Detector vs merged labels — DEBUGGING SIGNAL, NOT VALIDATION")
    print(f"{'='*68}")
    print(f"  precision {prec:.3f}   recall {rec_:.3f}   tp/fp/fn = {tp}/{fp}/{fn}")
    print(f"  detector disagrees with the merged set on {len(disagree_with_detector)} sentences")
    print("\n  These annotators share the detector author's priors, so this number")
    print("  measures self-consistency, not correctness. Use it to find broken")
    print("  cases; do not cite it as a validation figure.")

    # --- stratified estimate ---------------------------------------------
    if args.strata:
        strata = json.loads((ROOT / args.strata).read_text())
        fires = [r for r in merged if strata.get(r["id"]) == "fire"]
        mention = [r for r in merged if strata.get(r["id"]) == "mention_only"]

        s_tp = sum(1 for r in fires if r["label"] == POSITIVE)
        s_fp = len(fires) - s_tp
        s_prec = s_tp / len(fires) if fires else 0.0

        fn_sampled = sum(1 for r in mention if r["label"] == POSITIVE)
        scale = args.recall_population / len(mention) if mention else 0.0
        fn_est = fn_sampled * scale
        s_rec = s_tp / (s_tp + fn_est) if (s_tp + fn_est) else 0.0

        print(f"\n{'='*68}")
        print("  Stratified estimate")
        print(f"{'='*68}")
        print(f"  precision stratum: {len(fires)} fires, ALL labelled")
        print(f"    true positives   {s_tp}")
        print(f"    false positives  {s_fp}")
        print(f"    PRECISION        {s_prec:.3f}   (exact — no sampling error)")
        print(f"\n  recall stratum: {len(mention)} labelled, "
              f"sampled from {args.recall_population} (x{scale:.2f})")
        print(f"    missed positives in sample  {fn_sampled}")
        print(f"    estimated FN in population  {fn_est:.0f}")
        print(f"    RECALL (estimated)          {s_rec:.3f}")
        print("\n  Recall assumes no conclusion is phrased entirely outside the")
        print("  topic-keyword list. That is a simplifying assumption, not a")
        print("  proof — such a sentence would sit in the unsampled remainder.")

    print("\n  Adjudicate the clashes:")
    print("    python scripts/eval/build_think_label_tool.py \\")
    print(f"        --ids-file {args.out_clashes} \\")
    print("        --out results/analysis/adjudicate_clashes.html\n")


if __name__ == "__main__":
    main()
