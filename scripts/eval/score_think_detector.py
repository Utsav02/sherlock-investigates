#!/usr/bin/env python3
"""Score the t_think_07 detector against hand labels. This is Gate 1.

Reads data/probes/think_stance_labels_v1.jsonl (produced by the browser tool
from build_think_label_tool.py) and reports precision / recall / F1 for the
directed-suspicion detector, with the legacy topic-mention detector as a
baseline for comparison.

Positive class = "conclusion" (the sentence asserts the interlocutor is an AI).
"task_restatement" and "neither" are negatives; "skip" rows are dropped.

Gate: precision >= 0.8 before any pilot data is collected. Precision is
weighted above recall deliberately — a false positive sets t_think_07 too
early and inflates the measured commitment gap, which is the headline number.
A false negative loses a conversation from the numerator, which is recoverable.

Every disagreement is printed. Read them: the failure cases are what tell you
whether to keep patching the regex (Option A) or move to a stance classifier
(Option B, ../experiment.md section 4.2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "conversation"))

import conv_logging  # noqa: E402

POSITIVE = "conclusion"
GATE_PRECISION = 0.80


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(rows: list[dict], predict) -> dict:
    tp = fp = tn = fn = 0
    errors: list[tuple[str, dict]] = []
    for row in rows:
        gold = row["label"] == POSITIVE
        pred = bool(predict(row["text"]))
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
            errors.append(("FP", row))
        elif not pred and gold:
            fn += 1
            errors.append(("FN", row))
        else:
            tn += 1
    precision, recall, f1 = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(rows) if rows else 0.0,
            "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", default="data/probes/think_stance_labels_v1.jsonl")
    ap.add_argument("--show-errors", type=int, default=25,
                    help="how many disagreements to print (0 for none)")
    args = ap.parse_args()

    path = ROOT / args.labels
    if not path.exists():
        raise SystemExit(
            f"no label file at {path}\n"
            f"  build the tool:  python scripts/eval/build_think_label_tool.py\n"
            f"  label in a browser, export, and save it there."
        )

    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("label") != "skip"]
    if not rows:
        raise SystemExit("label file has no non-skip rows")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    directed = evaluate(rows, conv_logging._think_block_suspicious)
    legacy = evaluate(rows, conv_logging._think_block_mentions_ai)

    print(f"\n{'='*66}")
    print(f"  t_think_07 detector validation — n={len(rows)} labelled sentences")
    print(f"{'='*66}")
    print("  label distribution: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"  positive class: {POSITIVE!r} ({counts.get(POSITIVE, 0)} sentences)\n")

    header = f"  {'detector':<24}{'prec':>8}{'recall':>8}{'F1':>8}{'acc':>8}   tp/fp/fn"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, res in (("directed (t_think_07)", directed),
                      ("legacy (t_think_topic)", legacy)):
        print(f"  {name:<24}{res['precision']:>8.3f}{res['recall']:>8.3f}"
              f"{res['f1']:>8.3f}{res['accuracy']:>8.3f}   "
              f"{res['tp']}/{res['fp']}/{res['fn']}")

    if args.show_errors and directed["errors"]:
        print(f"\n  Disagreements (directed detector), first {args.show_errors}:")
        for kind, row in directed["errors"][:args.show_errors]:
            marker = "fired, should not have" if kind == "FP" else "missed a conclusion"
            print(f"\n   [{kind}] {marker}  ({row['id']}, gold={row['label']})")
            print(f"        {row['text'][:200]}")

    gate_pass = directed["precision"] >= GATE_PRECISION
    print(f"\n{'='*66}")
    if gate_pass:
        print(f"  GATE 1 PASS — precision {directed['precision']:.3f} >= {GATE_PRECISION}")
        print("  The instrument is validated. Proceed to the conversation runs.")
    else:
        print(f"  GATE 1 FAIL — precision {directed['precision']:.3f} < {GATE_PRECISION}")
        print("  Do NOT collect pilot data yet. Read the false positives above:")
        print("    - a few near-miss patterns  -> extend the vetoes in conv_logging")
        print("    - diffuse, varied failures  -> switch to a stance classifier")
        print("      (Option B, ../experiment.md section 4.2)")
    print(f"{'='*66}")
    print("  Log the resulting precision/recall in the CLAUDE.md Decision Log —")
    print("  the detector is the measurement instrument.\n")

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
