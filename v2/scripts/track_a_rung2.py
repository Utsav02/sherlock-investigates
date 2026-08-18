#!/usr/bin/env python3
"""
Gate 1 rung 2: SONA/UCSD <-> Prolific transfer for arm A0.

Train every detector on the games of ONE recruitment population and evaluate on
the OTHER, in both directions. Only train+dev games are used; the frozen test
split is never read (Gate 5, one shot).

What this holds out, stated precisely
-------------------------------------
Recruitment source is **perfectly nested inside the participant co-occurrence
components** — measured this session: 0 of 15 components mix the two sources.
Two consequences, and they pull in opposite directions:

  * **Good:** training on one source automatically holds out every participant
    of the other, so this is a people holdout *and* a population holdout at once.
    No participant leakage is possible by construction.
  * **Bad:** because the nesting is perfect, population is entangled with the
    component structure — i.e. with batch and lobby scheduling. A drop in
    transfer cannot be attributed to "different people" as opposed to "different
    collection batch", because nothing in the corpus varies them independently.

So this is a **between-experiment holdout, confounded with batch/lobby
structure**. It is NOT a clean population holdout and must not be described as
one. It is also not an out-of-SOURCE holdout in the dataset sense: both halves
share one collection apparatus, one time window, and one witness-system set, so
it cannot detect a collection artefact common to both (that is rung 3's job).

Model-set balance was verified before running (all six witness systems present in
both sources; Prolific share 0.54-0.59 per system), so a transfer drop is not
explained by one population facing different AI systems.

Stdlib only.

Usage:
    venv/bin/python v2/scripts/track_a_rung2.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_canonical  # noqa: E402
import track_a_a0 as a0  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "v2" / "results" / "track_a"

# tt_profile.source codes, per the release codebook.
SOURCES = {"1": "prolific", "2": "sona_ucsd"}


def transfer(dialogues, games, condition, train_src, eval_src, n_boot, variant="raw"):
    """Fit on train_src games, score eval_src games. One fit, no cross-fitting:
    the two populations are participant-disjoint by construction."""
    by_game = defaultdict(dict)
    for d in dialogues:
        by_game[d["game_id"]][d["conversation_label"]] = d

    pool = [g for g in games if g["split"] in ("train", "dev")]
    train_games = [g for g in pool
                   if SOURCES.get(g["interrogator_recruitment_source"]) == train_src]
    held_games = [g for g in pool
                  if SOURCES.get(g["interrogator_recruitment_source"]) == eval_src]
    train_d = [by_game[g["game_id"]][l] for g in train_games for l in ("A", "B")]
    held_d = [by_game[g["game_id"]][l] for g in held_games for l in ("A", "B")]

    predictions, tie_breaks, diagnostics = {}, {}, {}
    for detector in a0.make_detectors(condition):
        detector.fit(train_d, variant)
        probs = detector.predict(held_d, variant)
        predictions[detector.name] = {
            d["example_id"]: p for d, p in zip(held_d, probs)
        }
        tie_breaks[detector.name] = detector.tie_break
        diagnostics[detector.name] = detector.diagnostics()

    scored = {
        name: a0.score_games(pr, held_games, by_game, tie_breaks.get(name))
        for name, pr in predictions.items()
    }
    index = list(range(len(held_games)))
    point = {n: a0.metrics_from(s, index) for n, s in scored.items()}
    intervals = a0.bootstrap_intervals(scored, held_games, n_boot, a0.SEED)

    participant = {"detectors": {}}
    for name in sorted(scored):
        participant["detectors"][name] = {
            key: a0.widen(intervals["interrogator"]["detectors"][name][key],
                          intervals["human_witness"]["detectors"][name][key])
            for key in list(a0.METRIC_KEYS) + ["game_accuracy_diff_vs_majority"]
        }
    participant["n_clusters"] = (
        f"{intervals['interrogator']['n_clusters']} interrogators / "
        f"{intervals['human_witness']['n_clusters']} human witnesses "
        f"(max-of-marginals)")
    intervals["participant"] = participant

    by_system = {}
    for name, s in scored.items():
        rows = defaultdict(list)
        for g, hit in zip(held_games, s["game_correct"]):
            rows[g["witness_system"]].append(hit)
        by_system[name] = {k: round(sum(v) / len(v), 4) for k, v in sorted(rows.items())}

    return {
        "condition": condition.as_dict(),
        "train_source": train_src, "eval_source": eval_src,
        "n_train_games": len(train_games), "n_eval_games": len(held_games),
        "eval_witness_systems": dict(
            sorted(Counter(g["witness_system"] for g in held_games).items())),
        "point_estimates": {n: {k: round(v, 4) for k, v in m.items()}
                            for n, m in point.items()},
        "P2_calibration": {
            n: {"dialogue_brier": round(point[n]["dialogue_brier"], 4),
                "dialogue_ece": round(a0.ece(s["dialogue_probs"], s["dialogue_labels"]), 4),
                "game_brier": round(point[n]["game_brier"], 4)}
            for n, s in scored.items()
        },
        "intervals": intervals,
        "descriptive_accuracy_by_witness_system": by_system,
        "fit_diagnostics": diagnostics,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--conditions", nargs="*",
                        default=["A0-full", "A0-wit-nolen", "A0-wit-nolen-capped"])
    args = parser.parse_args(argv)

    dialogues, games, manifest = build_canonical.load()
    by_name = {c.name: c for c in a0.CONDITIONS}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_by": "v2/scripts/track_a_rung2.py",
        "generated_at_utc": stamp,
        "gate": "Gate 1 rung 2",
        "holdout_type": "between-experiment (recruitment population), CONFOUNDED "
                        "with batch/lobby structure: recruitment source is perfectly "
                        "nested inside the co-occurrence components (0 of 15 mixed). "
                        "NOT a clean population holdout; NOT a dataset/source holdout.",
        "test_split": "UNTOUCHED (Gate 5, one shot)",
        "canonical": {"source_revision": manifest["source_revision"],
                      "split_sha256": manifest["split_sha256"]},
        "bootstrap": {"replicates": args.bootstrap, "seed": a0.SEED},
        "directions": {},
    }

    for cname in args.conditions:
        for train_src, eval_src in (("sona_ucsd", "prolific"), ("prolific", "sona_ucsd")):
            key = f"{cname}::{train_src}->{eval_src}"
            started = time.time()
            print(f"[{key}] ...", flush=True)
            result = transfer(dialogues, games, by_name[cname],
                              train_src, eval_src, args.bootstrap)
            result["seconds"] = round(time.time() - started, 1)
            payload["directions"][key] = result
            for n, m in result["point_estimates"].items():
                print(f"  {n:16s} game_acc={m['game_accuracy']:.4f} "
                      f"auroc={m['dialogue_auroc']:.4f}", flush=True)

    out = OUT_DIR / f"a0_rung2_transfer_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
