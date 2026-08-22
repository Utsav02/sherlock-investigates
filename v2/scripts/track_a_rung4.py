#!/usr/bin/env python3
"""
Gate 1 rung 4: leave-one-SYSTEM-out and leave-one-PERSONA-out for arm A0.

This is the binding item. Every earlier Track A number holds out *people*, never
*witness systems*: all six systems appear in every training fold, so A0 has
always been a six-system classifier evaluated in-distribution. Rung 4 asks the
question the corpus has not yet been made to answer — does the signal survive a
witness the model has never seen?

Two cuts, both on the **A0-wit-nolen-capped** condition, the only cell whose
signal is not length (witness turns only, every length-derived feature dropped,
each side truncated to a fixed 20-token budget):

  LOSO-system   6 folds, holding out eliza, gpt-4.5_minimal, gpt-4.5_quinn,
                gpt-4o_minimal, llama-405b_minimal, llama-405b_quinn.
  LOSO-persona  train on all `_minimal` systems and evaluate on all `_quinn`,
                and the reverse. The sharper cut: it separates "learned the
                persona prompt" from "learned AI-ness". ELIZA belongs to neither
                persona class and is excluded from this cut, which is stated
                rather than quietly folded in.

Participant nesting, which is the part that is easy to get wrong
---------------------------------------------------------------
Holding out a system alone does NOT hold out people. The same interrogator and
the same human witness play several games against different AI systems, so a
naive system holdout trains on person P against system X and evaluates on person
P against system S — and because the paired task can be solved by recognising
the *human* side, that leak would inflate the transfer number precisely where it
matters.

So the component holdout is nested inside the system holdout. For held-out
system S and held-out component C:

    train = games whose system != S AND whose component != C
    eval  = games whose system == S AND whose component == C

Every game falls in exactly one (S, C) cell, so the union of the eval sets is
the whole corpus and each game is scored by a model that saw neither its witness
system nor either of its participants. The same nesting is applied to the
persona cut, over components.

Evaluation-set policy is inherited from `track_a_a0.empty_witness_games`: games
where either witness side sent no message are dropped, in every cell, because
silence is not a text property. The frozen test split is never read.

Stdlib only.

Usage:
    venv/bin/python v2/scripts/track_a_rung4.py
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

PERSONA_OF = {
    "gpt-4.5_minimal": "minimal", "gpt-4o_minimal": "minimal",
    "llama-405b_minimal": "minimal",
    "gpt-4.5_quinn": "quinn", "llama-405b_quinn": "quinn",
    "eliza": "eliza",           # neither; excluded from the persona cut
}


def _fit_cell(train_games, held_games, by_game, condition, variant="raw"):
    """One nested cell. Returns per-dialogue p(AI) plus the majority tie-break."""
    train_d = [by_game[g["game_id"]][l] for g in train_games for l in ("A", "B")]
    held_d = [by_game[g["game_id"]][l] for g in held_games for l in ("A", "B")]
    preds, ties = {}, {}
    for det in a0.make_detectors(condition):
        det.fit(train_d, variant)
        for d, p in zip(held_d, det.predict(held_d, variant)):
            preds.setdefault(det.name, {})[d["example_id"]] = p
        ties[det.name] = det.tie_break
    return preds, ties


def _score(preds, ties, games, by_game, n_boot, dialogues):
    scored = {n: a0.score_games(pr, games, by_game, ties.get(n))
              for n, pr in preds.items()}
    index = list(range(len(games)))
    point = {n: a0.metrics_from(s, index) for n, s in scored.items()}
    intervals = a0.bootstrap_intervals(scored, games, n_boot, a0.SEED)
    # Did the model actually RECOGNISE the unseen AI, or just recognise humans?
    # The paired task can be won from the human side alone, so this separates
    # the two.
    recog = {}
    for n, pr in preds.items():
        ai = [pr[by_game[g["game_id"]][g["ai_conversation_label"]]["example_id"]]
              for g in games]
        hu = [pr[by_game[g["game_id"]][g["human_conversation_label"]]["example_id"]]
              for g in games]
        recog[n] = {
            "mean_p_ai_on_AI_side": round(sum(ai) / len(ai), 4) if ai else None,
            "mean_p_ai_on_HUMAN_side": round(sum(hu) / len(hu), 4) if hu else None,
            "ai_side_above_half": round(sum(1 for p in ai if p >= 0.5) / len(ai), 4) if ai else None,
            "human_side_below_half": round(sum(1 for p in hu if p < 0.5) / len(hu), 4) if hu else None,
        }
    return {
        "n_games": len(games),
        "point_estimates": {n: {k: round(v, 4) for k, v in m.items()}
                            for n, m in point.items()},
        "intervals": intervals,
        "side_recognition": recog,
        "calibration": {n: {"dialogue_ece": round(
            a0.ece(s["dialogue_probs"], s["dialogue_labels"]), 4)}
            for n, s in scored.items()},
    }


def loso_system(dialogues, games, by_game, condition, n_boot):
    systems = sorted({g["witness_system"] for g in games})
    out, cells = {}, []
    for held_system in systems:
        preds, ties = {}, {}
        held_all = [g for g in games if g["witness_system"] == held_system]
        for comp in sorted({g["component"] for g in held_all}):
            held = [g for g in held_all if g["component"] == comp]
            train = [g for g in games
                     if g["witness_system"] != held_system and g["component"] != comp]
            if not held or not train:
                continue
            p, t = _fit_cell(train, held, by_game, condition)
            for name, mapping in p.items():
                preds.setdefault(name, {}).update(mapping)
            ties.update(t)
            cells.append({"held_system": held_system, "held_component": comp,
                          "n_train": len(train), "n_eval": len(held)})
        scored_games = [g for g in held_all
                        if by_game[g["game_id"]]["A"]["example_id"]
                        in next(iter(preds.values()))]
        out[held_system] = _score(preds, ties, scored_games, by_game, n_boot, dialogues)
        out[held_system]["n_cells"] = sum(
            1 for c in cells if c["held_system"] == held_system)
        print(f"  LOSO-system {held_system:20s} n={out[held_system]['n_games']:3d} "
              f"tfidf={out[held_system]['point_estimates']['tfidf_lr']['game_accuracy']:.4f}",
              flush=True)
    return out, cells


def loso_persona(dialogues, games, by_game, condition, n_boot):
    out = {}
    for train_p, eval_p in (("minimal", "quinn"), ("quinn", "minimal")):
        preds, ties = {}, {}
        eval_all = [g for g in games if PERSONA_OF.get(g["witness_system"]) == eval_p]
        for comp in sorted({g["component"] for g in eval_all}):
            held = [g for g in eval_all if g["component"] == comp]
            train = [g for g in games
                     if PERSONA_OF.get(g["witness_system"]) == train_p
                     and g["component"] != comp]
            if not held or not train:
                continue
            p, t = _fit_cell(train, held, by_game, condition)
            for name, mapping in p.items():
                preds.setdefault(name, {}).update(mapping)
            ties.update(t)
        scored_games = [g for g in eval_all
                        if by_game[g["game_id"]]["A"]["example_id"]
                        in next(iter(preds.values()))]
        key = f"{train_p}->{eval_p}"
        out[key] = _score(preds, ties, scored_games, by_game, n_boot, dialogues)
        out[key]["eval_witness_systems"] = dict(sorted(
            Counter(g["witness_system"] for g in scored_games).items()))
        print(f"  LOSO-persona {key:18s} n={out[key]['n_games']:3d} "
              f"tfidf={out[key]['point_estimates']['tfidf_lr']['game_accuracy']:.4f}",
              flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--condition", default="A0-wit-nolen-capped")
    ap.add_argument("--cuts", nargs="*", default=["system", "persona"],
                    choices=["system", "persona"],
                    help="which rung-4 cuts to run (default both)")
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument("--retain-empty", action="store_true",
                    help="keep empty-witness games (default drops them)")
    args = ap.parse_args(argv)

    dialogues, games_all, manifest = build_canonical.load()
    condition = next(c for c in a0.CONDITIONS if c.name == args.condition)
    drop = set() if args.retain_empty else a0.empty_witness_games(dialogues)
    pool = [g for g in games_all if g["split"] in ("train", "dev")]
    games = [g for g in pool if g["game_id"] not in drop]
    n_dropped_from_pool = len(pool) - len(games)
    by_game = defaultdict(dict)
    for d in dialogues:
        by_game[d["game_id"]][d["conversation_label"]] = d

    print(f"condition={condition.name}  train+dev={len(pool)}  "
          f"dropped={n_dropped_from_pool}  evaluated={len(games)}", flush=True)
    started = time.time()
    systems, cells = ({}, [])
    if "system" in args.cuts:
        systems, cells = loso_system(dialogues, games, by_game, condition, args.bootstrap)
    personas = {}
    if "persona" in args.cuts:
        personas = loso_persona(dialogues, games, by_game, condition, args.bootstrap)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "generated_by": "v2/scripts/track_a_rung4.py",
        "generated_at_utc": stamp,
        "gate": "Gate 1 rung 4 (held-out witness system / held-out persona)",
        "condition": condition.as_dict(),
        "evalset_policy": ("empty-witness games DROPPED (silence is not a text "
                           "property); same set in every cell"
                           if drop else "all games retained"),
        "n_games_dropped_from_eval_pool": n_dropped_from_pool,
        "n_train_dev_games_before_drop": len(pool),
        "evalset_composition": a0.evalset_composition(games, dialogues),
        "nesting": ("component holdout NESTED inside the system/persona holdout: "
                    "train = (system != held) AND (component != held); "
                    "eval = (system == held) AND (component == held). No eval game "
                    "shares a witness system OR a participant with its training set."),
        "test_split": "UNTOUCHED (Gate 5, one shot)",
        "canonical": {"source_revision": manifest["source_revision"],
                      "split_sha256": manifest["split_sha256"]},
        "inference": {
            "bootstrap_replicates": args.bootstrap, "seed": a0.SEED,
            "additive_metrics": "dyadic participant-cluster sandwich",
            "nonadditive_metrics": "connected-component percentile bootstrap",
        },
        "cuts_run": args.cuts,
        "loso_system": systems,
        "loso_persona": personas,
        "cells": cells,
        "seconds": round(time.time() - started, 1),
    }
    tag = f"_{args.tag}" if args.tag else ""
    out = OUT_DIR / f"a0_rung4_loso_{stamp}{tag}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
