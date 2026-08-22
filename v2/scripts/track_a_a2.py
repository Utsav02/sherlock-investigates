#!/usr/bin/env python3
"""
Track A arm A2: probabilistic head on a temporally clean frozen representation
model, EVALUATED for calibration (design §5, §12, §13.1).

Naming, because the distinction is load-bearing
-----------------------------------------------
Design §13.1 specifies a "calibrated full-history classifier head". This module
does NOT implement that. It fits an ordinary L2 logistic regression and *measures*
Brier/ECE/reliability on out-of-fold predictions. There is no Platt, isotonic, or
temperature stage fitted inside the training folds.

**Out-of-fold calibration EVALUATION is not out-of-fold CALIBRATION.** The
reported Brier and ECE numbers are legitimate measurements of how well this head's
raw probabilities happen to be calibrated; they are not evidence that a
calibration procedure was applied. Calling this arm "calibrated" would claim a
component that does not exist, so it is called a probabilistic head throughout.
Adding a properly nested calibrator is outstanding work, deliberately not done
here.

The question this run exists to answer
--------------------------------------
A0 (bag of words) reaches 0.925 paired accuracy when evaluation holds out people
but not witness systems, and collapses to 0.58-0.64 under a held-out persona
prompt. Is that ceiling a property of BAG OF WORDS, or of the CORPUS? A2 swaps
the representation for a frozen neural one and re-runs the same two cuts. If A2
also collapses, the limit is not specific to bag-of-words, and Track A concludes.

Stated at the width the design permits: a second method failing the same cut is
evidence that the limitation is not an artefact of lexical representation. It is
NOT proof that the corpus contains no learnable general signal — that would need
more than two methods, and A1 and A3 were never run.

Temporal cleanliness
--------------------
Representation model: **Qwen2.5-7B, released 2024-09-19**, already present in the
local Ollama store (pulled ~4 months before this session; no new download). The
witness conversations were collected **2025-03-06 to 2025-03-14**, so the
checkpoint is finalized ~6 months before the data it is asked to represent. It
cannot have been trained on these transcripts.

This removes one specific contamination route. It does NOT prove freedom from
every kind of training contamination — Qwen2.5 has certainly seen Turing-test
discussion, chat transcripts, and LLM output in general.

The model is used ONLY as a frozen feature extractor: mean-pooled final hidden
states via llama-server's embedding endpoint, weights never updated. Only a
lightweight logistic head is trained, per §5.

Why a random projection sits between the embedding and the head
---------------------------------------------------------------
This venv has no numpy and no BLAS, and the embedding is 3,584-dimensional. A
logistic fit on 3,584 dense dimensions in pure Python is ~1.6 s per gradient step
per fold, which does not fit a time-boxed session across 33 folds. So the
embeddings pass through a **very sparse random projection** (Li et al. 2006,
density 1/sqrt(d)) to 512 dimensions with a fixed seed, which preserves inner
products in expectation.

Stated plainly because it bounds the conclusion: the projection can only *lose*
information, never add it. So it is a conservative choice — if A2 still collapses
on the persona cut while scoring high on the people-only cut, the projection is
not the explanation, because the same projection is in both numbers.

Usage:
    venv/bin/python v2/scripts/track_a_a2.py --embed-url http://127.0.0.1:51999
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_canonical  # noqa: E402
import track_a_a0 as a0  # noqa: E402
import track_a_rung4 as r4  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "v2" / "results" / "track_a"
CACHE = REPO_ROOT / "v2" / "data" / "canonical" / "a2_embeddings"

REPRESENTATION = {
    "model": "Qwen2.5-7B-Instruct (Q4_K_M, via Ollama blob)",
    "released": "2024-09-19",
    "witness_data_collected": "2025-03-06 .. 2025-03-14",
    "temporally_clean": True,
    "pooling": "mean over final hidden states",
    "dim": 3584,
    "frozen": "weights never updated; used only as a feature extractor",
    "provenance": "already present in the local Ollama store; no new download",
}
PROJ_DIM = 512
PROJ_SEED = 20260818


# ---------------------------------------------------------------------------
# embedding (cached)
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str], url: str) -> list[list[float]]:
    req = urllib.request.Request(
        f"{url}/v1/embeddings",
        data=json.dumps({"input": texts, "model": "frozen"}).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "sherlock-investigates-v2-research/0.1"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return [d["embedding"] for d in json.loads(r.read())["data"]]


def build_embeddings(dialogues, url, batch=16) -> dict[str, list[float]]:
    """Embed the witness-only text of each dialogue. Cached to disk as float32."""
    CACHE.mkdir(parents=True, exist_ok=True)
    bin_path, idx_path = CACHE / "witness_raw.f32", CACHE / "witness_raw.index.json"
    if bin_path.exists() and idx_path.exists():
        order = json.loads(idx_path.read_text())
        raw = bin_path.read_bytes()
        dim = REPRESENTATION["dim"]
        out = {}
        for i, key in enumerate(order):
            out[key] = list(struct.unpack_from(f"<{dim}f", raw, i * dim * 4))
        print(f"  loaded {len(out)} cached embeddings", flush=True)
        return out

    texts, keys = [], []
    for d in dialogues:
        msgs = a0.dialogue_messages(d, "raw", "witness")
        texts.append("\n".join(msgs) if msgs else " ")
        keys.append(d["example_id"])

    vectors, started = [], time.time()
    for i in range(0, len(texts), batch):
        vectors.extend(embed_batch(texts[i:i + batch], url))
        if (i // batch) % 10 == 0:
            done = min(i + batch, len(texts))
            rate = (time.time() - started) / max(done, 1)
            print(f"  embedded {done}/{len(texts)}  eta {rate*(len(texts)-done)/60:.1f} min",
                  flush=True)
    with bin_path.open("wb") as fh:
        for v in vectors:
            fh.write(struct.pack(f"<{len(v)}f", *v))
    idx_path.write_text(json.dumps(keys))
    return dict(zip(keys, vectors))


def sparse_projection(d_in: int, d_out: int, seed: int):
    """Very sparse random projection (Li, Hastie & Church 2006), density 1/sqrt(d).

    Returns, for each OUTPUT coordinate, a list of (input_index, +-scale). Fixed
    seed, so the projection is a deterministic function of (d_in, d_out, seed)
    and identical across every fold and every cut.
    """
    rng = random.Random(seed)
    s = math.sqrt(d_in)                      # 1/density
    scale = math.sqrt(s) / math.sqrt(d_out)
    nnz = max(1, int(round(d_in / s)))
    cols = []
    for _ in range(d_out):
        idx = rng.sample(range(d_in), nnz)
        cols.append([(i, scale if rng.random() < 0.5 else -scale) for i in idx])
    return cols


def project(vec: list[float], cols) -> list[float]:
    out = [sum(vec[i] * w for i, w in col) for col in cols]
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]            # L2-normalise, as for TF-IDF


# ---------------------------------------------------------------------------
# the head
# ---------------------------------------------------------------------------

class FrozenRepHead:
    """Lightweight logistic head over frozen, projected representations."""

    name = "a2_frozen_rep"
    tie_break = None

    def __init__(self, feats: dict[str, list[float]], l2: float = 1.0,
                 iters: int = 150):
        self.feats, self.l2, self.iters = feats, l2, iters
        self.w, self.bias, self.diag = [], 0.0, {}

    def fit(self, train, variant=None):
        rows = [list(enumerate(self.feats[d["example_id"]])) for d in train]
        y = [0 if d["is_human"] else 1 for d in train]
        self.w, self.bias, self.diag = a0.fit_sparse_logistic(
            rows, y, dim=PROJ_DIM, l2=self.l2, iters=self.iters, lr=2.0)

    def predict(self, dialogues, variant=None):
        out = []
        for d in dialogues:
            v = self.feats[d["example_id"]]
            out.append(a0.sigmoid(self.bias + sum(w * x for w, x in zip(self.w, v))))
        return out

    def diagnostics(self):
        return dict(self.diag)


def _fit_predict(train_games, held_games, by_game, feats, l2):
    train_d = [by_game[g["game_id"]][l] for g in train_games for l in ("A", "B")]
    held_d = [by_game[g["game_id"]][l] for g in held_games for l in ("A", "B")]
    head = FrozenRepHead(feats, l2=l2)
    head.fit(train_d)
    return {d["example_id"]: p for d, p in zip(held_d, head.predict(held_d))}, head


def _score(preds, games, by_game, n_boot):
    scored = {"a2_frozen_rep": a0.score_games(preds, games, by_game, None)}
    point = a0.metrics_from(scored["a2_frozen_rep"], list(range(len(games))))
    iv = a0.bootstrap_intervals(scored, games, n_boot, a0.SEED)
    participant = {
        k: iv["participant"]["detectors"]["a2_frozen_rep"][k]
        for k in list(a0.METRIC_KEYS) + ["game_accuracy_diff_vs_majority"]
    }
    component = {
        k: iv["component"]["detectors"]["a2_frozen_rep"][k]
        for k in list(a0.METRIC_KEYS) + ["game_accuracy_diff_vs_majority"]
    }
    ai = [preds[by_game[g["game_id"]][g["ai_conversation_label"]]["example_id"]]
          for g in games]
    hu = [preds[by_game[g["game_id"]][g["human_conversation_label"]]["example_id"]]
          for g in games]
    s = scored["a2_frozen_rep"]
    return {
        "n_games": len(games),
        "point_estimates": {k: round(v, 4) for k, v in point.items()},
        "participant_ci": participant,
        "component_ci": component,
        "side_recognition": {
            "mean_p_ai_on_AI_side": round(sum(ai) / len(ai), 4),
            "mean_p_ai_on_HUMAN_side": round(sum(hu) / len(hu), 4),
            "ai_side_above_half": round(sum(1 for p in ai if p >= 0.5) / len(ai), 4),
            "human_side_below_half": round(sum(1 for p in hu if p < 0.5) / len(hu), 4),
        },
        "calibration": {
            "dialogue_brier": round(point["dialogue_brier"], 4),
            "dialogue_ece": round(a0.ece(s["dialogue_probs"], s["dialogue_labels"]), 4),
            "reliability": a0.reliability(s["dialogue_probs"], s["dialogue_labels"]),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embed-url", default="http://127.0.0.1:51999")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args(argv)

    dialogues, games_all, manifest = build_canonical.load()
    drop = a0.empty_witness_games(dialogues)
    pool = [g for g in games_all if g["split"] in ("train", "dev")]
    games = [g for g in pool if g["game_id"] not in drop]
    keep = {g["game_id"] for g in games}
    dial = [d for d in dialogues if d["game_id"] in keep]
    by_game = defaultdict(dict)
    for d in dial:
        by_game[d["game_id"]][d["conversation_label"]] = d

    print(f"games={len(games)} (dropped {len(pool)-len(games)} empty-witness) "
          f"dialogues={len(dial)}", flush=True)

    print("embedding (frozen Qwen2.5-7B, mean-pooled) ...", flush=True)
    raw = build_embeddings(dial, args.embed_url)
    cols = sparse_projection(REPRESENTATION["dim"], PROJ_DIM, PROJ_SEED)
    print(f"projecting {REPRESENTATION['dim']} -> {PROJ_DIM} (very sparse RP, "
          f"seed {PROJ_SEED}) ...", flush=True)
    feats = {k: project(v, cols) for k, v in raw.items()}

    payload = {
        "generated_by": "v2/scripts/track_a_a2.py",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "arm": "A2",
        "representation": REPRESENTATION,
        "projection": {"method": "very sparse random projection (Li et al. 2006)",
                       "d_out": PROJ_DIM, "seed": PROJ_SEED,
                       "note": "can only lose information; conservative for A2"},
        "head": {"type": "L2 logistic regression (NOT calibrated; no Platt/"
                     "isotonic/temperature stage)", "l2": args.l2, "iters": 150,
                     "calibration": "evaluated out of fold, not fitted"},
        "evalset_policy": "empty-witness games dropped (same set as A0)",
        "test_split": "UNTOUCHED (Gate 5, one shot)",
        "canonical": {"source_revision": manifest["source_revision"],
                      "split_sha256": manifest["split_sha256"]},
        "inference": {
            "bootstrap_replicates": args.bootstrap, "seed": a0.SEED,
            "additive_metrics": "dyadic participant-cluster sandwich",
            "nonadditive_metrics": "connected-component percentile bootstrap",
        },
        "evalset_composition": a0.evalset_composition(games, dial),
    }

    # --- §12 feasibility check: train split -> dev split, must beat majority --
    tr = [g for g in games if g["split"] == "train"]
    dv = [g for g in games if g["split"] == "dev"]
    print(f"\n[§12 feasibility] fit on train ({len(tr)}) -> dev ({len(dv)})", flush=True)
    preds, head = _fit_predict(tr, dv, by_game, feats, args.l2)
    feas = _score(preds, dv, by_game, args.bootstrap)
    maj_slot = max(("A", "B"), key=lambda L: sum(
        1 for g in tr if g["human_conversation_label"] == L))
    maj = sum(1 for g in dv if g["human_conversation_label"] == maj_slot) / len(dv)
    feas["majority_baseline"] = round(maj, 4)
    feas["passes"] = feas["point_estimates"]["game_accuracy"] > maj
    feas["head_diagnostics"] = head.diagnostics()
    payload["feasibility_check"] = feas
    print(f"  A2 dev game_accuracy = {feas['point_estimates']['game_accuracy']:.4f}"
          f"   majority = {maj:.4f}   -> {'PASS' if feas['passes'] else 'FAIL'}", flush=True)

    if not feas["passes"]:
        payload["verdict"] = ("STOPPED at the §12 feasibility check: the temporally "
                              "clean arm is capability-floored on this corpus.")
        out = OUT_DIR / f"a2_frozen_rep_{payload['generated_at_utc']}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nFEASIBILITY FAILED - stopping as specified.\nwrote {out}")
        return 0

    # --- cut 1: people-only, 11-fold leave-one-component-out ------------------
    print("\n[cut 1] people-only, leave-one-component-out", flush=True)
    preds_all = {}
    for comp in sorted({g["component"] for g in games}):
        held = [g for g in games if g["component"] == comp]
        train = [g for g in games if g["component"] != comp]
        p, _ = _fit_predict(train, held, by_game, feats, args.l2)
        preds_all.update(p)
        print(f"  fold comp={comp} n_train={len(train)} n_held={len(held)}", flush=True)
    payload["people_only"] = _score(preds_all, games, by_game, args.bootstrap)
    print(f"  A2 people-only game_accuracy = "
          f"{payload['people_only']['point_estimates']['game_accuracy']:.4f}", flush=True)

    # --- cut 2: LOSO-persona, component holdout nested inside -----------------
    print("\n[cut 2] LOSO-persona (component holdout nested)", flush=True)
    payload["loso_persona"] = {}
    for train_p, eval_p in (("minimal", "quinn"), ("quinn", "minimal")):
        preds_p = {}
        eval_all = [g for g in games if r4.PERSONA_OF.get(g["witness_system"]) == eval_p]
        for comp in sorted({g["component"] for g in eval_all}):
            held = [g for g in eval_all if g["component"] == comp]
            train = [g for g in games
                     if r4.PERSONA_OF.get(g["witness_system"]) == train_p
                     and g["component"] != comp]
            if not held or not train:
                continue
            p, _ = _fit_predict(train, held, by_game, feats, args.l2)
            preds_p.update(p)
        scored_games = [g for g in eval_all
                        if by_game[g["game_id"]]["A"]["example_id"] in preds_p]
        key = f"{train_p}->{eval_p}"
        payload["loso_persona"][key] = _score(preds_p, scored_games, by_game,
                                              args.bootstrap)
        sr = payload["loso_persona"][key]["side_recognition"]
        print(f"  {key:18s} n={len(scored_games):3d} "
              f"acc={payload['loso_persona'][key]['point_estimates']['game_accuracy']:.4f} "
              f"AI-flagged={sr['ai_side_above_half']:.3f} "
              f"human-cleared={sr['human_side_below_half']:.3f}", flush=True)

    out = OUT_DIR / f"a2_frozen_rep_{payload['generated_at_utc']}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
