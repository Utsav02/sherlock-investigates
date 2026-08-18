#!/usr/bin/env python3
"""
Precision / minimum-detectable-difference analysis for the proposed Track A contrasts.

Design §15 requires, before Stage B: "use the observed participant/game structure
and plausible intra-participant dependence to estimate expected interval widths
or minimum detectable differences for the proposed primary contrasts. Reduce and
freeze the contrast set if the available corpus cannot resolve them."

This script does exactly that, using the frozen split (`build_splits.py`) and the
measured games-per-user distribution rather than nominal row counts.

Two distinct notions of "unit" appear here and must not be conflated:

  * The **atom for splitting** is the connected component of the participant
    co-occurrence graph (15 in the main study). That controls *leakage*: two
    people who shared a game cannot sit in different splits.
  * The **cluster for inference** is the participant. That controls *dependence*:
    a user's ~4 games are correlated. Two users in the same component who never
    shared a game are not thereby correlated, so components are the wrong unit
    for variance and would be far too conservative.

Intra-cluster correlation (ICC) is unmeasured until an estimator exists, so every
quantity is reported across a plausible ICC grid rather than at one assumed value.

Stdlib only.

Usage:
    venv/bin/python v2/scripts/precision_track_a.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_three_party import SOURCE_DIR, is_blank, norm_id, read_csv  # noqa: E402
import build_splits  # noqa: E402

csv.field_size_limit(10_000_000)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "v2" / "results" / "stage_a" / "precision_track_a.json"

Z_ALPHA = 1.959964     # two-sided 95%
Z_BETA = 0.8416212     # 80% power
ICC_GRID = (0.0, 0.05, 0.10, 0.20)

# A contrast is "resolvable" if the effect it must detect is plausibly larger
# than its MDD at ICC=0.10. Thresholds are stated per contrast below.
RESOLVABLE_ICC = 0.10


# ---------------------------------------------------------------------------
# statistics (pure, unit-tested)
# ---------------------------------------------------------------------------

def design_effect(mean_cluster_size: float, icc: float) -> float:
    """Kish design effect for unequal-but-modest cluster sizes: 1 + (m-1)*ICC."""
    return 1.0 + (mean_cluster_size - 1.0) * icc


def effective_n(n: int, mean_cluster_size: float, icc: float) -> float:
    return n / design_effect(mean_cluster_size, icc)


def ci_halfwidth_proportion(n: int, p: float, m: float, icc: float) -> float:
    """95% CI half-width for one proportion under clustering."""
    return Z_ALPHA * math.sqrt(p * (1 - p) / effective_n(n, m, icc))


def mdd_paired_binary(n: int, discordance: float, m: float, icc: float) -> float:
    """MDD for two estimators scored on the SAME games (McNemar).

    Only the discordant games carry information, so precision is governed by the
    discordance rate, not by n alone.
    """
    return (Z_ALPHA + Z_BETA) * math.sqrt(
        discordance / effective_n(n, m, icc)
    )


def mdd_unpaired_binary(n1: int, n2: int, p: float, m: float, icc: float) -> float:
    """MDD between two disjoint groups of games (e.g. persona vs no-persona)."""
    var = p * (1 - p) * (1 / effective_n(n1, m, icc) + 1 / effective_n(n2, m, icc))
    return (Z_ALPHA + Z_BETA) * math.sqrt(var)


def auroc_ci_halfwidth(auc: float, n_pos: int, n_neg: int,
                       m: float, icc: float) -> float:
    """Hanley & McNeil (1982) standard error, with cluster inflation."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    npos = effective_n(n_pos, m, icc)
    nneg = effective_n(n_neg, m, icc)
    var = (auc * (1 - auc)
           + (npos - 1) * (q1 - auc ** 2)
           + (nneg - 1) * (q2 - auc ** 2)) / (npos * nneg)
    return Z_ALPHA * math.sqrt(max(var, 0.0))


def ci_halfwidth_mean(n: int, sd: float, m: float, icc: float) -> float:
    """95% CI half-width for a mean score (Brier, log loss) under clustering."""
    return Z_ALPHA * sd / math.sqrt(effective_n(n, m, icc))


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def measure_structure() -> dict:
    """Games per user, per split, measured — not assumed."""
    payload = build_splits.build()
    game_split = payload["game_split"]

    base = SOURCE_DIR / "data"
    _, games = read_csv(base / "tt_game.csv")
    _, interrogators = read_csv(base / "tt_interrogator.csv")
    _, witnesses = read_csv(base / "tt_witness.csv")
    seat_i = {norm_id(r["id"]): norm_id(r["user_id"]) for r in interrogators}
    seat_w = {norm_id(r["id"]): norm_id(r["user_id"]) for r in witnesses}

    per_split_interrogator: dict[str, Counter] = defaultdict(Counter)
    per_split_any: dict[str, Counter] = defaultdict(Counter)
    for game in games:
        gid = norm_id(game["id"])
        split = game_split[gid]
        i_user = seat_i.get(norm_id(game["interrogator_id"]))
        w_user = seat_w.get(norm_id(game["human_witness_id"]))
        if not is_blank(i_user):
            per_split_interrogator[split][i_user] += 1
            per_split_any[split][i_user] += 1
        if not is_blank(w_user):
            per_split_any[split][w_user] += 1

    out = {"splits": {}}
    for split in sorted(per_split_interrogator):
        counts = list(per_split_interrogator[split].values())
        out["splits"][split] = {
            "games": payload["summary"]["by_split"][split]["games"],
            "users": payload["summary"]["by_split"][split]["users"],
            "itb_557_games": payload["summary"]["by_split"][split]["itb_557_games"],
            "witness_systems": payload["summary"]["by_split"][split]["witness_systems"],
            "interrogators": len(counts),
            "games_per_interrogator_mean": round(sum(counts) / len(counts), 3),
            "games_per_interrogator_max": max(counts),
            "games_per_user_either_role_mean": round(
                sum(per_split_any[split].values()) / len(per_split_any[split]), 3
            ),
        }
    out["split_sha256"] = payload["sha256"]
    out["components"] = payload["summary"]["components"]
    return out


# ---------------------------------------------------------------------------
# contrasts
# ---------------------------------------------------------------------------

def evaluate_contrasts(structure: dict) -> list[dict]:
    """Each proposed Track A primary contrast, with its precision on this corpus."""
    S = structure["splits"]
    contrasts: list[dict] = []

    def grid(fn) -> dict:
        return {f"icc_{icc:g}": round(fn(icc), 4) for icc in ICC_GRID}

    # --- C1: A2 vs A0, paired on the same games -----------------------------
    for split in ("dev", "test"):
        s = S[split]
        n, m = s["games"], s["games_per_interrogator_mean"]
        contrasts.append({
            "id": f"C1[{split}]",
            "name": f"A2 (classifier) vs A0 (majority/lexical baseline), accuracy, paired, {split}",
            "design": "paired (same games, two estimators); McNemar",
            "n_games": n,
            "clusters": s["interrogators"],
            "mean_cluster_size": m,
            "assumption": "discordance 0.25 (estimators disagree on 1 game in 4)",
            "mdd_pp": {k: round(v * 100, 2)
                       for k, v in grid(lambda i: mdd_paired_binary(n, 0.25, m, i)).items()},
            "target_effect_pp": 10.0,
            "target_rationale": (
                "A0 on this corpus is ~53% (measured interrogator accuracy); a "
                "detector worth building should clear it by >=10pp"
            ),
        })

    # --- C2: A2 vs A1 (ITB reproduction), paired on the ITB subset ----------
    for split in ("dev", "test"):
        s = S[split]
        n = s["itb_557_games"]
        m = s["games_per_interrogator_mean"]
        contrasts.append({
            "id": f"C2[{split}]",
            "name": f"A2 vs A1 (Inverse Turing Bench reproduction), accuracy, paired, {split}",
            "design": "paired, restricted to the ITB 557 (both dialogues >=50 tokens)",
            "n_games": n,
            "clusters": s["interrogators"],
            "mean_cluster_size": m,
            "assumption": "discordance 0.25",
            "mdd_pp": {k: round(v * 100, 2)
                       for k, v in grid(lambda i: mdd_paired_binary(n, 0.25, m, i)).items()},
            "target_effect_pp": 10.0,
            "target_rationale": "same threshold as C1; the ITB subset is 48.9% of the corpus",
        })

    # --- C3: persona degradation, unpaired between witness systems ----------
    for split in ("dev", "test"):
        s = S[split]
        n1 = s["witness_systems"].get("gpt-4.5_quinn", 0)
        n2 = s["witness_systems"].get("gpt-4.5_minimal", 0)
        m = s["games_per_interrogator_mean"]
        contrasts.append({
            "id": f"C3[{split}]",
            "name": f"Persona degradation: GPT-4.5 PERSONA vs GPT-4.5 minimal, accuracy, {split}",
            "design": "unpaired, disjoint game sets",
            "n_games": n1 + n2,
            "group_sizes": [n1, n2],
            "clusters": s["interrogators"],
            "mean_cluster_size": m,
            "assumption": "p=0.5 (most conservative variance)",
            "mdd_pp": {k: round(v * 100, 2)
                       for k, v in grid(lambda i: mdd_unpaired_binary(n1, n2, 0.5, m, i)).items()},
            "target_effect_pp": 36.3,
            "target_rationale": (
                "measured human interrogator gap on the full corpus: 0.636 "
                "(gpt-4.5_minimal) - 0.273 (gpt-4.5_quinn) = 36.3pp"
            ),
        })

    # --- C4: family holdout, unpaired --------------------------------------
    for split in ("dev", "test"):
        s = S[split]
        ws = s["witness_systems"]
        n_openai = ws.get("gpt-4.5_quinn", 0) + ws.get("gpt-4.5_minimal", 0) + \
            ws.get("gpt-4o_minimal", 0)
        n_llama = ws.get("llama-405b_quinn", 0) + ws.get("llama-405b_minimal", 0)
        m = s["games_per_interrogator_mean"]
        contrasts.append({
            "id": f"C4[{split}]",
            "name": f"Leave-one-family-out: OpenAI vs Meta-Llama, accuracy, {split}",
            "design": "unpaired; TWO families only, so exploratory by design (§14)",
            "n_games": n_openai + n_llama,
            "group_sizes": [n_openai, n_llama],
            "clusters": s["interrogators"],
            "mean_cluster_size": m,
            "assumption": "p=0.5",
            "mdd_pp": {k: round(v * 100, 2)
                       for k, v in grid(lambda i: mdd_unpaired_binary(n_openai, n_llama, 0.5, m, i)).items()},
            "target_effect_pp": None,
            "target_rationale": (
                "no pre-registered effect size; a 2-point family comparison "
                "cannot support a generalisation claim regardless of precision"
            ),
        })

    # --- C5: template normalisation, paired --------------------------------
    for split in ("dev", "test"):
        s = S[split]
        n, m = s["games"], s["games_per_interrogator_mean"]
        contrasts.append({
            "id": f"C5[{split}]",
            "name": f"Template-normalised vs raw accuracy for the same estimator, {split}",
            "design": "paired (same games, same estimator, two input encodings)",
            "n_games": n,
            "clusters": s["interrogators"],
            "mean_cluster_size": m,
            "assumption": "discordance 0.15 (normalisation changes few decisions)",
            "mdd_pp": {k: round(v * 100, 2)
                       for k, v in grid(lambda i: mdd_paired_binary(n, 0.15, m, i)).items()},
            "target_effect_pp": 10.0,
            "target_rationale": (
                "a drop >=10pp under normalisation would mean the signal is "
                "substantially collection artefact"
            ),
        })

    # --- C6: absolute AUROC / calibration precision ------------------------
    for split in ("dev", "test"):
        s = S[split]
        n, m = s["games"], s["games_per_interrogator_mean"]
        # Each game contributes one human dialogue and one AI dialogue.
        contrasts.append({
            "id": f"C6[{split}]",
            "name": f"Absolute AUROC of a single estimator, {split}",
            "design": "one estimator, one number; n_pos = n_neg = games",
            "n_games": n,
            "clusters": s["interrogators"],
            "mean_cluster_size": m,
            "assumption": "true AUROC 0.75",
            "ci_halfwidth": grid(lambda i: auroc_ci_halfwidth(0.75, n, n, m, i)),
            "target_effect_pp": None,
            "target_rationale": "reported as an interval, not tested against a threshold",
        })
        contrasts.append({
            "id": f"C7[{split}]",
            "name": f"Absolute accuracy / Brier of a single estimator, {split}",
            "design": "one estimator, one number",
            "n_games": n,
            "clusters": s["interrogators"],
            "mean_cluster_size": m,
            "assumption": "accuracy 0.5 (worst case); Brier per-game sd 0.25",
            "accuracy_ci_halfwidth_pp": {
                k: round(v * 100, 2)
                for k, v in grid(lambda i: ci_halfwidth_proportion(n, 0.5, m, i)).items()
            },
            "brier_ci_halfwidth": grid(lambda i: ci_halfwidth_mean(n, 0.25, m, i)),
            "target_effect_pp": None,
            "target_rationale": "reported as an interval",
        })

    # --- C8/C9: grouped cross-fitting instead of a single held-out split ----
    # Components never straddle splits, so interrogators in train/dev/test are
    # disjoint and these pooled cluster sizes are exact, not approximations.
    def pooled(names: tuple[str, ...]) -> tuple[int, int, float]:
        n = sum(S[s]["games"] for s in names)
        clusters = sum(S[s]["interrogators"] for s in names)
        m = sum(S[s]["games_per_interrogator_mean"] * S[s]["interrogators"]
                for s in names) / clusters
        return n, clusters, m

    for cid, names, note in (
        ("C8[train+dev, cross-fitted]", ("train", "dev"),
         "11 folds; PRESERVES the frozen test split for Gate 5"),
        ("C9[all, cross-fitted]", ("train", "dev", "test"),
         "15 folds; uses every game and leaves NO untouched final test"),
    ):
        n, clusters, m = pooled(names)
        contrasts.append({
            "id": cid,
            "name": f"A2 vs A0, paired, leave-one-component-out cross-fitting ({'+'.join(names)})",
            "design": (
                "grouped CV, folds = co-occurrence components; every game is "
                f"scored by a model that never saw its participants. {note}"
            ),
            "n_games": n,
            "clusters": clusters,
            "mean_cluster_size": round(m, 3),
            "assumption": "discordance 0.25",
            "mdd_pp": {
                f"icc_{icc:g}": round(mdd_paired_binary(n, 0.25, m, icc) * 100, 2)
                for icc in ICC_GRID
            },
            "target_effect_pp": 10.0,
            "target_rationale": "same threshold as C1, on more games",
        })
    return contrasts


def classify(contrast: dict) -> str:
    """RESOLVABLE / MARGINAL / NOT RESOLVABLE at the reference ICC."""
    target = contrast.get("target_effect_pp")
    mdd = contrast.get("mdd_pp")
    if mdd is None:
        return "INTERVAL-ONLY"
    value = mdd[f"icc_{RESOLVABLE_ICC:g}"]
    if target is None:
        return "NO PRE-REGISTERED EFFECT"
    if value <= target * 0.7:
        return "RESOLVABLE"
    if value <= target:
        return "MARGINAL"
    return "NOT RESOLVABLE"


def main() -> int:
    structure = measure_structure()
    contrasts = evaluate_contrasts(structure)
    for contrast in contrasts:
        contrast["verdict"] = classify(contrast)

    payload = {
        "generated_by": "v2/scripts/precision_track_a.py",
        "split_sha256": structure["split_sha256"],
        "icc_grid": list(ICC_GRID),
        "reference_icc": RESOLVABLE_ICC,
        "power": 0.80,
        "alpha": 0.05,
        "structure": structure,
        "contrasts": contrasts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"split sha256 {structure['split_sha256'][:16]}…   "
          f"reference ICC {RESOLVABLE_ICC}, 80% power, alpha 0.05\n")
    for split, s in structure["splits"].items():
        print(f"  {split:5s} games={s['games']:4d} interrogators={s['interrogators']:3d} "
              f"games/interrogator={s['games_per_interrogator_mean']:.2f} "
              f"(max {s['games_per_interrogator_max']}) itb={s['itb_557_games']}")
    print()
    for c in contrasts:
        key = "mdd_pp" if "mdd_pp" in c else "accuracy_ci_halfwidth_pp"
        val = c.get(key, {}).get(f"icc_{RESOLVABLE_ICC:g}", "-")
        tgt = c.get("target_effect_pp")
        print(f"  {c['id']:22s} n={c['n_games']:4d}  "
              f"{'MDD' if key=='mdd_pp' else '+/-'}={val:>6}pp  "
              f"target={tgt if tgt else '-':>6}  {c['verdict']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
