#!/usr/bin/env python3
"""Exact, deterministic D0 simulator and preregistered Gate 2A analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from resumable import Run, append_jsonl, atomic_write_json

POLICIES = ("random", "fixed", "bed_eig", "uot_sample")
CATEGORIES = ("human_cue", "neutral", "ai_cue")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "v2/configs/d0_gate2a_v1.json"
DEFAULT_DIR = ROOT / "v2/results/d0_gate2a"


class LimitReached(RuntimeError):
    """Intentional dry-run interruption; leaves the simulate stage resumable."""


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def seeded_rng(seed: int, *parts: Any) -> random.Random:
    key = "\x1f".join([str(seed), *(str(p) for p in parts)]).encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:16], "big"))


def load_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "seed", "episodes_per_family", "turn_budget",
                "prior_p_ai", "bootstrap_replicates", "default_discrimination",
                "fixed_order", "response_categories", "response_renderings",
                "questions", "families", "gate"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    qids = [q["id"] for q in config["questions"]]
    if len(qids) != len(set(qids)) or len(qids) != 12:
        raise ValueError("question IDs must be 12 unique values")
    if set(config["fixed_order"]) - set(qids):
        raise ValueError("fixed order contains unknown questions")
    if tuple(config["response_categories"]) != CATEGORIES:
        raise ValueError("response categories differ from frozen schema")
    if not 0 < float(config["prior_p_ai"]) < 1:
        raise ValueError("prior must be strictly between zero and one")
    if int(config["turn_budget"]) > len(qids):
        raise ValueError("turn budget exceeds question bank")
    families = config["families"]
    if len(families) != 16 or len({f["id"] for f in families}) != 16:
        raise ValueError("expected 16 unique scenario families")
    if sum(f["split"] == "development" for f in families) != 8 or sum(
            f["split"] == "heldout" for f in families) != 8:
        raise ValueError("expected eight families per split")
    for split in ("development", "heldout"):
        banks = config["response_renderings"][split]
        for category in CATEGORIES:
            if not banks.get(category):
                raise ValueError(f"empty rendering bank: {split}/{category}")
    for family in families:
        if set(family["signals"]) - set(qids):
            raise ValueError(f"unknown signal question in {family['id']}")
        for qid in qids:
            probs = likelihoods(config, family, qid)
            for typ in ("human", "ai"):
                if not math.isclose(sum(probs[typ].values()), 1.0, abs_tol=1e-15):
                    raise ValueError(f"probabilities do not sum to one: {family['id']}/{qid}")
                if any(not 0 < p < 1 for p in probs[typ].values()):
                    raise ValueError(f"non-strict probability: {family['id']}/{qid}")
    dev_text = {q["development"] for q in config["questions"]} | {
        x for bank in config["response_renderings"]["development"].values() for x in bank}
    hold_text = {q["heldout"] for q in config["questions"]} | {
        x for bank in config["response_renderings"]["heldout"].values() for x in bank}
    if dev_text & hold_text:
        raise ValueError("development and held-out surfaces overlap")


def likelihoods(config: dict[str, Any], family: dict[str, Any], qid: str) -> dict[str, dict[str, float]]:
    n = float(family["neutral_mass"])
    d = float(family["signals"].get(qid, config["default_discrimination"]))
    same, opposite = (1 - n) * (1 + d) / 2, (1 - n) * (1 - d) / 2
    return {
        "human": {"human_cue": same, "neutral": n, "ai_cue": opposite},
        "ai": {"human_cue": opposite, "neutral": n, "ai_cue": same},
    }


def entropy(p_ai: float) -> float:
    if p_ai <= 0 or p_ai >= 1:
        return 0.0
    return -p_ai * math.log(p_ai) - (1 - p_ai) * math.log(1 - p_ai)


def posterior(p_ai: float, p_response_ai: float, p_response_human: float) -> float:
    numerator = p_ai * p_response_ai
    denominator = numerator + (1 - p_ai) * p_response_human
    if denominator <= 0:
        raise ValueError("response has zero predictive probability")
    return numerator / denominator


def exact_eig(config: dict[str, Any], family: dict[str, Any], qid: str, p_ai: float) -> float:
    probs = likelihoods(config, family, qid)
    expected = 0.0
    for category in CATEGORIES:
        pa, ph = probs["ai"][category], probs["human"][category]
        predictive = p_ai * pa + (1 - p_ai) * ph
        expected += predictive * entropy(posterior(p_ai, pa, ph))
    return entropy(p_ai) - expected


def sample_categorical(probabilities: dict[str, float], rng: random.Random) -> str:
    u, running = rng.random(), 0.0
    for category in CATEGORIES:
        running += probabilities[category]
        if u <= running:
            return category
    return CATEGORIES[-1]


def episode_labels(config: dict[str, Any], family: dict[str, Any]) -> list[str]:
    count = int(config["episodes_per_family"])
    if count % 2:
        raise ValueError("episodes_per_family must be even")
    labels = ["human"] * (count // 2) + ["ai"] * (count // 2)
    seeded_rng(config["seed"], "labels", family["id"]).shuffle(labels)
    return labels


def response_schedule(config: dict[str, Any], family: dict[str, Any], episode: int,
                      true_type: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    split = family["split"]
    for question in config["questions"]:
        qid = question["id"]
        probs = likelihoods(config, family, qid)
        category = sample_categorical(
            probs[true_type], seeded_rng(config["seed"], "response", family["id"], episode, qid))
        bank = config["response_renderings"][split][category]
        text = bank[seeded_rng(config["seed"], "render", family["id"], episode, qid).randrange(len(bank))]
        result[qid] = {"category": category, "text": text,
                        "text_probability": probs[true_type][category] / len(bank)}
    return result


def select_question(policy: str, config: dict[str, Any], family: dict[str, Any],
                    episode: int, turn: int, unused: list[str], p_ai: float) -> str:
    if policy == "fixed":
        return next(qid for qid in config["fixed_order"] if qid in unused)
    if policy == "random":
        ordered = sorted(unused)
        return ordered[seeded_rng(config["seed"], "policy", policy, family["id"], episode, turn).randrange(len(ordered))]
    if policy == "bed_eig":
        return min(unused, key=lambda qid: (-exact_eig(config, family, qid, p_ai), qid))
    if policy == "uot_sample":
        scores: dict[str, float] = {}
        for qid in unused:
            probs = likelihoods(config, family, qid)
            rh = sample_categorical(probs["human"], seeded_rng(
                config["seed"], "uot", family["id"], episode, turn, qid, "human"))
            ra = sample_categorical(probs["ai"], seeded_rng(
                config["seed"], "uot", family["id"], episode, turn, qid, "ai"))
            scores[qid] = entropy(p_ai) if rh != ra else 0.0
        return min(unused, key=lambda qid: (-scores[qid], qid))
    raise ValueError(f"unknown policy {policy}")


def trajectory_id(family_id: str, episode: int, policy: str) -> str:
    return f"{family_id}:{episode:03d}:{policy}"


def simulate_trajectory(config: dict[str, Any], family: dict[str, Any], episode: int,
                        policy: str) -> dict[str, Any]:
    labels = episode_labels(config, family)
    true_type = labels[episode]
    schedule = response_schedule(config, family, episode, true_type)
    qmap = {q["id"]: q for q in config["questions"]}
    unused = [q["id"] for q in config["questions"]]
    p_ai = float(config["prior_p_ai"])
    ledger = []
    for turn in range(1, int(config["turn_budget"]) + 1):
        qid = select_question(policy, config, family, episode, turn, unused, p_ai)
        unused.remove(qid)
        observed = schedule[qid]
        probs = likelihoods(config, family, qid)
        category = observed["category"]
        pa, ph = probs["ai"][category], probs["human"][category]
        predictive = p_ai * pa + (1 - p_ai) * ph
        after = posterior(p_ai, pa, ph)
        ledger.append({
            "turn": turn, "question_id": qid, "strategy_code": qmap[qid]["strategy_code"],
            "question_text": qmap[qid][family["split"]], "response_category": category,
            "response_text": observed["text"], "prior_p_ai": p_ai,
            "posterior_p_ai": after, "likelihood_human": ph, "likelihood_ai": pa,
            "predictive_probability": predictive, "emitted_text_probability": observed["text_probability"],
            "exact_eig": exact_eig(config, family, qid, p_ai),
            "realized_entropy_reduction": entropy(p_ai) - entropy(after),
            "probability_authority": "known_synthetic_distribution",
        })
        p_ai = after
    correct_probability = p_ai if true_type == "ai" else 1 - p_ai
    return {
        "schema_version": config["schema_version"], "config_sha256": canonical_hash(config),
        "trajectory_id": trajectory_id(family["id"], episode, policy),
        "family_id": family["id"], "split": family["split"], "surface": family["split"],
        "episode": episode, "true_type": true_type, "policy": policy,
        "schedule_sha256": canonical_hash(schedule), "turn_budget": config["turn_budget"],
        "ledger": ledger, "final_p_ai": p_ai,
        "final_log_loss": -math.log(correct_probability),
        "final_brier": (p_ai - (1 if true_type == "ai" else 0)) ** 2,
        "final_correct": int((p_ai >= 0.5) == (true_type == "ai")),
    }


def expected_ids(config: dict[str, Any]) -> list[str]:
    return [trajectory_id(f["id"], episode, policy) for f in config["families"]
            for episode in range(int(config["episodes_per_family"])) for policy in POLICIES]


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}")
    return rows


def validate_rows(config: dict[str, Any], rows: list[dict[str, Any]], require_complete: bool = True,
                  reproduce: bool = True) -> dict[str, Any]:
    expected = expected_ids(config)
    expected_set = set(expected)
    seen: set[str] = set()
    failures: list[str] = []
    family_map = {f["id"]: f for f in config["families"]}
    config_hash = canonical_hash(config)
    for row in rows:
        rid = row.get("trajectory_id")
        if rid in seen:
            failures.append(f"duplicate:{rid}")
        seen.add(rid)
        if rid not in expected_set:
            failures.append(f"unexpected:{rid}")
            continue
        if row.get("config_sha256") != config_hash:
            failures.append(f"config_hash:{rid}")
        ledger = row.get("ledger", [])
        qids = [turn.get("question_id") for turn in ledger]
        if len(ledger) != int(config["turn_budget"]) or len(qids) != len(set(qids)):
            failures.append(f"budget_or_repeat:{rid}")
        prior = float(config["prior_p_ai"])
        for turn in ledger:
            calculated = posterior(prior, turn["likelihood_ai"], turn["likelihood_human"])
            if abs(turn["prior_p_ai"] - prior) > config["gate"]["bayes_tolerance"] or abs(
                    turn["posterior_p_ai"] - calculated) > config["gate"]["bayes_tolerance"]:
                failures.append(f"bayes:{rid}:turn{turn.get('turn')}")
            if turn.get("probability_authority") != "known_synthetic_distribution":
                failures.append(f"authority:{rid}:turn{turn.get('turn')}")
            prior = calculated
        if reproduce:
            regenerated = simulate_trajectory(config, family_map[row["family_id"]], row["episode"], row["policy"])
            if canonical_hash(regenerated) != canonical_hash(row):
                failures.append(f"reproduction:{rid}")
    if require_complete and seen != expected_set:
        failures.append(f"coverage:missing={len(expected_set-seen)},extra={len(seen-expected_set)}")
    return {"valid": not failures, "row_count": len(rows), "unique_ids": len(seen),
            "expected_rows": len(expected), "failure_count": len(failures), "failures": failures[:100]}


def run_benchmark(config: dict[str, Any], output: Path, state: Path, resume: bool,
                  limit: int | None = None) -> None:
    rows = read_rows(output) if output.exists() else []
    initial = validate_rows(config, rows, require_complete=False, reproduce=True)
    if not initial["valid"]:
        raise SystemExit(f"existing output is invalid: {initial['failures'][:5]}")
    if rows and not resume:
        raise SystemExit(f"output exists ({output}); pass --resume or choose another path")
    done = {row["trajectory_id"] for row in rows}
    family_map = {f["id"]: f for f in config["families"]}
    metadata = {"schema_version": config["schema_version"], "config_sha256": canonical_hash(config),
                "seed": config["seed"], "output": str(output),
                "resume_command": f"{sys.executable} {Path(__file__)} run --resume"}
    run = Run(state, resume=resume, config=metadata)
    made = 0
    try:
        with run.stage("simulate") as stage:
            if stage.skipped and len(done) != len(expected_ids(config)):
                raise SystemExit("state says complete but output coverage is incomplete")
            stage.count = len(done)
            if not stage.skipped:
                for rid in expected_ids(config):
                    if rid in done:
                        continue
                    family_id, episode_text, policy = rid.rsplit(":", 2)
                    row = simulate_trajectory(config, family_map[family_id], int(episode_text), policy)
                    append_jsonl(output, row)
                    done.add(rid)
                    stage.tick(last_trajectory_id=rid, output_rows=len(done))
                    made += 1
                    if limit is not None and made >= limit:
                        stage.note(stopped_by_limit=True)
                        raise LimitReached(f"intentional stop after {made} new rows")
    except LimitReached:
        return
    print(json.dumps(validate_rows(config, read_rows(output), require_complete=True), indent=2))


def quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    location = (len(ordered) - 1) * p
    lo, hi = math.floor(location), math.ceil(location)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (location - lo)


def describe(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "q1": quantile(values, .25), "q3": quantile(values, .75),
            "min": min(values), "max": max(values)}


def cluster_interval(family_means: dict[str, float], replicates: int, seed: int,
                     comparator: str, split: str) -> list[float]:
    values = list(family_means.values())
    rng = seeded_rng(seed, "bootstrap", comparator, split)
    draws = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(replicates)]
    return [quantile(draws, .025), quantile(draws, .975)]


def evaluate_gate(config: dict[str, Any], comparisons: dict[str, Any], integrity: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not integrity["valid"]:
        return {"decision": "FAIL", "reasons": ["trajectory integrity failure"]}
    held = comparisons["heldout"]
    dev = comparisons["development"]
    nonpositive = [c for c in config["gate"]["comparators"] if held[c]["episode_delta"]["mean"] <= 0]
    if nonpositive:
        return {"decision": "FAIL", "reasons": [f"non-positive held-out mean versus {c}" for c in nonpositive]}
    for comparator in config["gate"]["comparators"]:
        result = held[comparator]
        if result["episode_delta"]["mean"] < config["gate"]["min_mean_log_loss_improvement_nats"]:
            reasons.append(f"held-out mean below 0.05 versus {comparator}")
        if result["cluster_bootstrap_95"][0] <= config["gate"]["bootstrap_lower_must_exceed"]:
            reasons.append(f"cluster interval includes zero versus {comparator}")
        if result["positive_family_count"] < config["gate"]["min_positive_families"]:
            reasons.append(f"fewer than 7 positive held-out families versus {comparator}")
        if dev[comparator]["episode_delta"]["mean"] <= 0:
            reasons.append(f"development direction non-positive versus {comparator}")
    return {"decision": "INCONCLUSIVE" if reasons else "PASS", "reasons": reasons or ["all frozen criteria met"]}


def analyze(config: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    integrity = validate_rows(config, rows, require_complete=True, reproduce=True)
    by_key = {(r["family_id"], r["episode"], r["policy"]): r for r in rows}
    summaries: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for split in ("development", "heldout"):
        split_rows = [r for r in rows if r["split"] == split]
        summaries[split] = {}
        for policy in POLICIES:
            subset = [r for r in split_rows if r["policy"] == policy]
            summaries[split][policy] = {
                "episodes": len(subset), "final_log_loss": describe([r["final_log_loss"] for r in subset]),
                "final_brier": describe([r["final_brier"] for r in subset]),
                "accuracy": statistics.fmean(r["final_correct"] for r in subset),
                "mean_selected_exact_eig": statistics.fmean(
                    turn["exact_eig"] for r in subset for turn in r["ledger"]),
            }
        comparisons[split] = {}
        families = [f["id"] for f in config["families"] if f["split"] == split]
        for comparator in ("random", "fixed", "uot_sample"):
            deltas_by_family: dict[str, list[float]] = {f: [] for f in families}
            for family in families:
                for episode in range(config["episodes_per_family"]):
                    delta = by_key[(family, episode, comparator)]["final_log_loss"] - by_key[
                        (family, episode, "bed_eig")]["final_log_loss"]
                    deltas_by_family[family].append(delta)
            family_means = {f: statistics.fmean(v) for f, v in deltas_by_family.items()}
            all_deltas = [d for values in deltas_by_family.values() for d in values]
            comparisons[split][comparator] = {
                "estimand": f"log_loss({comparator}) - log_loss(bed_eig)",
                "episode_delta": describe(all_deltas), "family_means": family_means,
                "family_mean_range": [min(family_means.values()), max(family_means.values())],
                "positive_family_count": sum(v > 0 for v in family_means.values()),
                "cluster_bootstrap_95": cluster_interval(family_means, config["bootstrap_replicates"],
                                                         config["seed"], comparator, split),
            }
    gate = evaluate_gate(config, comparisons, integrity)
    result = {"schema_version": config["schema_version"], "config_sha256": canonical_hash(config),
              "integrity": integrity, "summaries": summaries, "comparisons": comparisons,
              "gate_2a": gate, "claim_boundary": "Synthetic mechanics only; no real-world validity claim."}
    inspection_rows = []
    held_rows = [r for r in rows if r["split"] == "heldout"]
    for policy in POLICIES:
        for true_type in ("human", "ai"):
            inspection_rows.append(next(r for r in held_rows if r["policy"] == policy and r["true_type"] == true_type))
    worst = sorted(
        (r for r in held_rows if r["policy"] == "bed_eig"),
        key=lambda r: r["final_log_loss"], reverse=True)[:8]
    inspection = {"selection": "first held-out human and AI trajectory per policy, plus eight worst BED losses",
                  "representative": inspection_rows, "worst_bed": worst}
    return result, inspection


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate_p = sub.add_parser("validate")
    validate_p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_p = sub.add_parser("run")
    run_p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_p.add_argument("--output", type=Path, default=DEFAULT_DIR / "trajectories.jsonl")
    run_p.add_argument("--state", type=Path, default=DEFAULT_DIR / "run_state.json")
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument("--limit", type=int)
    analysis_p = sub.add_parser("analyze")
    analysis_p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    analysis_p.add_argument("--input", type=Path, default=DEFAULT_DIR / "trajectories.jsonl")
    analysis_p.add_argument("--output", type=Path, default=DEFAULT_DIR / "result.json")
    analysis_p.add_argument("--inspection", type=Path, default=DEFAULT_DIR / "inspection.json")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate":
        print(json.dumps({"valid": True, "config_sha256": canonical_hash(config),
                          "expected_rows": len(expected_ids(config))}, indent=2))
    elif args.command == "run":
        run_benchmark(config, args.output, args.state, args.resume, args.limit)
    else:
        result, inspection = analyze(config, read_rows(args.input))
        atomic_write_json(args.output, result)
        atomic_write_json(args.inspection, inspection)
        print(json.dumps({"gate_2a": result["gate_2a"], "integrity": result["integrity"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
