#!/usr/bin/env python3
"""Provenance-locked out-of-corpus detector bridge and nested calibration.

The expensive ``score`` command uses the pinned OpenAI GPT-2 detector locally
and appends one text-free record per dialogue. The stdlib-only ``analyze``
command performs component-nested Platt calibration and corrected Track A
inference. Neither command evaluates the frozen test split.

Examples:
  .bridge-venv/bin/python -u v2/scripts/track_a_bridge.py score --limit 2 \
      --output /tmp/bridge_dry.jsonl --state /tmp/bridge_dry.state.json
  .bridge-venv/bin/python -u v2/scripts/track_a_bridge.py score --resume
  venv/bin/python v2/scripts/track_a_bridge.py analyze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_canonical  # noqa: E402
import track_a_a0 as a0  # noqa: E402
import track_a_rung4 as rung4  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "v2" / "results" / "bridge"
DEFAULT_SCORES = OUT_DIR / "openai_gpt2_detector_raw.jsonl"
DEFAULT_STATE = OUT_DIR / "openai_gpt2_detector_run_state.json"
DEFAULT_RESULT = OUT_DIR / "openai_gpt2_detector_bridge.json"
DEFAULT_CACHE = REPO_ROOT / "v2" / ".cache" / "huggingface"

MODEL_ID = "openai-community/roberta-base-openai-detector"
MODEL_REVISION = "6cba99c003b711c7fe94f8a3aa2be35a792cb6fa"
WEIGHT_SHA256 = "3abd6d2b005f5876b945cb5b68ddde04f6e28fbd9c5d6dc5adfb06ba647e0546"
MODEL_FILES = (
    "config.json", "merges.txt", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "vocab.json",
)
MAX_LENGTH = 512
L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
SEED = 20260822


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def witness_text(dialogue: dict) -> str:
    return "\n".join(
        turn["content"] for turn in dialogue["turns"]
        if turn["role"] == "W" and turn["content"]
    )


def load_bridge_data() -> tuple[list[dict], list[dict], dict, dict[str, dict]]:
    """Load canonical records, then expose only non-empty train+dev games.

    The canonical files are physically shared across splits, so their JSON lines
    are parsed, but test examples are rejected before text extraction and never
    enter scoring, fitting, metric computation, or result artifacts.
    """
    dialogues, games, manifest = build_canonical.load()
    # Also exclude redaction-erased sides: they can report witness-message rows
    # while every released content string is empty (for example game 2809).
    empty_games = (a0.empty_witness_games(dialogues)
                   | {d["game_id"] for d in dialogues if not witness_text(d)})
    allowed_games = {
        g["game_id"] for g in games
        if g["split"] in ("train", "dev") and g["game_id"] not in empty_games
    }
    selected_games = [g for g in games if g["game_id"] in allowed_games]
    selected_dialogues = [d for d in dialogues if d["game_id"] in allowed_games]
    if any(d["split"] == "test" for d in selected_dialogues):
        raise RuntimeError("test-split dialogue reached bridge selection")
    by_game: dict[str, dict[str, dict]] = defaultdict(dict)
    for dialogue in selected_dialogues:
        text = witness_text(dialogue)
        if not text:
            raise RuntimeError(f"empty witness text survived policy: {dialogue['example_id']}")
        by_game[dialogue["game_id"]][dialogue["conversation_label"]] = dialogue
    for game in selected_games:
        if set(by_game[game["game_id"]]) != {"A", "B"}:
            raise RuntimeError(f"game lacks exactly two sides: {game['game_id']}")
    return selected_dialogues, selected_games, manifest, dict(by_game)


def scoring_records(limit: int | None = None) -> list[dict]:
    dialogues, _, _, _ = load_bridge_data()
    rows = []
    for dialogue in sorted(dialogues, key=lambda d: d["example_id"]):
        text = witness_text(dialogue)
        rows.append({
            "example_id": dialogue["example_id"],
            "game_id": dialogue["game_id"],
            "conversation_label": dialogue["conversation_label"],
            "split": dialogue["split"],
            "component": dialogue["component"],
            "is_human": dialogue["is_human"],
            "text": text,
            "text_sha256": sha256_bytes(text.encode("utf-8")),
        })
    return rows[:limit] if limit is not None else rows


def read_score_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = row["example_id"]
        if example_id in rows:
            raise RuntimeError(f"duplicate score {example_id} at {path}:{line_no}")
        rows[example_id] = row
    return rows


def validate_existing(existing: dict[str, dict], expected: dict[str, dict]) -> None:
    unknown = sorted(set(existing) - set(expected))
    if unknown:
        raise RuntimeError(f"score file contains unexpected IDs: {unknown[:3]}")
    for example_id, row in existing.items():
        source = expected[example_id]
        if row.get("text_sha256") != source["text_sha256"]:
            raise RuntimeError(f"text hash changed for {example_id}")
        if row.get("model_revision") != MODEL_REVISION:
            raise RuntimeError(f"model revision changed for {example_id}")
        prob = row.get("p_ai")
        if not isinstance(prob, (int, float)) or not math.isfinite(prob):
            raise RuntimeError(f"invalid probability for {example_id}")


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        try:
            torch.ones(1).to("mps")
            return "mps"
        except RuntimeError as exc:
            print(f"MPS advertised but unusable; falling back to CPU: {exc}",
                  file=sys.stderr, flush=True)
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def command_score(args) -> int:
    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "bridge dependencies missing; run scripts/setup_bridge_env.sh first"
        ) from exc

    output, state = Path(args.output), Path(args.state)
    expected_rows = scoring_records(args.limit)
    expected = {r["example_id"]: r for r in expected_rows}
    existing = read_score_rows(output)
    if existing and not args.resume:
        raise SystemExit(f"{output} exists; pass --resume after inspecting it")
    validate_existing(existing, expected)

    prior_state = None
    if state.exists():
        try:
            prior_state = json.loads(state.read_text())
        except json.JSONDecodeError:
            raise RuntimeError(f"run state is corrupt: {state}")
    if (args.resume and len(existing) == len(expected_rows) and prior_state
            and prior_state.get("status") == "complete"):
        observed = sha256_file(output)
        if prior_state.get("scores_sha256") != observed:
            raise RuntimeError("completed state/output hash mismatch")
        print(f"already complete: {len(existing)}/{len(expected_rows)}; state preserved",
              flush=True)
        return 0

    config = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weight_sha256": WEIGHT_SHA256,
        "max_length": MAX_LENGTH,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "output": str(output),
        "expected_examples": len(expected_rows),
    }
    atomic_json(state, {
        "status": "initializing", "config": config,
        "processed": len(existing), "updated_at": time.time(),
        "resume_command": (
            f"{sys.executable} -u v2/scripts/track_a_bridge.py score --resume "
            f"--output {output} --state {state}"
        ),
    })

    snapshot = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=list(MODEL_FILES),
        cache_dir=str(Path(args.cache)),
    ))
    weights = snapshot / "model.safetensors"
    observed_weight_hash = sha256_file(weights)
    if observed_weight_hash != WEIGHT_SHA256:
        raise RuntimeError(
            f"weight hash mismatch: {observed_weight_hash} != {WEIGHT_SHA256}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot), local_files_only=True, use_safetensors=True,
    )
    labels = {int(k): v for k, v in model.config.id2label.items()}
    if labels != {0: "Fake", 1: "Real"}:
        raise RuntimeError(f"unexpected label mapping: {labels}")
    device = choose_device(torch, args.device)
    model.to(device)
    model.eval()
    print(f"model={MODEL_ID}@{MODEL_REVISION} device={device}", flush=True)
    print(f"expected={len(expected_rows)} already={len(existing)} output={output}", flush=True)

    pending = [row for row in expected_rows if row["example_id"] not in existing]
    started = time.time()
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        encoded = tokenizer(
            [row["text"] for row in batch], padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        token_counts = encoded["attention_mask"].sum(dim=1).tolist()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)[:, 0]
        for row, p_ai, n_tokens in zip(batch, probabilities.detach().cpu().tolist(), token_counts):
            if not math.isfinite(p_ai):
                raise RuntimeError(f"non-finite probability for {row['example_id']}")
            record = {
                "example_id": row["example_id"],
                "game_id": row["game_id"],
                "conversation_label": row["conversation_label"],
                "split": row["split"],
                "component": row["component"],
                "text_sha256": row["text_sha256"],
                "p_ai": p_ai,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "weight_sha256": observed_weight_hash,
                "token_count_with_special_tokens": int(n_tokens),
                "truncated": int(n_tokens) >= MAX_LENGTH,
            }
            append_jsonl(output, record)
            existing[row["example_id"]] = record
            atomic_json(state, {
                "status": "running", "config": config,
                "processed": len(existing), "last_example_id": row["example_id"],
                "updated_at": time.time(), "seconds": time.time() - started,
                "device": device,
            })
        print(f"scored {len(existing)}/{len(expected_rows)}", flush=True)

    atomic_json(state, {
        "status": "complete", "config": config, "processed": len(existing),
        "updated_at": time.time(), "seconds": time.time() - started,
        "device": device, "torch": torch.__version__,
        "transformers": transformers.__version__,
        "scores_sha256": sha256_file(output),
    })
    return 0


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def fit_platt(rows: list[dict], l2: float) -> dict:
    # Stable summation/order makes the frozen result byte-reproducible even if a
    # caller constructs the same fold in a different list order.
    rows = sorted(rows, key=lambda r: (
        r["component"], r.get("example_id", ""), r["label"], r["p_ai"]
    ))
    raw = [logit(r["p_ai"]) for r in rows]
    mean = sum(raw) / len(raw)
    variance = sum((x - mean) ** 2 for x in raw) / len(raw)
    sd = math.sqrt(variance) if variance > 1e-12 else 1.0
    x = [[(value - mean) / sd] for value in raw]
    y = [r["label"] for r in rows]
    return {"mean": mean, "sd": sd, "beta": a0.fit_dense_logistic(x, y, l2=l2)}


def predict_platt(model: dict, p_ai: float) -> float:
    x = (logit(p_ai) - model["mean"]) / model["sd"]
    return a0.predict_dense(model["beta"], [x])


def select_l2_nested(rows: list[dict]) -> tuple[float, dict[str, float]]:
    rows = sorted(rows, key=lambda r: (
        r["component"], r.get("example_id", ""), r["label"], r["p_ai"]
    ))
    components = sorted({r["component"] for r in rows})
    if len(components) < 2:
        raise RuntimeError("nested calibration needs at least two training components")
    scores = {}
    for l2 in L2_GRID:
        losses = []
        for component in components:
            train = [r for r in rows if r["component"] != component]
            held = [r for r in rows if r["component"] == component]
            if not train or not held:
                continue
            model = fit_platt(train, l2)
            losses.extend((predict_platt(model, r["p_ai"]) - r["label"]) ** 2 for r in held)
        scores[str(l2)] = sum(losses) / len(losses)
    selected = min(L2_GRID, key=lambda value: (scores[str(value)], -value))
    return selected, scores


def dialogue_rows(games: list[dict], by_game: dict, raw: dict[str, dict]) -> list[dict]:
    rows = []
    for game in games:
        for side in ("A", "B"):
            dialogue = by_game[game["game_id"]][side]
            score = raw[dialogue["example_id"]]
            rows.append({
                "example_id": dialogue["example_id"],
                "component": game["component"],
                "p_ai": score["p_ai"],
                "label": 0 if dialogue["is_human"] else 1,
            })
    return rows


def nested_calibrated_predictions(train_games: list[dict], held_games: list[dict],
                                  by_game: dict, raw: dict[str, dict]) -> tuple[dict, dict]:
    train_rows = dialogue_rows(train_games, by_game, raw)
    selected, inner_scores = select_l2_nested(train_rows)
    model = fit_platt(train_rows, selected)
    predictions = {}
    for row in dialogue_rows(held_games, by_game, raw):
        predictions[row["example_id"]] = predict_platt(model, row["p_ai"])
    return predictions, {
        "selected_l2": selected,
        "inner_brier": inner_scores,
        "platt_intercept": model["beta"][0],
        "platt_slope_standardized_logit": model["beta"][1],
        "platt_slope_raw_logit": model["beta"][1] / model["sd"],
    }


def crossfit_people(games: list[dict], by_game: dict, raw: dict[str, dict]) -> tuple[dict, list[dict]]:
    predictions, diagnostics = {}, []
    for component in sorted({g["component"] for g in games}):
        train = [g for g in games if g["component"] != component]
        held = [g for g in games if g["component"] == component]
        cell, diag = nested_calibrated_predictions(train, held, by_game, raw)
        predictions.update(cell)
        diagnostics.append({"held_component": component, "n_train_games": len(train),
                            "n_held_games": len(held), **diag})
    return predictions, diagnostics


def crossfit_persona(games: list[dict], by_game: dict, raw: dict[str, dict],
                     train_persona: str, eval_persona: str) -> tuple[dict, list[dict], list[dict]]:
    eval_games = [g for g in games if rung4.PERSONA_OF.get(g["witness_system"]) == eval_persona]
    predictions, diagnostics = {}, []
    for component in sorted({g["component"] for g in eval_games}):
        held = [g for g in eval_games if g["component"] == component]
        train = [g for g in games
                 if rung4.PERSONA_OF.get(g["witness_system"]) == train_persona
                 and g["component"] != component]
        cell, diag = nested_calibrated_predictions(train, held, by_game, raw)
        predictions.update(cell)
        diagnostics.append({"held_component": component, "n_train_games": len(train),
                            "n_held_games": len(held), **diag})
    return predictions, eval_games, diagnostics


def score_payload(predictions: dict, games: list[dict], by_game: dict,
                  bootstrap: int) -> dict:
    scored = a0.score_games(predictions, games, by_game)
    point = a0.metrics_from(scored, list(range(len(games))))
    intervals = a0.bootstrap_intervals({"external": scored}, games, bootstrap, SEED)
    return {
        "n_games": len(games),
        "point_estimates": {k: round(v, 6) for k, v in point.items()},
        "dialogue_ece": round(a0.ece(scored["dialogue_probs"], scored["dialogue_labels"]), 6),
        "intervals": intervals,
    }


def gate_decision(transfers: dict) -> dict:
    details, hard_fail, uncertain = {}, False, False
    for direction, result in transfers.items():
        point = result["nested_calibrated"]["point_estimates"]
        intervals = result["nested_calibrated"]["intervals"]
        participant = intervals["participant"]["detectors"]["external"]["game_accuracy"]
        component = intervals["component"]["detectors"]["external"]["game_accuracy"]
        fail = point["game_accuracy"] <= 0.5 or point["dialogue_brier"] >= 0.25
        crosses = participant["lo"] <= 0.5 or component["lo"] <= 0.5
        hard_fail = hard_fail or fail
        uncertain = uncertain or crosses
        details[direction] = {
            "game_accuracy": point["game_accuracy"],
            "dialogue_brier": point["dialogue_brier"],
            "participant_lower": participant["lo"],
            "component_lower": component["lo"],
            "point_threshold_failed": fail,
            "interval_threshold_not_cleared": crosses,
        }
    verdict = "FAIL" if hard_fail else ("INCONCLUSIVE" if uncertain else "PASS")
    return {"verdict": verdict, "directions": details,
            "test_split": "UNTOUCHED; no test dialogue scored"}


def command_analyze(args) -> int:
    dialogues, games, manifest, by_game = load_bridge_data()
    expected = {r["example_id"]: r for r in scoring_records()}
    raw = read_score_rows(Path(args.scores))
    validate_existing(raw, expected)
    missing = sorted(set(expected) - set(raw))
    if missing:
        raise SystemExit(f"score file incomplete: {len(missing)} missing; first={missing[0]}")

    raw_predictions = {example_id: row["p_ai"] for example_id, row in raw.items()}
    people_predictions, people_diag = crossfit_people(games, by_game, raw)
    payload = {
        "generated_by": "v2/scripts/track_a_bridge.py analyze",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "v2/BRIDGE_PROTOCOL.md (frozen before detector run)",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                  "weight_sha256": WEIGHT_SHA256},
        "canonical": {key: manifest[key] for key in
                      ("source_revision", "split_sha256", "dialogues_sha256", "games_sha256")},
        "scores_sha256": sha256_file(Path(args.scores)),
        "test_split": "UNTOUCHED; excluded before scoring, fitting and evaluation",
        "exclusions": {"empty_witness_games": 911 - len(games) if len(games) <= 911 else None,
                       "retained_games": len(games), "retained_dialogues": len(dialogues)},
        "raw_zero_shot_people_only": score_payload(raw_predictions, games, by_game, args.bootstrap),
        "nested_calibrated_people_only": score_payload(people_predictions, games, by_game, args.bootstrap),
        "people_calibration_diagnostics": people_diag,
        "persona_transfer": {},
        "calibration": {"method": "Platt on raw-score logit",
                        "l2_grid": list(L2_GRID),
                        "selection": "inner leave-one-component-out dialogue Brier; larger l2 wins ties"},
        "inference": {"additive": "dyadic participant-cluster sandwich",
                      "nonadditive": "connected-component percentile bootstrap",
                      "bootstrap_replicates": args.bootstrap, "seed": SEED},
    }
    for train_persona, eval_persona in (("minimal", "quinn"), ("quinn", "minimal")):
        key = f"{train_persona}->{eval_persona}"
        predictions, eval_games, diagnostics = crossfit_persona(
            games, by_game, raw, train_persona, eval_persona)
        eval_raw = {by_game[g["game_id"]][side]["example_id"]:
                    raw_predictions[by_game[g["game_id"]][side]["example_id"]]
                    for g in eval_games for side in ("A", "B")}
        payload["persona_transfer"][key] = {
            "raw_zero_shot": score_payload(eval_raw, eval_games, by_game, args.bootstrap),
            "nested_calibrated": score_payload(predictions, eval_games, by_game, args.bootstrap),
            "calibration_diagnostics": diagnostics,
        }
    payload["bridge_gate"] = gate_decision(payload["persona_transfer"])
    output = Path(args.output)
    atomic_json(output, payload)
    print(json.dumps(payload["bridge_gate"], indent=2))
    print(f"wrote {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="run pinned detector with durable output")
    score.add_argument("--output", default=str(DEFAULT_SCORES))
    score.add_argument("--state", default=str(DEFAULT_STATE))
    score.add_argument("--cache", default=str(DEFAULT_CACHE))
    score.add_argument("--batch-size", type=int, default=16)
    score.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    score.add_argument("--limit", type=int, default=None)
    score.add_argument("--resume", action="store_true")
    score.set_defaults(func=command_score)
    analyze = sub.add_parser("analyze", help="nested calibration and corrected inference")
    analyze.add_argument("--scores", default=str(DEFAULT_SCORES))
    analyze.add_argument("--output", default=str(DEFAULT_RESULT))
    analyze.add_argument("--bootstrap", type=int, default=1000)
    analyze.set_defaults(func=command_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BaseException as exc:
        if args.command == "score":
            state_path = Path(args.state)
            prior = {}
            if state_path.exists():
                try:
                    prior = json.loads(state_path.read_text())
                except json.JSONDecodeError:
                    prior = {"prior_state_corrupt": True}
            prior.update({
                "status": "failed",
                "failed_at": time.time(),
                "error": f"{type(exc).__name__}: {exc}",
                "resume_command": (
                    f"{sys.executable} -u v2/scripts/track_a_bridge.py score "
                    f"--resume --output {args.output} --state {args.state} "
                    f"--cache {args.cache} --device auto"
                ),
            })
            atomic_json(state_path, prior)
        raise


if __name__ == "__main__":
    sys.exit(main())
