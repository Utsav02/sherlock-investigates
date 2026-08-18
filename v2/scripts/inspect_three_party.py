#!/usr/bin/env python3
"""
Measure the Jones & Bergen 2025 three-party Turing test release.

Stage A step 4 of `v2/experiment_design.md`: report ACTUAL numbers from the
downloaded files, never numbers quoted from the paper. Everything printed here
is computed from `v2/data/sources/jones_bergen_2025/`.

The source tree is read-only: this script opens files for reading only and
writes solely to `v2/results/stage_a/`.

Stdlib only (csv/json/statistics), so it runs under the repo venv's `python`.

Usage:
    venv/bin/python v2/scripts/inspect_three_party.py            # data/ (main study)
    venv/bin/python v2/scripts/inspect_three_party.py --subdir 15_mins
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "v2" / "data" / "sources" / "jones_bergen_2025"
OUT_DIR = REPO_ROOT / "v2" / "results" / "stage_a"

# csv fields in this release include whole transcripts; the default 128K limit
# is comfortable but raise it so a long field can never silently truncate.
csv.field_size_limit(10_000_000)

WORD_RE = re.compile(r"\S+")

# Conservative PII sweeps over free text. These are screens that flag rows for
# human review; they are not a claim that the text is clean or that every hit
# is real PII.
PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "url": re.compile(r"https?://\S+|\bwww\.\S+"),
    "phone_like": re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"),
    "anonymization_placeholder": re.compile(r"\[[A-Z_]{3,}\]|<[A-Z_]{3,}>"),
}


# ---------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_v2_inspect.py)
# ---------------------------------------------------------------------------

def summarize(values: list[float]) -> dict:
    """Distribution summary: n, mean, sd, min, quartiles, median, p90, max."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "sd": round(statistics.stdev(ordered), 3) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p25": percentile(ordered, 25),
        "median": percentile(ordered, 50),
        "p75": percentile(ordered, 75),
        "p90": percentile(ordered, 90),
        "max": ordered[-1],
    }


def percentile(ordered: list[float], pct: float) -> float:
    """Nearest-rank percentile on an already-sorted list."""
    if not ordered:
        raise ValueError("empty sequence")
    rank = max(1, min(len(ordered), int(-(-pct * len(ordered) // 100))))
    return ordered[rank - 1]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def parse_transcript(transcript: str) -> list[tuple[str, str]]:
    """Split a released transcript into (role, text) pairs.

    Released transcripts are newline-separated lines prefixed 'I: ' for the
    interrogator and 'W: ' for the witness. A line without a prefix is a
    continuation of the previous speaker's message, not a new turn.
    """
    turns: list[tuple[str, str]] = []
    for line in transcript.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("I:") or stripped.startswith("W:"):
            role = stripped[0]
            turns.append((role, stripped[2:].strip()))
        elif turns:
            role, text = turns[-1]
            turns[-1] = (role, (text + "\n" + stripped).strip())
    return turns


def scan_pii(text: str) -> list[str]:
    """Return the names of PII screens that fire on this text."""
    return [name for name, pattern in PII_PATTERNS.items() if pattern.search(text)]


def strip_role_prefixes(transcript: str) -> str:
    """The transcript text with the leading 'I: ' / 'W: ' markers removed."""
    return " ".join(text for _role, text in parse_transcript(transcript))


def candidate_length_units(transcripts: list[dict], threshold: int = 50) -> dict:
    """Count games passing a >= threshold filter under each candidate unit.

    Inverse Turing Bench says its 557 pairs are "dialogues of length 50 or
    more". A pair is a game's two transcripts, so the filter must be evaluated
    per game with both sides passing. Duplicate (game, label) rows are collapsed
    first so a re-released duplicate cannot inflate a count.
    """
    by_game: dict[str, dict[str, str]] = defaultdict(dict)
    for row in transcripts:
        by_game[row["game_id"]].setdefault(
            row["conversation_label"], row["transcript"]
        )
    complete = [sides for sides in by_game.values() if len(sides) == 2]

    units = {
        "whitespace_tokens_incl_role_prefixes": lambda t: word_count(t),
        "whitespace_tokens_excl_role_prefixes": lambda t: word_count(
            strip_role_prefixes(t)
        ),
        "turns": lambda t: len(parse_transcript(t)),
        "characters": len,
    }
    out = {
        "games_with_two_transcripts": len(complete),
        "threshold": threshold,
    }
    for name, measure in units.items():
        out[name] = {
            "both_sides_pass": sum(
                1 for sides in complete
                if all(measure(t) >= threshold for t in sides.values())
            ),
            "either_side_passes": sum(
                1 for sides in complete
                if any(measure(t) >= threshold for t in sides.values())
            ),
            "sum_of_sides_passes": sum(
                1 for sides in complete
                if sum(measure(t) for t in sides.values()) >= threshold
            ),
        }
    return out


def counter_table(counter: Counter) -> dict:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def norm_id(value: str | None) -> str:
    """Normalise a foreign key written inconsistently across tables.

    The 15-minute release writes ids through pandas floats, so tt_witness has
    '109819.0' where tt_profile has '109819'. A naive string join silently
    drops every human-witness link, which would understate participant counts
    to zero. Integral floats are collapsed to their integer form.
    """
    text = (value or "").strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def is_blank(value: str | None) -> bool:
    """R's write.csv emits NA for missing; treat it as absent, not as text."""
    return (value or "").strip() in ("", "NA", "NaN", "nan", "NULL", "None")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def file_inventory(subdir: Path) -> list[dict]:
    inventory = []
    for path in sorted(subdir.rglob("*")):
        if path.is_dir() or path.name == "MANIFEST.json":
            continue
        record = {
            "path": str(path.relative_to(SOURCE_DIR)),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".csv":
            fields, rows = read_csv(path)
            record["rows"] = len(rows)
            record["columns"] = fields
            record["n_columns"] = len(fields)
        inventory.append(record)
    return inventory


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def inspect(subdir_name: str) -> dict:
    base = SOURCE_DIR / subdir_name
    if not base.is_dir():
        raise SystemExit(f"missing source dir: {base} — run the fetch script first")

    report: dict = {"subdir": subdir_name, "source_dir": str(base)}
    report["inventory"] = file_inventory(base)

    _, games = read_csv(base / "tt_game.csv")
    _, witnesses = read_csv(base / "tt_witness.csv")
    _, conversations = read_csv(base / "tt_conversation.csv")
    _, messages = read_csv(base / "tt_message_anonymized.csv")
    _, profiles = read_csv(base / "tt_profile.csv")
    _, aimodels = read_csv(base / "tt_aimodel.csv")
    _, transcripts = read_csv(base / "tt_transcripts.csv")
    interrogators_path = base / "tt_interrogator.csv"
    _, interrogators = read_csv(interrogators_path)
    verdict_path = base / "tt_verdict.csv"
    verdicts = read_csv(verdict_path)[1] if verdict_path.exists() else []

    # ---- games -------------------------------------------------------------
    # How many foreign keys are written in a form that breaks a naive join?
    report["id_format"] = {
        table: sum(
            1 for row in rows
            for key in keys
            if row.get(key) is not None and row[key] != norm_id(row[key])
        )
        for table, rows, keys in (
            ("tt_witness", witnesses, ("id", "user_id", "ai_model_id")),
            ("tt_interrogator", interrogators, ("id", "user_id")),
            ("tt_game", games, ("id", "interrogator_id", "human_witness_id",
                                "ai_witness_id")),
            ("tt_conversation", conversations, ("id", "game_id", "witness_id")),
            ("tt_message_anonymized", messages, ("conversation_id",)),
            ("tt_profile", profiles, ("user_id",)),
            ("tt_aimodel", aimodels, ("id",)),
        )
    }

    report["games"] = {
        "rows_in_tt_game": len(games),
        "status": counter_table(Counter(g["status"] for g in games)),
        "is_test": counter_table(Counter(g["is_test"] for g in games)),
        "rows_in_tt_transcripts": len(transcripts),
        "distinct_game_ids_in_transcripts": len({t["game_id"] for t in transcripts}),
        "created_at_min": min((g["created_at"] for g in games), default=None),
        "created_at_max": max((g["created_at"] for g in games), default=None),
    }

    # ---- participants ------------------------------------------------------
    # Ids are normalised because the 15-minute release stores some of them as
    # floats; see norm_id.
    interrogator_user = {norm_id(i["id"]): norm_id(i["user_id"]) for i in interrogators}
    witness_user = {norm_id(w["id"]): norm_id(w["user_id"]) for w in witnesses}
    witness_type = {norm_id(w["id"]): w["witness_type"] for w in witnesses}
    witness_model = {norm_id(w["id"]): norm_id(w["ai_model_id"]) for w in witnesses}
    for game in games:
        for key in ("interrogator_id", "human_witness_id", "ai_witness_id", "id"):
            game[key] = norm_id(game[key])

    users_as_interrogator = {
        interrogator_user.get(g["interrogator_id"])
        for g in games
        if not is_blank(interrogator_user.get(g["interrogator_id"]))
    }
    users_as_human_witness = {
        witness_user.get(g["human_witness_id"])
        for g in games
        if not is_blank(witness_user.get(g["human_witness_id"]))
    }
    profile_users = {norm_id(p["user_id"]) for p in profiles if not is_blank(p["user_id"])}

    report["participants"] = {
        "profile_rows": len(profiles),
        "distinct_user_ids_in_profiles": len(profile_users),
        "distinct_users_as_interrogator": len(users_as_interrogator),
        "distinct_users_as_human_witness": len(users_as_human_witness),
        "distinct_users_either_role": len(users_as_interrogator | users_as_human_witness),
        "users_in_both_roles": len(users_as_interrogator & users_as_human_witness),
        "interrogator_rows": len(interrogators),
        "witness_rows": len(witnesses),
        "witness_rows_by_type": counter_table(Counter(w["witness_type"] for w in witnesses)),
        "stable_identifier": (
            "user_id links tt_profile, tt_interrogator and human tt_witness rows; "
            "tt_interrogator.id and tt_witness.id are per-game seat ids, not people"
        ),
        "profile_source": counter_table(Counter(p.get("source", "") for p in profiles)),
        "profile_study": counter_table(Counter(p.get("study", "") for p in profiles)),
    }

    games_per_interrogator_user = Counter(
        interrogator_user.get(g["interrogator_id"]) for g in games
    )
    games_per_interrogator_user.pop(None, None)
    games_per_witness_user = Counter(
        witness_user.get(g["human_witness_id"]) for g in games
    )
    games_per_witness_user.pop(None, None)
    per_user_total = Counter()
    for user, count in games_per_interrogator_user.items():
        per_user_total[user] += count
    for user, count in games_per_witness_user.items():
        per_user_total[user] += count

    report["games_per_participant"] = {
        "as_interrogator": summarize(list(games_per_interrogator_user.values())),
        "as_human_witness": summarize(list(games_per_witness_user.values())),
        "either_role": summarize(list(per_user_total.values())),
        "as_interrogator_histogram": counter_table(
            Counter(games_per_interrogator_user.values())
        ),
    }

    # ---- witness systems ---------------------------------------------------
    for model in aimodels:
        model["id"] = norm_id(model["id"])
    games_per_model = Counter()
    for game in games:
        model_id = witness_model.get(game["ai_witness_id"])
        games_per_model[model_id] += 1
    report["witness_systems"] = {
        "tt_aimodel_rows": len(aimodels),
        "models": [
            {
                "id": m["id"],
                "name": m["name"],
                "family": m["family"],
                "model_name": m["model_name"],
                "response_method": m["response_method"],
                "prompt_id": m["prompt_id"],
                "temperature": m["temperature"],
                "active": m["active"],
                "games_in_tt_game": games_per_model.get(m["id"], 0),
                "wins_reported": m["wins"],
                "losses_reported": m["losses"],
            }
            for m in aimodels
        ],
        "games_by_model_id": counter_table(games_per_model),
        "games_by_transcript_witness_label": counter_table(
            Counter(t["witness"] for t in transcripts)
        ),
    }

    # ---- messages ----------------------------------------------------------
    for conversation in conversations:
        for key in ("id", "game_id", "witness_id", "interrogator_id"):
            conversation[key] = norm_id(conversation[key])
    conv_game = {c["id"]: c["game_id"] for c in conversations}
    conv_witness = {c["id"]: c["witness_id"] for c in conversations}
    msgs_by_conv: dict[str, list[dict]] = defaultdict(list)
    for message in messages:
        msgs_by_conv[norm_id(message["conversation_id"])].append(message)

    msgs_per_game = Counter()
    for conv_id, conv_messages in msgs_by_conv.items():
        game_id = conv_game.get(conv_id)
        if game_id is not None:
            msgs_per_game[game_id] += len(conv_messages)

    role_counter = Counter(m["sender_role"] for m in messages)
    hidden_counter = Counter(m["hidden"] for m in messages)
    changed_counter = Counter(m["is_changed"] for m in messages)

    chars_by_role: dict[str, list[int]] = defaultdict(list)
    words_by_role: dict[str, list[int]] = defaultdict(list)
    for message in messages:
        content = message["content"] or ""
        chars_by_role[message["sender_role"]].append(len(content))
        words_by_role[message["sender_role"]].append(word_count(content))

    msgs_per_conv_by_witness_type = defaultdict(list)
    for conv_id, conv_messages in msgs_by_conv.items():
        wtype = witness_type.get(conv_witness.get(conv_id, ""), "?")
        msgs_per_conv_by_witness_type[wtype].append(len(conv_messages))

    report["messages"] = {
        "rows_in_tt_message_anonymized": len(messages),
        "distinct_conversation_ids_in_messages": len(msgs_by_conv),
        "conversation_rows": len(conversations),
        "by_sender_role": counter_table(role_counter),
        "hidden_flag": counter_table(hidden_counter),
        "is_changed_flag": counter_table(changed_counter),
        "empty_content": sum(1 for m in messages if not (m["content"] or "").strip()),
        "per_game": summarize([float(v) for v in msgs_per_game.values()]),
        "per_conversation": summarize(
            [float(len(v)) for v in msgs_by_conv.values()]
        ),
        "per_conversation_by_witness_type": {
            wtype: summarize([float(v) for v in vals])
            for wtype, vals in sorted(msgs_per_conv_by_witness_type.items())
        },
        "chars_per_message_by_role": {
            role: summarize([float(v) for v in vals])
            for role, vals in sorted(chars_by_role.items())
        },
        "words_per_message_by_role": {
            role: summarize([float(v) for v in vals])
            for role, vals in sorted(words_by_role.items())
        },
        "role_distinguishable": sorted(role_counter),
    }

    # ---- transcripts (the joined convenience table) -------------------------
    transcript_turns = []
    transcript_role_counts = Counter()
    prefix_ok = 0
    for row in transcripts:
        turns = parse_transcript(row["transcript"])
        transcript_turns.append(float(len(turns)))
        transcript_role_counts.update(role for role, _ in turns)
        if turns and turns[0][0] == "I":
            prefix_ok += 1
    declared_counts = [float(int(t["message_count"])) for t in transcripts
                       if t["message_count"]]
    report["transcripts"] = {
        "rows": len(transcripts),
        "message_count_column": summarize(declared_counts),
        "parsed_turns": summarize(transcript_turns),
        "parsed_turns_by_role": counter_table(transcript_role_counts),
        "rows_starting_with_interrogator": prefix_ok,
        "is_human_values": counter_table(Counter(t["is_human"] for t in transcripts)),
        "declared_equals_parsed": sum(
            1 for row, parsed in zip(transcripts, transcript_turns)
            if row["message_count"] and float(int(row["message_count"])) == parsed
        ),
        # Inverse Turing Bench filters dialogues at "length >= 50" and reports
        # 557 pairs. Measure every candidate unit at the PAIR level (a pair is
        # one game's two transcripts) so the unit is identified, not guessed.
        "candidate_length_units_ge_50": candidate_length_units(transcripts),
        "words_per_transcript": summarize(
            [float(word_count(t["transcript"])) for t in transcripts]
        ),
        "chars_per_transcript": summarize(
            [float(len(t["transcript"])) for t in transcripts]
        ),
    }

    # ---- verdicts / confidence / reasons -----------------------------------
    if verdicts:
        verdict_rows, verdict_source = verdicts, "tt_verdict.csv"
    else:
        # No verdict table (15-minute release): the verdict is repeated on both
        # transcript rows of a game, so collapse to one row per game.
        by_game_verdict: dict[str, dict] = {}
        for row in transcripts:
            by_game_verdict.setdefault(row["game_id"], row)
        verdict_rows = list(by_game_verdict.values())
        verdict_source = "tt_transcripts.csv (one row per game)"
    confidences = [float(r["confidence"]) for r in verdict_rows
                   if r.get("confidence") not in (None, "", "NA")]
    reasons = [(r.get("reason") or "").strip() for r in verdict_rows]
    report["verdicts"] = {
        "source_table": verdict_source,
        "rows": len(verdict_rows),
        "verdict_values": counter_table(Counter(r.get("verdict") for r in verdict_rows)),
        "is_correct_values": counter_table(
            Counter(r.get("is_correct") for r in verdict_rows)
        ),
        "confidence": summarize(confidences),
        "confidence_present": len(confidences),
        "reason_present": sum(1 for r in reasons if r),
        "reason_chars": summarize([float(len(r)) for r in reasons if r]),
        "reason_words": summarize([float(word_count(r)) for r in reasons if r]),
    }

    # ---- outcome by witness system (measured, not quoted) ------------------
    correct_by_witness: dict[str, Counter] = defaultdict(Counter)
    seen_games: set[tuple[str, str]] = set()
    for row in transcripts:
        key = (row["game_id"], row["conversation_label"])
        if key in seen_games or row["is_human"].strip().upper() == "TRUE":
            continue
        seen_games.add(key)
        correct_by_witness[row["witness"]][row["is_correct"].strip().upper()] += 1
    report["accuracy_by_witness_system"] = {
        witness: {
            "games": sum(counts.values()),
            "interrogator_correct": counts.get("TRUE", 0),
            "interrogator_correct_rate": round(
                counts.get("TRUE", 0) / max(1, sum(counts.values())), 3
            ),
        }
        for witness, counts in sorted(correct_by_witness.items())
    }

    # ---- who is excluded by the study's own filters -------------------------
    aware_users = {
        norm_id(p["user_id"]) for p in profiles
        if (p.get("expt_aware") or "").strip().upper() in ("TRUE", "1")
    }
    games_touching_aware = sum(
        1 for g in games
        if interrogator_user.get(g["interrogator_id"]) in aware_users
        or witness_user.get(g["human_witness_id"]) in aware_users
    )
    report["study_filters"] = {
        "profiles_expt_aware_true": len(aware_users),
        "games_with_an_expt_aware_participant": games_touching_aware,
        "games_if_those_removed": len(games) - games_touching_aware,
        "note": (
            "The release is the post-filter export described in the codebook; "
            "any further published subset must be reproduced from the R scripts, "
            "not assumed."
        ),
    }

    # ---- free text and PII screens ----------------------------------------
    free_text_fields = {
        "tt_message_anonymized.content": [m["content"] or "" for m in messages],
        "verdict.reason": [r for r in reasons if r],
        "tt_profile.strategy": [p.get("strategy") or "" for p in profiles],
        "tt_profile.strategy_change": [p.get("strategy_change") or "" for p in profiles],
        "tt_profile.other": [p.get("other") or "" for p in profiles],
        "tt_profile.expt_aware_details": [
            p.get("expt_aware_details") or "" for p in profiles
        ],
        "tt_witness.prompt": [w.get("prompt") or "" for w in witnesses],
    }
    pii = {}
    for field, values in free_text_fields.items():
        hits = Counter()
        nonempty = 0
        for value in values:
            if is_blank(value):
                continue
            nonempty += 1
            for name in scan_pii(value):
                hits[name] += 1
        pii[field] = {
            "nonempty_values": nonempty,
            "screen_hits": counter_table(hits),
        }
    report["free_text_and_pii_screens"] = pii

    # Demographic columns are direct identifiers-adjacent; report presence.
    report["demographics_present"] = {
        column: sum(1 for p in profiles if not is_blank(p.get(column)))
        for column in ("gender", "year_of_birth", "education",
                       "chatbot_interaction_frequency", "familiarity_with_GPT",
                       "emotion", "intelligence", "accuracy_estimate")
        if profiles and column in profiles[0]
    }

    # ---- referential integrity and duplicates ------------------------------
    transcript_key_counts = Counter(
        (t["game_id"], t["conversation_label"]) for t in transcripts
    )
    verdict_game_counts = Counter(v["game_id"] for v in verdicts)
    convs_with_messages = set(msgs_by_conv)
    report["integrity"] = {
        "duplicate_transcript_rows": {
            f"{game}/{label}": count
            for (game, label), count in transcript_key_counts.items()
            if count > 1
        },
        "games_with_multiple_verdicts": {
            game: count for game, count in verdict_game_counts.items() if count > 1
        },
        "verdict_game_ids_absent_from_tt_game": len(
            {v["game_id"] for v in verdicts} - {g["id"] for g in games}
        ),
        "conversations_with_zero_messages": len(
            {c["id"] for c in conversations} - convs_with_messages
        ),
        "games_with_zero_messages": len(
            {g["id"] for g in games}
            - {conv_game[c] for c in convs_with_messages if c in conv_game}
        ),
        "message_conversation_ids_absent_from_tt_conversation": len(
            convs_with_messages - {c["id"] for c in conversations}
        ),
        "games_without_two_conversations": sum(
            1 for count in Counter(c["game_id"] for c in conversations).values()
            if count != 2
        ),
    }

    # ---- ledger-relevant absences -----------------------------------------
    report["track_a_field_presence"] = {
        "per_turn_belief_or_confidence": any(
            "confidence" in (m.keys()) for m in messages[:1]
        ),
        "message_timestamps": bool(messages and messages[0].get("timestamp")),
        "message_level_columns": sorted(messages[0].keys()) if messages else [],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdir", default="data",
                        help="'data' (main release) or '15_mins'")
    parser.add_argument("--out", default=None, help="output JSON path")
    args = parser.parse_args(argv)

    report = inspect(args.subdir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        OUT_DIR / f"three_party_inspection_{args.subdir}.json"
    )
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
