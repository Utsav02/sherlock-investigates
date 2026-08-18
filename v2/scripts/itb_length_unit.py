#!/usr/bin/env python3
"""
Determine what Inverse Turing Bench's "dialogues of length 50 or more" measures.

Stage B step 2 of `v2/experiment_design.md`, pulled forward into Stage A because
the answer decides how much of the three-party corpus a reproduction can use.
The design doc's instruction is explicit: inspect the released representation,
do not guess the unit.

Method — no guessing anywhere in it:
  1. Read the benchmark's released dialogue file (id, dialogueA, dialogueB).
  2. Read the three-party release's tt_transcripts.csv.
  3. Match each released dialogue back to its source transcript by whitespace-
     normalized string equality, recovering which games the benchmark kept.
  4. Count, for every candidate unit (whitespace tokens with and without the
     'I:'/'W:' role prefixes, turns, characters), how many games have BOTH
     transcripts at or above 50 — and compare each count with the benchmark's
     own row count and with the recovered game set.

A unit is identified only if its count equals the released row count AND the
set of games it selects equals the set actually present in the release.

The benchmark file is NOT committed to this repo: it is a separate source and
needs its own §8.2 registry record before it enters v2/data/sources/. Pass its
path explicitly.

Usage:
    venv/bin/python v2/scripts/itb_length_unit.py --itb-csv /path/to/\
InverseTuringBench_o50_conversations_shuffled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_three_party import (  # noqa: E402
    SOURCE_DIR,
    OUT_DIR,
    parse_transcript,
    strip_role_prefixes,
    summarize,
    word_count,
)

csv.field_size_limit(10_000_000)

UNITS = {
    "whitespace_tokens_incl_role_prefixes": word_count,
    "whitespace_tokens_excl_role_prefixes": lambda t: word_count(strip_role_prefixes(t)),
    "turns": lambda t: len(parse_transcript(t)),
    "characters": len,
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_transcripts(subdir: str) -> dict[str, dict[str, str]]:
    path = SOURCE_DIR / subdir / "tt_transcripts.csv"
    by_game: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_game[row["game_id"]].setdefault(
                row["conversation_label"], row["transcript"]
            )
    return by_game


def analyze(itb_csv: Path, subdir: str, threshold: int) -> dict:
    with itb_csv.open(newline="", encoding="utf-8") as handle:
        itb_rows = list(csv.DictReader(handle))
    by_game = load_transcripts(subdir)
    complete = {g: sides for g, sides in by_game.items() if len(sides) == 2}

    index: dict[str, list[str]] = defaultdict(list)
    for game, sides in complete.items():
        for text in sides.values():
            index[normalize(text)].append(game)

    # One released dialogue differs from its source only by the casing of a
    # pandas-emitted missing value ('nan' vs 'NaN'). A case-folded second pass
    # recovers it; the count of such recoveries is reported, never hidden.
    fold_index: dict[str, list[str]] = defaultdict(list)
    for key, games in index.items():
        fold_index[key.casefold()].extend(games)

    matched_games: set[str] = set()
    unmatched_sides = 0
    case_fallback_sides = 0
    cross_game_pairs = 0
    for row in itb_rows:
        sides = []
        for column in ("dialogueA", "dialogueB"):
            key = normalize(row[column])
            games = set(index.get(key, []))
            if not games:
                games = set(fold_index.get(key.casefold(), []))
                if games:
                    case_fallback_sides += 1
            sides.append(games)
        games_a, games_b = sides
        unmatched_sides += (not games_a) + (not games_b)
        shared = games_a & games_b
        if shared:
            matched_games |= shared
        elif games_a and games_b:
            cross_game_pairs += 1

    units = {}
    for name, measure in UNITS.items():
        selected = {
            game for game, sides in complete.items()
            if all(measure(text) >= threshold for text in sides.values())
        }
        values = [float(measure(t)) for sides in complete.values()
                  for t in sides.values()]
        released = [
            float(measure(row[col])) for row in itb_rows
            for col in ("dialogueA", "dialogueB")
        ]
        units[name] = {
            "games_selected": len(selected),
            "equals_released_row_count": len(selected) == len(itb_rows),
            "games_selected_not_in_release": len(selected - matched_games),
            "games_in_release_not_selected": len(matched_games - selected),
            "set_matches_release": selected == matched_games,
            "released_dialogue_values": summarize(released),
            "released_below_threshold": sum(1 for v in released if v < threshold),
            "corpus_dialogue_values": summarize(values),
        }

    identified = [
        name for name, result in units.items()
        if result["equals_released_row_count"]
        and result["set_matches_release"]
        and result["released_below_threshold"] == 0
    ]

    return {
        "itb_csv": str(itb_csv),
        "itb_rows": len(itb_rows),
        "itb_columns": list(itb_rows[0].keys()) if itb_rows else [],
        "source_subdir": subdir,
        "threshold": threshold,
        "games_with_two_transcripts": len(complete),
        "released_dialogues_matched_to_source_games": len(matched_games),
        "released_dialogue_sides_unmatched": unmatched_sides,
        "released_dialogue_sides_matched_via_case_fallback": case_fallback_sides,
        "released_pairs_spanning_two_games": cross_game_pairs,
        "units": units,
        "identified_unit": identified,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--itb-csv", required=True, type=Path)
    parser.add_argument("--subdir", default="data")
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--out", default=str(OUT_DIR / "itb_length_unit.json"))
    args = parser.parse_args(argv)

    result = analyze(args.itb_csv, args.subdir, args.threshold)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
