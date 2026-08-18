#!/usr/bin/env python3
"""
Freeze the participant-level Track A split for the Jones & Bergen three-party corpus.

Design §14 requires that "one human participant belongs to one split" and that
"conversations from the same participant stay together". Two measured facts
decide how that rule must be implemented:

1. `tt_game.interrogator_id` is a **per-game seat id**, not a person: 1,140
   distinct values for 1,140 games (max 1 game per value). The same is true of
   `tt_witness.id` (2,280 for 2,280 rows). The person key is `user_id`, reached
   via `tt_interrogator.user_id` and `tt_witness.user_id`. Splitting on the seat
   id would produce a split that looks participant-level and is not.

2. 297 of the 323 main-study users appear as BOTH interrogator and human
   witness, so the split key must be `user_id` unioned over both roles. And
   because every one of the 1,140 games joins exactly two such users, the two
   people in a game must land in the same split or that game straddles two
   splits. The atom is therefore the **connected component** of the participant
   co-occurrence graph (users linked when they shared a game), not the user.

The main study has only 15 such components, the largest holding 360 games. That
is the real granularity available, and it is what makes an exact 60/20/20 by
game count impossible; the assignment below is the closest reachable partition,
chosen by exhaustive dynamic programming so it is deterministic rather than
sampled.

The 15-minute study shares ZERO user_ids with the main study (measured: the id
spaces are disjoint, 2448-3866 vs 109818-110649), so it is kept whole as a
held-out source rather than partitioned. See the caveat in the output: disjoint
id ranges prove the releases were numbered separately, not that no human being
took part in both.

Writes the assignment to `v2/data/canonical/splits/` (gitignored, per the
source's unclear redistribution terms) and prints a sha256 that
`tests/test_v2_splits.py` re-derives, so the freeze is verifiable from the
committed hash without publishing participant identifiers.

Stdlib only.

Usage:
    venv/bin/python v2/scripts/build_splits.py            # write + print hash
    venv/bin/python v2/scripts/build_splits.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_three_party import SOURCE_DIR, is_blank, norm_id, read_csv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "v2" / "data" / "canonical" / "splits"

csv.field_size_limit(10_000_000)

SPLIT_VERSION = "main_study_v1"
# Track A trains only a classifier head (registry §12); dev carries calibration
# and threshold selection, test is untouched until Gate 5.
TARGETS = {"train": 0.60, "dev": 0.20, "test": 0.20}


# ---------------------------------------------------------------------------
# pure helpers (unit-tested in tests/test_v2_splits.py)
# ---------------------------------------------------------------------------

def connected_components(game_users: dict[str, set[str]]) -> dict[str, int]:
    """Map user_id -> component index, linking users who shared a game.

    Deterministic: components are renumbered by (descending user count,
    smallest member) so the labelling does not depend on dict ordering.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for users in game_users.values():
        members = sorted(users)
        for other in members[1:]:
            union(members[0], other)
        for user in members:
            find(user)

    groups: dict[str, set[str]] = defaultdict(set)
    for user in parent:
        groups[find(user)].add(user)
    ordered = sorted(groups.values(), key=lambda s: (-len(s), min(s)))
    return {user: idx for idx, members in enumerate(ordered) for user in members}


def partition_components(weights: list[int], targets: dict[str, float]) -> list[str]:
    """Assign each component to a split, minimising squared game-count error.

    Exhaustive DP over reachable (train_games, dev_games) states, so the result
    is the exact optimum rather than a greedy or randomised approximation. Ties
    break on the lexicographically smallest assignment, making the output a pure
    function of `weights` and `targets`.
    """
    names = sorted(targets)                       # dev, test, train
    total = sum(weights)
    # state -> best assignment tuple, keyed by cumulative games in each of the
    # first len(names)-1 splits (the last is determined by subtraction).
    states: dict[tuple[int, ...], tuple[str, ...]] = {(0,) * (len(names) - 1): ()}
    for weight in weights:
        nxt: dict[tuple[int, ...], tuple[str, ...]] = {}
        for key, assignment in states.items():
            for i, name in enumerate(names):
                new_key = list(key)
                if i < len(names) - 1:
                    new_key[i] += weight
                new_key = tuple(new_key)
                candidate = assignment + (name,)
                if new_key not in nxt or candidate < nxt[new_key]:
                    nxt[new_key] = candidate
        states = nxt

    def cost(key: tuple[int, ...]) -> float:
        counts = dict(zip(names[:-1], key))
        counts[names[-1]] = total - sum(key)
        return sum((counts[n] - targets[n] * total) ** 2 for n in names)

    best_key = min(states, key=lambda k: (cost(k), states[k]))
    return list(states[best_key])


def digest(payload: dict) -> str:
    """sha256 over a canonical serialisation, so the freeze is checkable."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_study(subdir: str) -> dict:
    """Return per-game participant sets plus the role each user played."""
    base = SOURCE_DIR / subdir
    games = read_csv(base / "tt_game.csv")[1]
    interrogators = read_csv(base / "tt_interrogator.csv")[1]
    witnesses = read_csv(base / "tt_witness.csv")[1]
    transcripts = read_csv(base / "tt_transcripts.csv")[1]

    seat_to_user_i = {norm_id(r["id"]): norm_id(r["user_id"]) for r in interrogators}
    seat_to_user_w = {norm_id(r["id"]): norm_id(r["user_id"]) for r in witnesses}

    game_users: dict[str, set[str]] = {}
    roles: dict[str, set[str]] = defaultdict(set)
    for game in games:
        gid = norm_id(game["id"])
        interrogator = seat_to_user_i.get(norm_id(game["interrogator_id"]))
        witness = seat_to_user_w.get(norm_id(game["human_witness_id"]))
        users = set()
        if not is_blank(interrogator):
            users.add(interrogator)
            roles[interrogator].add("interrogator")
        if not is_blank(witness):
            users.add(witness)
            roles[witness].add("human_witness")
        game_users[gid] = users

    witness_label = {}
    for row in transcripts:
        if row["is_human"].strip().upper() != "TRUE":
            witness_label.setdefault(norm_id(row["game_id"]), row["witness"])

    return {
        "game_users": game_users,
        "roles": {u: sorted(r) for u, r in roles.items()},
        "witness_label": witness_label,
    }


def itb_game_ids() -> set[str]:
    """The 557 games in the Inverse Turing Bench release.

    Recomputed from the source rather than read from the benchmark file, which
    is a separate source with its own registry requirement and is not committed.
    `itb_length_unit.json` established the rule exactly — whitespace tokens of
    the released transcript string WITH the `I: `/`W: ` prefixes counted, >= 50
    on BOTH sides of the pair — and that rule reproduced the released set with 0
    games in either direction, so applying it here is a derivation, not a guess.
    """
    _, transcripts = read_csv(SOURCE_DIR / "data" / "tt_transcripts.csv")
    by_game: dict[str, dict[str, str]] = defaultdict(dict)
    for row in transcripts:
        by_game[norm_id(row["game_id"])].setdefault(
            row["conversation_label"], row["transcript"]
        )
    return {
        gid for gid, sides in by_game.items()
        if len(sides) == 2 and all(len(t.split()) >= 50 for t in sides.values())
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build() -> dict:
    main = load_study("data")
    game_users, roles = main["game_users"], main["roles"]

    straddle_risk = sum(1 for users in game_users.values() if len(users) == 2)
    comp_of = connected_components(game_users)
    n_components = max(comp_of.values()) + 1 if comp_of else 0

    comp_games: dict[int, list[str]] = defaultdict(list)
    for gid, users in game_users.items():
        comps = {comp_of[u] for u in users}
        if len(comps) != 1:
            raise AssertionError(f"game {gid} spans components {comps}")
        comp_games[comps.pop()].append(gid)

    weights = [len(comp_games[i]) for i in range(n_components)]
    assignment = partition_components(weights, TARGETS)

    user_split = {u: assignment[c] for u, c in comp_of.items()}
    game_split = {
        gid: assignment[comp_of[next(iter(users))]]
        for gid, users in game_users.items()
    }

    itb = itb_game_ids()
    payload = {
        "split_version": SPLIT_VERSION,
        "source": "jones_bergen_2025/data (main 5-minute study)",
        "split_key": "user_id, unioned over interrogator and human-witness roles",
        "split_unit": "connected component of the participant co-occurrence graph",
        "targets": TARGETS,
        "user_split": dict(sorted(user_split.items(), key=lambda kv: int(kv[0]))),
        "game_split": dict(sorted(game_split.items(), key=lambda kv: int(kv[0]))),
        "component_of_user": dict(sorted(comp_of.items(), key=lambda kv: int(kv[0]))),
    }

    held = load_study("15_mins")
    held_users = sorted(held["roles"], key=int)
    payload["heldout_15min"] = {
        "status": "NOT partitioned - held whole, pending Gate 0 (registry §14 item 3)",
        "users": held_users,
        "games": sorted(held["game_users"], key=int),
    }

    summary = {
        "components": n_components,
        "games_with_two_participants": straddle_risk,
        "component_weights": weights,
        "by_split": {},
        "cross_study_user_overlap": len(set(user_split) & set(held_users)),
    }
    for name in sorted(TARGETS):
        gids = [g for g, s in game_split.items() if s == name]
        summary["by_split"][name] = {
            "games": len(gids),
            "games_pct": round(100 * len(gids) / len(game_split), 1),
            "users": sum(1 for s in user_split.values() if s == name),
            "components": sum(1 for a in assignment if a == name),
            "itb_557_games": len(itb & set(gids)) if itb else None,
            "witness_systems": dict(
                sorted(Counter(main["witness_label"].get(g, "?") for g in gids).items())
            ),
        }
    payload["summary"] = summary
    payload["sha256"] = digest(
        {k: payload[k] for k in ("split_version", "user_split", "game_split")}
    )
    payload["roles_per_user"] = roles
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="rebuild and compare against the frozen file; write nothing")
    args = parser.parse_args(argv)

    payload = build()
    out_path = OUT_DIR / f"{SPLIT_VERSION}.json"

    if args.check:
        if not out_path.exists():
            print(f"FAIL: {out_path} does not exist", file=sys.stderr)
            return 1
        frozen = json.loads(out_path.read_text())
        ok = frozen.get("sha256") == payload["sha256"]
        print(f"frozen  {frozen.get('sha256')}\nrebuilt {payload['sha256']}\n"
              f"{'MATCH' if ok else 'MISMATCH'}")
        return 0 if ok else 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    (OUT_DIR / f"{SPLIT_VERSION}.sha256").write_text(
        f"{payload['sha256']}  {SPLIT_VERSION}.json\n"
    )

    s = payload["summary"]
    print(f"split_version : {SPLIT_VERSION}")
    print(f"sha256        : {payload['sha256']}")
    print(f"components    : {s['components']} (weights {s['component_weights']})")
    print(f"games w/ 2 ppts: {s['games_with_two_participants']} of {len(payload['game_split'])}")
    print(f"cross-study user overlap: {s['cross_study_user_overlap']}")
    for name, row in s["by_split"].items():
        print(f"  {name:5s} games={row['games']:4d} ({row['games_pct']:4.1f}%) "
              f"users={row['users']:3d} comps={row['components']} "
              f"itb557={row['itb_557_games']}")
    print(f"heldout_15min : {len(payload['heldout_15min']['games'])} games, "
          f"{len(payload['heldout_15min']['users'])} users (unpartitioned)")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
