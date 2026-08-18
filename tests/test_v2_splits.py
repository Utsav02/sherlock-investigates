"""
Tests for the frozen Track A participant split.

The split assignment lives under `v2/data/canonical/`, which is gitignored
because the source's redistribution terms are unclear (registry §5). The freeze
therefore lives HERE, as `FROZEN_SHA256`: the builder is deterministic, so a
committed hash pins the assignment without publishing participant identifiers.
If the split ever needs to change, bump `SPLIT_VERSION` in the builder and add a
new constant — do not edit this one in place.

The headline assertion, per design §14 ("one human participant belongs to one
split"), is `test_no_user_in_two_splits_in_either_role`. It is the check that
would have caught splitting on `tt_game.interrogator_id`, which is a per-game
seat id rather than a person.

Data-dependent tests skip themselves when the corpus is absent, so `make test`
stays green and network-free on a fresh clone.
"""

import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2" / "scripts"))

import build_splits  # noqa: E402

SOURCE_DIR = REPO_ROOT / "v2" / "data" / "sources" / "jones_bergen_2025"
SPLIT_PATH = (
    REPO_ROOT / "v2" / "data" / "canonical" / "splits"
    / f"{build_splits.SPLIT_VERSION}.json"
)

# Frozen 2026-08-17 from `venv/bin/python v2/scripts/build_splits.py`.
FROZEN_SHA256 = "256543688a5da35879e157771cf1b527b82014923a1134ff06f10e8b24b8a8c7"

# Measured structural facts the split depends on; if any changes, the split is
# invalid and must be rebuilt rather than patched.
EXPECTED_COMPONENT_WEIGHTS = [360, 158, 157, 120, 101, 61, 54, 44, 30, 25, 25, 2, 1, 1, 1]
EXPECTED_ITB_TOTAL = 557


class TestComponentLogic(unittest.TestCase):
    """Pure logic — runs without the corpus."""

    def test_two_users_sharing_a_game_land_in_one_component(self):
        comps = build_splits.connected_components({"g1": {"a", "b"}, "g2": {"c"}})
        self.assertEqual(comps["a"], comps["b"])
        self.assertNotEqual(comps["a"], comps["c"])

    def test_components_chain_transitively(self):
        # a-b, b-c => a, b, c are one atom even though a and c never met.
        comps = build_splits.connected_components({"g1": {"a", "b"}, "g2": {"b", "c"}})
        self.assertEqual(len({comps["a"], comps["b"], comps["c"]}), 1)

    def test_component_labels_are_deterministic_and_size_ordered(self):
        games = {"g1": {"a", "b"}, "g2": {"b", "c"}, "g3": {"z"}}
        first = build_splits.connected_components(games)
        second = build_splits.connected_components(dict(reversed(list(games.items()))))
        self.assertEqual(first, second)
        self.assertEqual(first["a"], 0)   # larger component gets index 0
        self.assertEqual(first["z"], 1)

    def test_partition_hits_targets_when_divisible(self):
        assignment = build_splits.partition_components(
            [10] * 10, {"train": 0.6, "dev": 0.2, "test": 0.2}
        )
        counts = {n: assignment.count(n) for n in ("train", "dev", "test")}
        self.assertEqual(counts, {"train": 6, "dev": 2, "test": 2})

    def test_partition_is_deterministic(self):
        weights = [7, 5, 3, 11, 2, 9]
        targets = {"train": 0.6, "dev": 0.2, "test": 0.2}
        self.assertEqual(
            build_splits.partition_components(weights, targets),
            build_splits.partition_components(weights, targets),
        )

    def test_partition_assigns_every_component_exactly_once(self):
        weights = [360, 158, 157, 120, 101, 61, 54, 44, 30, 25, 25, 2, 1, 1, 1]
        assignment = build_splits.partition_components(
            weights, {"train": 0.6, "dev": 0.2, "test": 0.2}
        )
        self.assertEqual(len(assignment), len(weights))
        self.assertEqual(set(assignment), {"train", "dev", "test"})

    def test_digest_is_order_independent(self):
        self.assertEqual(
            build_splits.digest({"a": 1, "b": 2}),
            build_splits.digest({"b": 2, "a": 1}),
        )


@unittest.skipUnless(SOURCE_DIR.is_dir(), "source not downloaded (make v2-fetch-3p)")
class TestFrozenSplit(unittest.TestCase):
    """Assertions against the real corpus."""

    @classmethod
    def setUpClass(cls):
        cls.payload = build_splits.build()

    # ---- the headline invariant -------------------------------------------

    def test_no_user_in_two_splits_in_either_role(self):
        """Design §14: one participant belongs to one split.

        Derived from the GAMES, per role, not read back out of `user_split` —
        reading that dict back would be vacuous, since a dict cannot map one key
        to two values. Here every game contributes its split to both the person
        who interrogated and the person who was the human witness, so a user who
        interrogated in a train game and sat as witness in a test game shows up
        with two splits and fails.

        This is the check that catches splitting on `tt_game.interrogator_id`:
        297 of 323 users occupy both roles, so a seat-id split scatters the same
        person across partitions while looking participant-level.
        """
        game_split = self.payload["game_split"]
        base = build_splits.SOURCE_DIR / "data"
        _, games = build_splits.read_csv(base / "tt_game.csv")
        _, interrogators = build_splits.read_csv(base / "tt_interrogator.csv")
        _, witnesses = build_splits.read_csv(base / "tt_witness.csv")
        seat_i = {build_splits.norm_id(r["id"]): build_splits.norm_id(r["user_id"])
                  for r in interrogators}
        seat_w = {build_splits.norm_id(r["id"]): build_splits.norm_id(r["user_id"])
                  for r in witnesses}

        seen: dict[str, set[str]] = defaultdict(set)
        roles_seen: dict[str, set[str]] = defaultdict(set)
        for game in games:
            split = game_split[build_splits.norm_id(game["id"])]
            for seat_key, table, role in (
                ("interrogator_id", seat_i, "interrogator"),
                ("human_witness_id", seat_w, "human_witness"),
            ):
                user = table.get(build_splits.norm_id(game[seat_key]))
                if not build_splits.is_blank(user):
                    seen[user].add(split)
                    roles_seen[user].add(role)

        offenders = {u: sorted(s) for u, s in seen.items() if len(s) > 1}
        self.assertEqual(offenders, {}, f"user_id(s) in multiple splits: {offenders}")
        # The check is only meaningful if users really do span both roles.
        self.assertEqual(sum(1 for r in roles_seen.values() if len(r) == 2), 297)
        self.assertEqual(len(seen), 323)

    def test_no_game_straddles_two_splits(self):
        """Both participants of a game must share its split, or the game leaks."""
        user_split = self.payload["user_split"]
        main = build_splits.load_study("data")
        for gid, users in main["game_users"].items():
            splits = {user_split[u] for u in users}
            self.assertEqual(
                len(splits), 1, f"game {gid} straddles splits {sorted(splits)}"
            )
            self.assertEqual(splits.pop(), self.payload["game_split"][gid])

    def test_roles_are_genuinely_mixed(self):
        """Guards the premise: if this ever fails, the union key was unnecessary."""
        both = sum(1 for r in self.payload["roles_per_user"].values() if len(r) == 2)
        self.assertEqual(both, 297)

    def test_seat_ids_are_not_person_ids(self):
        """tt_game.interrogator_id is per-game; splitting on it would be wrong."""
        _, games = build_splits.read_csv(SOURCE_DIR / "data" / "tt_game.csv")
        seats = [build_splits.norm_id(g["interrogator_id"]) for g in games]
        self.assertEqual(len(seats), 1140)
        self.assertEqual(len(set(seats)), 1140)          # 1 game per seat id
        self.assertEqual(len(set(self.payload["user_split"])), 323)   # 323 people

    # ---- the freeze --------------------------------------------------------

    def test_rebuild_matches_frozen_hash(self):
        self.assertEqual(self.payload["sha256"], FROZEN_SHA256)

    def test_written_file_matches_frozen_hash(self):
        if not SPLIT_PATH.exists():
            self.skipTest("split not written (venv/bin/python v2/scripts/build_splits.py)")
        self.assertEqual(json.loads(SPLIT_PATH.read_text())["sha256"], FROZEN_SHA256)

    # ---- structural preconditions -----------------------------------------

    def test_component_weights_unchanged(self):
        self.assertEqual(
            self.payload["summary"]["component_weights"], EXPECTED_COMPONENT_WEIGHTS
        )

    def test_every_game_covered_exactly_once(self):
        self.assertEqual(len(self.payload["game_split"]), 1140)

    def test_split_proportions_within_tolerance(self):
        for name, target in build_splits.TARGETS.items():
            pct = self.payload["summary"]["by_split"][name]["games_pct"] / 100
            self.assertLess(
                abs(pct - target), 0.02,
                f"{name} at {pct:.3f} vs target {target}"
            )

    def test_itb_557_partitions_across_splits(self):
        counts = [
            self.payload["summary"]["by_split"][n]["itb_557_games"]
            for n in ("train", "dev", "test")
        ]
        self.assertEqual(sum(counts), EXPECTED_ITB_TOTAL)

    def test_two_studies_share_no_users(self):
        self.assertEqual(self.payload["summary"]["cross_study_user_overlap"], 0)

    def test_15min_study_is_not_partitioned(self):
        held = self.payload["heldout_15min"]
        self.assertEqual(len(held["games"]), 392)
        self.assertEqual(len(held["users"]), 227)
        self.assertNotIn("user_split", held)


if __name__ == "__main__":
    unittest.main()
