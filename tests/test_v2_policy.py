"""
Tests that the canonical-layer data-handling policy is enforced, not just written down.

The exclusion of `tt_profile.other` is a PII decision recorded in
`v2/data/sources/registry/jones_bergen_2025.md` §8.1. These tests are what make
that decision binding on future code: they fail if the column is dropped from the
policy, and they fail if any committed v2 artefact carries its values.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2" / "scripts"))

import canonical_policy as policy  # noqa: E402

SOURCE_DIR = REPO_ROOT / "v2" / "data" / "sources" / "jones_bergen_2025"
SPLITS_DIR = REPO_ROOT / "v2" / "data" / "canonical" / "splits"


class TestExclusionPolicy(unittest.TestCase):
    def test_profile_other_is_excluded(self):
        self.assertTrue(policy.is_excluded("tt_profile", "other"))

    def test_exclusion_carries_a_justification(self):
        reason = policy.EXCLUDED_COLUMNS[("tt_profile", "other")]
        self.assertIn("§8.1", reason)
        self.assertTrue(len(reason) > 60, "justification must be specific")

    def test_ordinary_columns_are_not_excluded(self):
        for column in ("user_id", "strategy", "expt_aware", "accuracy_estimate"):
            self.assertFalse(policy.is_excluded("tt_profile", column))

    def test_exclusion_is_table_scoped(self):
        """A column named 'other' in a different table is a different column."""
        self.assertFalse(policy.is_excluded("tt_game", "other"))

    def test_check_columns_fails_closed(self):
        with self.assertRaises(policy.ExcludedColumnError):
            policy.check_columns("tt_profile", ["user_id", "other"])

    def test_check_columns_passes_clean_set(self):
        policy.check_columns("tt_profile", ["user_id", "strategy"])   # must not raise

    def test_check_columns_error_names_the_column(self):
        with self.assertRaises(policy.ExcludedColumnError) as ctx:
            policy.check_columns("tt_profile", ["other"])
        self.assertIn("other", str(ctx.exception))

    def test_allowed_columns_filters(self):
        self.assertEqual(
            policy.allowed_columns("tt_profile", ["user_id", "other", "gender"]),
            ["user_id", "gender"],
        )

    def test_restricted_is_distinct_from_excluded(self):
        """Demographics stay usable locally; only §8.1 columns are barred."""
        self.assertTrue(policy.is_restricted("tt_profile", "gender"))
        self.assertFalse(policy.is_excluded("tt_profile", "gender"))


@unittest.skipUnless(SPLITS_DIR.is_dir(), "no split artefacts written yet")
class TestArtefactsCarryNoExcludedText(unittest.TestCase):
    """Derived artefacts must contain identifiers, never free-text profile values."""

    def test_split_files_hold_no_free_text(self):
        for path in SPLITS_DIR.glob("*.json"):
            payload = json.loads(path.read_text())
            keys = set()

            def walk(node):
                if isinstance(node, dict):
                    keys.update(node.keys())
                    for value in node.values():
                        walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)

            walk(payload)
            self.assertNotIn("other", keys, f"{path.name} carries a column named 'other'")

    def test_split_values_are_identifiers_not_prose(self):
        """A cheap shape check: no value in the split should be a long free-text
        string, which is what a leaked profile response would look like."""
        for path in SPLITS_DIR.glob("*.json"):
            payload = json.loads(path.read_text())
            long_strings = []

            def walk(node):
                if isinstance(node, dict):
                    for value in node.values():
                        walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)
                elif isinstance(node, str) and len(node) > 200:
                    long_strings.append(node[:40])

            walk(payload)
            self.assertEqual(long_strings, [], f"{path.name} holds long free text")


if __name__ == "__main__":
    unittest.main()
