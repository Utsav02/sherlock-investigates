"""
Tests for the Track A precision / MDD helpers.

These are the arithmetic behind a gating decision — which contrasts get frozen
as primary and which are dropped as unresolvable — so they are checked against
closed-form values rather than trusted. Pure logic; no corpus needed.
"""

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2" / "scripts"))

import precision_track_a as P  # noqa: E402


class TestDesignEffect(unittest.TestCase):
    def test_singleton_clusters_have_no_effect(self):
        for icc in (0.0, 0.1, 0.5, 1.0):
            self.assertAlmostEqual(P.design_effect(1.0, icc), 1.0)

    def test_zero_icc_has_no_effect(self):
        for m in (1.0, 4.0, 40.0):
            self.assertAlmostEqual(P.design_effect(m, 0.0), 1.0)

    def test_known_value(self):
        # Kish: 1 + (4-1)*0.1 = 1.3
        self.assertAlmostEqual(P.design_effect(4.0, 0.1), 1.3)

    def test_effective_n_shrinks_with_icc(self):
        self.assertAlmostEqual(P.effective_n(1300, 4.0, 0.1), 1000.0)
        self.assertEqual(P.effective_n(228, 3.62, 0.0), 228)


class TestProportionInterval(unittest.TestCase):
    def test_matches_textbook_wald_halfwidth(self):
        # 1.96 * sqrt(0.25/100) = 0.098
        self.assertAlmostEqual(
            P.ci_halfwidth_proportion(100, 0.5, 1.0, 0.0), 0.098, places=3
        )

    def test_widest_at_p_half(self):
        at_half = P.ci_halfwidth_proportion(200, 0.5, 1.0, 0.0)
        for p in (0.1, 0.3, 0.7, 0.9):
            self.assertLess(P.ci_halfwidth_proportion(200, p, 1.0, 0.0), at_half)

    def test_narrows_as_n_grows(self):
        self.assertAlmostEqual(
            P.ci_halfwidth_proportion(400, 0.5, 1.0, 0.0),
            P.ci_halfwidth_proportion(100, 0.5, 1.0, 0.0) / 2,
            places=6,
        )


class TestPairedBinaryMDD(unittest.TestCase):
    def test_closed_form(self):
        # (1.959964 + 0.8416212) * sqrt(0.25/228)
        expected = (P.Z_ALPHA + P.Z_BETA) * math.sqrt(0.25 / 228)
        self.assertAlmostEqual(P.mdd_paired_binary(228, 0.25, 1.0, 0.0), expected)

    def test_more_discordance_needs_bigger_effect(self):
        low = P.mdd_paired_binary(228, 0.10, 3.6, 0.1)
        high = P.mdd_paired_binary(228, 0.40, 3.6, 0.1)
        self.assertLess(low, high)

    def test_monotone_in_icc(self):
        values = [P.mdd_paired_binary(229, 0.25, 3.52, icc) for icc in P.ICC_GRID]
        self.assertEqual(values, sorted(values))

    def test_perfect_agreement_is_infinitely_precise(self):
        """Zero discordance carries no information about a difference, and the
        formula correctly returns 0 rather than a finite spurious MDD."""
        self.assertEqual(P.mdd_paired_binary(228, 0.0, 3.6, 0.1), 0.0)


class TestUnpairedBinaryMDD(unittest.TestCase):
    def test_symmetric_in_group_order(self):
        self.assertAlmostEqual(
            P.mdd_unpaired_binary(56, 30, 0.5, 3.5, 0.1),
            P.mdd_unpaired_binary(30, 56, 0.5, 3.5, 0.1),
        )

    def test_unbalanced_groups_are_worse_than_balanced(self):
        balanced = P.mdd_unpaired_binary(43, 43, 0.5, 3.5, 0.1)
        skewed = P.mdd_unpaired_binary(76, 10, 0.5, 3.5, 0.1)
        self.assertLess(balanced, skewed)

    def test_closed_form(self):
        expected = (P.Z_ALPHA + P.Z_BETA) * math.sqrt(0.25 / 100 + 0.25 / 100)
        self.assertAlmostEqual(
            P.mdd_unpaired_binary(100, 100, 0.5, 1.0, 0.0), expected
        )


class TestAurocInterval(unittest.TestCase):
    def test_hanley_mcneil_known_case(self):
        """Hanley & McNeil (1982) worked example: AUC 0.893, 51 pos, 58 neg,
        SE ~= 0.031 -> 95% half-width ~= 0.061."""
        half = P.auroc_ci_halfwidth(0.893, 51, 58, 1.0, 0.0)
        self.assertAlmostEqual(half / P.Z_ALPHA, 0.031, places=2)

    def test_narrower_for_higher_auc(self):
        self.assertLess(
            P.auroc_ci_halfwidth(0.95, 228, 228, 1.0, 0.0),
            P.auroc_ci_halfwidth(0.65, 228, 228, 1.0, 0.0),
        )

    def test_never_negative_variance(self):
        for auc in (0.5, 0.6, 0.75, 0.9, 0.99):
            self.assertGreaterEqual(P.auroc_ci_halfwidth(auc, 10, 10, 1.0, 0.0), 0.0)


class TestMeanInterval(unittest.TestCase):
    def test_closed_form(self):
        self.assertAlmostEqual(
            P.ci_halfwidth_mean(100, 0.25, 1.0, 0.0),
            P.Z_ALPHA * 0.25 / 10,
        )


class TestVerdictClassification(unittest.TestCase):
    def _contrast(self, mdd_pp, target):
        return {"mdd_pp": {f"icc_{P.RESOLVABLE_ICC:g}": mdd_pp},
                "target_effect_pp": target}

    def test_comfortably_below_target_is_resolvable(self):
        self.assertEqual(P.classify(self._contrast(4.68, 10.0)), "RESOLVABLE")

    def test_just_below_target_is_marginal(self):
        self.assertEqual(P.classify(self._contrast(8.07, 10.0)), "MARGINAL")

    def test_above_target_is_not_resolvable(self):
        self.assertEqual(P.classify(self._contrast(10.42, 10.0)), "NOT RESOLVABLE")

    def test_missing_target_is_flagged_not_silently_passed(self):
        self.assertEqual(P.classify(self._contrast(22.6, None)),
                         "NO PRE-REGISTERED EFFECT")

    def test_interval_only_contrast(self):
        self.assertEqual(P.classify({"target_effect_pp": None}), "INTERVAL-ONLY")


if __name__ == "__main__":
    unittest.main()
