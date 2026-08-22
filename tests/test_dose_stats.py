"""Tests for corrected dose-curve summaries and future prompt-level logging.

Pure logic — no torch, no GPU, no network. dose_curve.py imports torch only
inside main(), so the module imports cleanly here and the analysis functions are
directly testable. This is the local dry-parse the owner asked for: the actual
5.5 h re-run happens later on Kaggle.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

import dose_curve


class TestWilson(unittest.TestCase):
    def test_full_closure_is_not_degenerate(self):
        # The whole reason for Wilson over the normal approx: 8/8 must NOT be
        # [1.0, 1.0]. The lower bound should sit well below 1.
        lo, hi = dose_curve.wilson_interval(8, 8)
        self.assertLess(lo, 0.75)
        self.assertLessEqual(hi, 1.0)
        self.assertEqual(hi, 1.0)

    def test_bounds_stay_in_unit_interval(self):
        for k in range(0, 9):
            lo, hi = dose_curve.wilson_interval(k, 8)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)
            self.assertLessEqual(lo, hi)

    def test_zero_n(self):
        self.assertEqual(dose_curve.wilson_interval(0, 0), (0.0, 1.0))

    def test_half_is_symmetric_about_half(self):
        lo, hi = dose_curve.wilson_interval(4, 8)
        self.assertAlmostEqual((lo + hi) / 2, 0.5, places=6)


class TestFisher(unittest.TestCase):
    def test_lady_tasting_tea(self):
        # The textbook value; if this drifts the implementation is wrong.
        p = dose_curve.fisher_exact_two_sided(3, 1, 1, 3)
        self.assertAlmostEqual(p, 0.4857, places=3)

    def test_no_difference_is_p_one(self):
        self.assertAlmostEqual(
            dose_curve.fisher_exact_two_sided(5, 5, 5, 5), 1.0, places=6)

    def test_perfect_separation_is_small(self):
        p = dose_curve.fisher_exact_two_sided(8, 0, 0, 8)
        self.assertLess(p, 0.01)

    def test_p_in_unit_interval(self):
        for tbl in [(8, 0, 3, 5), (27, 5, 28, 28), (2, 6, 5, 3)]:
            p = dose_curve.fisher_exact_two_sided(*tbl)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


# The logged 2026-08-07 dose-curve numbers, preserved from the session log.
# base 8/8; 5->8/8, 15->7/8, 25->6/8, 35->6/8, 45->5/8, 55->3/8, 65->3/8,
# 75->5/8 (1 trunc), 85->5/8, 95->5/8, 103->2/8; final 3/8.
LOGGED_20260807 = [
    {"label": "base", "closure": 8, "n": 8},
    {"label": "step-5", "closure": 8, "n": 8, "step": 5},
    {"label": "step-15", "closure": 7, "n": 8, "step": 15},
    {"label": "step-25", "closure": 6, "n": 8, "step": 25},
    {"label": "step-35", "closure": 6, "n": 8, "step": 35},
    {"label": "step-45", "closure": 5, "n": 8, "step": 45},
    {"label": "step-55", "closure": 3, "n": 8, "step": 55},
    {"label": "step-65", "closure": 3, "n": 8, "step": 65},
    {"label": "step-75", "closure": 5, "n": 8, "step": 75, "truncated": 1},
    {"label": "step-85", "closure": 5, "n": 8, "step": 85},
    {"label": "step-95", "closure": 5, "n": 8, "step": 95},
    {"label": "step-103", "closure": 2, "n": 8, "step": 103},
    {"label": "final", "closure": 3, "n": 8, "step": None},
]


class TestAnalysisOn20260807(unittest.TestCase):
    def setUp(self):
        self.analysis = dose_curve.analyze_rows(LOGGED_20260807)

    def test_base_excluded_final_excluded_from_graded(self):
        # per_checkpoint covers only rows with an integer step: 11 of them.
        steps = [r["step"] for r in self.analysis["per_checkpoint"]]
        self.assertEqual(steps, [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 103])

    def test_historical_checkpoint_rows_are_descriptive(self):
        for r in self.analysis["per_checkpoint"]:
            self.assertIsNone(r["interval"])
            self.assertIn("descriptive", r["warning"])

    def test_early_vs_late_is_descriptive_without_inference(self):
        pooled = self.analysis["descriptive_early_late"]
        # early = steps 5,15,25,35 -> 8+7+6+6 = 27 of 32
        self.assertEqual((pooled["early_ok"], pooled["early_n"]), (27, 32))
        # late = steps 45..103 -> 5+3+3+5+5+5+2 = 28 of 56
        self.assertEqual((pooled["late_ok"], pooled["late_n"]), (28, 56))
        self.assertGreater(pooled["early_rate"], pooled["late_rate"])
        self.assertNotIn("fisher_p_two_sided", pooled)
        self.assertNotIn("early_wilson", pooled)
        self.assertIn("descriptive", pooled["warning"])
        self.assertIsNone(self.analysis["prompt_blocked"])
        self.assertIn("withdrawn", self.analysis["inference_status"])

    def test_prompt_blocked_summary_when_outcomes_exist(self):
        rows = []
        for step, outcomes in ((5, [1, 1, 0, 0]), (10, [1, 1, 1, 0]),
                               (45, [0, 1, 0, 0]), (50, [0, 0, 0, 0])):
            rows.append({
                "label": f"step-{step}", "step": step,
                "closure": sum(outcomes), "n": len(outcomes),
                "prompt_outcomes": [
                    {"prompt_id": f"p{i}", "closed": bool(v)}
                    for i, v in enumerate(outcomes)
                ],
            })
        got = dose_curve.analyze_rows(rows)["prompt_blocked"]
        self.assertEqual(got["n_prompts"], 4)
        self.assertLess(got["late_minus_early_rate"], 0.0)
        self.assertEqual(len(got["prompt_bootstrap_95"]), 2)
        self.assertIn("not a training-seed replication", got["interpretation"])

    def test_print_analysis_runs(self):
        # Smoke: the printer must not raise on real data.
        dose_curve.print_analysis(self.analysis)


if __name__ == "__main__":
    unittest.main()
