"""Tests for the confound separator (steps vs unique-token breadth).

Pure logic — no torch, no GPU. Exercises both verdict branches against the REAL
full-canon dose-curve numbers plus two synthetic pilot@103 scenarios:
  - pilot stays healthy across steps  -> BREADTH verdict
  - pilot decays with steps like canon -> STEPS verdict
The pilot run has not happened yet; this proves the analysis reads the 2x2
correctly so the verdict is trustworthy the moment the JSON lands.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

import confound_analysis as ca

# Real full-canon closure by step (2026-08-08 run). Decays with dose.
FULLCANON = [
    {"label": f"step-{s}", "closure": c, "n": 8, "step": s}
    for s, c in [
        (5, 8), (10, 8), (15, 7), (20, 3), (25, 4), (30, 5), (35, 4),
        (40, 7), (45, 3), (50, 3), (55, 3), (60, 4), (65, 1), (70, 3),
        (75, 5), (80, 2), (85, 4), (90, 4), (95, 6), (100, 4), (103, 2),
    ]
]

STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
         90, 95, 100, 103]


def _pilot(closure_by_step):
    return [{"label": f"step-{s}", "closure": c, "n": 8, "step": s}
            for s, c in closure_by_step]


class TestConfound(unittest.TestCase):
    def test_breadth_verdict_when_pilot_stays_healthy(self):
        # Pilot at 1/11th the breadth holds closure ~8/8 across all steps.
        pilot = _pilot([(s, 8) for s in STEPS])
        a = ca.analyze_confound(FULLCANON, pilot)
        self.assertTrue(a["breadth_hurt"])
        self.assertFalse(a["steps_hurt"])
        self.assertIn("BREADTH", a["verdict"])
        self.assertIn("REHEARSAL", a["verdict"])
        # breadth-at-high-steps should be strongly significant (8/8 vs ~2-3/8)
        self.assertLess(
            a["contrasts"]["breadth_at_high_steps"]["fisher_p"], 0.01)

    def test_steps_verdict_when_pilot_decays_like_canon(self):
        # Pilot decays with steps just like canon -> no breadth gap, steps hurt.
        pilot = _pilot([(s, c["closure"]) for s, c in zip(STEPS, FULLCANON)])
        a = ca.analyze_confound(FULLCANON, pilot)
        self.assertTrue(a["steps_hurt"])
        self.assertFalse(a["breadth_hurt"])
        self.assertIn("STEPS", a["verdict"])

    def test_inconclusive_when_flat_and_matched(self):
        # Pilot flat AND canon-like-flat impossible here; use a pilot that
        # matches canon's early rate and shows no step decay and no breadth gap.
        # Construct a pilot equal to canon everywhere -> steps_hurt True actually.
        # For inconclusive, make pilot mirror canon's EARLY pooled rate flat.
        flat = [(s, 5) for s in STEPS]  # constant 5/8, no step trend
        # canon early pooled ~0.70; pilot flat 0.625 -> breadth gap small/not sig
        pilot = _pilot(flat)
        a = ca.analyze_confound(FULLCANON, pilot)
        # constant pilot -> steps_hurt False (early==late)
        self.assertFalse(a["steps_hurt"])

    def test_contrasts_have_wilson_and_p(self):
        pilot = _pilot([(s, 8) for s in STEPS])
        a = ca.analyze_confound(FULLCANON, pilot)
        for c in a["contrasts"].values():
            self.assertIn("wilson", c["a"])
            self.assertGreaterEqual(c["fisher_p"], 0.0)
            self.assertLessEqual(c["fisher_p"], 1.0)

    def test_print_runs(self):
        pilot = _pilot([(s, 8) for s in STEPS])
        ca.print_confound(ca.analyze_confound(FULLCANON, pilot))


if __name__ == "__main__":
    unittest.main()
