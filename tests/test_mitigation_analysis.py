"""Tests for the low-rank mitigation analysis (closure x effect overlay).

Pure logic — no torch, no GPU. This is a checkpoint screen, not evidence that a
reasoning intervention succeeded.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

import mitigation_analysis as ma


def _closure(rows):  # rows: (step, closure_out_of_8)
    return [{"label": f"step-{s}", "closure": c, "n": 8, "step": s}
            for s, c in rows]


def _effect(rows):   # rows: (step, drop_pct)
    return [{"label": f"step-{s}", "heldout_ppl": 100 - d, "heldout_drop_pct": d,
             "step": s} for s, d in rows]


STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


class TestMitigation(unittest.TestCase):
    def test_rescued_when_intact_and_effect_coincide(self):
        # closure stays high; PPL drop grows past 5% while closure still intact.
        closure = _closure([(s, 8 if s <= 30 else 6) for s in STEPS])
        effect = _effect([(s, 0.5 * (s / 5)) for s in STEPS])  # 0.5%..5% by step 50
        # boost an early-ish step above the gate while closure is intact:
        effect = _effect([(5, 1), (10, 2), (15, 4), (20, 6), (25, 7), (30, 8),
                          (35, 9), (40, 10), (45, 11), (50, 12)])
        a = ma.analyze_mitigation(closure, effect)
        self.assertEqual(a["verdict_key"], "CANDIDATE_WINDOW")
        self.assertTrue(a["window"])
        # step-20 is the first with closure>=0.75 (8/8) AND drop>=5 (6%)
        self.assertTrue(any(j["step"] == 20 for j in a["window"]))

    def test_coupled_when_effect_only_after_collapse(self):
        # closure collapses early; the PPL drop only arrives once closure is gone.
        closure = _closure([(s, 8 if s <= 15 else 2) for s in STEPS])
        effect = _effect([(s, 0.0 if s <= 15 else 9.0) for s in STEPS])
        a = ma.analyze_mitigation(closure, effect)
        self.assertEqual(a["verdict_key"], "PROXY_COUPLED")
        self.assertFalse(a["window"])

    def test_too_weak_when_no_effect_anywhere(self):
        # closure fine early, but PPL never drops past the gate -> learned nothing.
        closure = _closure([(s, 8 if s <= 30 else 5) for s in STEPS])
        effect = _effect([(s, 1.0) for s in STEPS])  # never reaches 5%
        a = ma.analyze_mitigation(closure, effect)
        self.assertEqual(a["verdict_key"], "NO_PROXY_WINDOW")
        self.assertFalse(a["window"])

    def test_join_is_on_shared_steps_only(self):
        closure = _closure([(5, 8), (10, 8), (15, 7)])
        effect = _effect([(10, 6), (15, 7), (20, 8)])  # 20 has no closure
        a = ma.analyze_mitigation(closure, effect)
        self.assertEqual([j["step"] for j in a["joined"]], [10, 15])

    def test_print_runs(self):
        closure = _closure([(s, 8) for s in STEPS])
        effect = _effect([(s, 6) for s in STEPS])
        ma.print_mitigation(ma.analyze_mitigation(closure, effect))


if __name__ == "__main__":
    unittest.main()
