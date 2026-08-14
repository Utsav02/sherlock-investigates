"""Tests for the thinking-shift register profiler (pure logic, no torch/GPU)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

import thinking_shift as ts


class TestRegisterProfile(unittest.TestCase):
    def test_empty_think(self):
        p = ts.register_profile(None)
        self.assertEqual(p["words"], 0)
        self.assertEqual(p["deduction_per_1k"], 0.0)

    def test_counts_markers(self):
        p = ts.register_profile("Therefore he must be a sailor; clearly the tan proves it.")
        # "therefore", "must be", "clearly" -> 3 deduction hits
        self.assertEqual(p["deduction_hits"], 3)
        self.assertEqual(p["hedging_hits"], 0)

    def test_hedging(self):
        p = ts.register_profile("Maybe he is a sailor, perhaps, I'm not sure.")
        self.assertGreaterEqual(p["hedging_hits"], 3)
        self.assertEqual(p["deduction_hits"], 0)

    def test_rate_is_per_1k_words(self):
        # 2 deduction markers in 4 words -> 500 per 1k
        p = ts.register_profile("Therefore thus indeed so")
        self.assertEqual(p["words"], 4)
        self.assertEqual(p["deduction_hits"], 2)  # therefore, thus
        self.assertEqual(p["deduction_per_1k"], 500.0)


class TestAggregate(unittest.TestCase):
    def _pair(self, cat, b_ded, f_ded):
        return {"category": cat,
                "base_profile": {"deduction_per_1k": b_ded, "hedging_per_1k": 0.0, "words": 100},
                "ft_profile": {"deduction_per_1k": f_ded, "hedging_per_1k": 0.0, "words": 100}}

    def test_delta_by_category(self):
        pairs = [self._pair("DEDUCTION_INVITING", 10, 30),
                 self._pair("DEDUCTION_INVITING", 20, 40),
                 self._pair("NEUTRAL", 5, 6)]
        agg = ts.aggregate_by_category(pairs)
        # deduction category: mean base 15 -> ft 35, delta +20
        self.assertEqual(agg["DEDUCTION_INVITING"]["delta_deduction_per_1k"], 20.0)
        # neutral (control) barely moves
        self.assertEqual(agg["NEUTRAL"]["delta_deduction_per_1k"], 1.0)
        self.assertEqual(agg["DEDUCTION_INVITING"]["n"], 2)

    def test_transcript_writes(self):
        import tempfile
        pairs = [{"id": 0, "category": "DEDUCTION_INVITING", "prompt": "p",
                  "base_think": "Maybe.", "base_answer": "a",
                  "ft_think": "Therefore, clearly.", "ft_answer": "b",
                  "base_profile": ts.register_profile("Maybe."),
                  "ft_profile": ts.register_profile("Therefore, clearly.")}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.md"
            ts.write_transcript(pairs, out, {"base": "b", "adapter": "a",
                                             "max_new_tokens": 900})
            txt = out.read_text()
            self.assertIn("BASE think", txt)
            self.assertIn("FINE-TUNED think", txt)
            self.assertIn("Therefore, clearly.", txt)


if __name__ == "__main__":
    unittest.main()
