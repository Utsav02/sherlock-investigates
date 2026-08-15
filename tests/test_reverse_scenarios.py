"""Tests for the reverse-construction leak detector (pure logic, no Ollama)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data_prep"))

import reverse_scenarios as rs


class TestDetectLeak(unittest.TestCase):
    def test_catches_observed_nurse_leak(self):
        # The real first-cut leak: scenario "mentions working night shifts".
        leak, terms = rs.detect_leak(
            "a night-shift hospital nurse coming off a long shift",
            ["tired eyes"], "She mentions working night shifts. What do you make of them?")
        self.assertTrue(leak)
        self.assertTrue(any(t in ("night", "shift", "hospital", "nurse") for t in terms))

    def test_catches_violin_variant(self):
        # 'violin' must trip on ground truth 'violinist' (morphological variant).
        leak, terms = rs.detect_leak(
            "a professional concert violinist",
            ["a violin case in one hand"], "They carry a violin case. What do you make of them?")
        self.assertTrue(leak)
        self.assertIn("violinist", terms)

    def test_clean_scenario_passes(self):
        # Indirect cues, no answer words -> not flagged.
        leak, terms = rs.detect_leak(
            "a retired sergeant of the Royal Marines",
            ["ramrod-straight posture", "a faded anchor tattoo on the forearm",
             "boots polished to a mirror shine"],
            "A grey-haired man stands rigidly upright at the bus stop, boots gleaming. "
            "What do you make of them?")
        self.assertFalse(leak)
        self.assertEqual(terms, [])

    def test_generic_words_not_flagged(self):
        # 'professional'/'long' etc. are stopped, so a scenario using them is fine.
        leak, _ = rs.detect_leak(
            "a professional concert violinist",
            ["long delicate fingers"], "A person with long fingers. What do you make of them?")
        self.assertFalse(leak)

    def test_returns_terms_list(self):
        leak, terms = rs.detect_leak("a locksmith", ["holds a lockpick"],
                                     "He fiddles with a lock. What do you make of them?")
        self.assertTrue(leak)
        self.assertIn("locksmith", terms)


if __name__ == "__main__":
    unittest.main()
