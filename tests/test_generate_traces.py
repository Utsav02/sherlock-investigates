"""Tests for the trace ground-truth match filter (pure logic, no Ollama)."""

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts" / "data_prep"))
sys.path.insert(0, str(_ROOT / "scripts" / "eval"))

import generate_traces as gt


class TestAnswerMatches(unittest.TestCase):
    def test_correct_answer_matches(self):
        m, terms = gt.answer_matches(
            "a professional concert violinist",
            "This is a violinist who plays in a concert orchestra.")
        self.assertTrue(m)
        self.assertIn("violinist", terms)

    def test_morphological_variant_matches(self):
        # 'violin' should count as reaching 'violinist'.
        m, _ = gt.answer_matches("a professional concert violinist",
                                 "This is a violin player.")
        self.assertTrue(m)

    def test_wrong_answer_rejected(self):
        m, terms = gt.answer_matches(
            "a night-shift hospital nurse coming off a long shift",
            "This is a long-haul taxi driver just finishing work.")
        self.assertFalse(m)
        self.assertEqual(terms, [])

    def test_partial_identity_word_matches(self):
        m, _ = gt.answer_matches("a retired sergeant of the Royal Marines",
                                 "This is a former marine, judging by the bearing.")
        self.assertTrue(m)  # 'marine' ~ 'marines'


if __name__ == "__main__":
    unittest.main()
