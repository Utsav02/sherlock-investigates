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


class TestSplitThink(unittest.TestCase):
    """The Claude backend's parser. Claude does not emit <think> natively, so
    the format is prompt-enforced and this is what verifies it landed — a
    regression here silently feeds malformed traces into the SFT set."""

    def test_clean_format(self):
        think, ans = gt.split_think(
            "<think>A implies B.</think>\nThis is a locksmith.")
        self.assertEqual(think, "A implies B.")
        self.assertEqual(ans, "This is a locksmith.")

    def test_strips_markdown_fence(self):
        think, ans = gt.split_think(
            "```\n<think>A implies B.</think>\nThis is a locksmith.\n```")
        self.assertEqual(think, "A implies B.")
        self.assertEqual(ans, "This is a locksmith.")

    def test_prefers_decisive_sentence_over_trailing_chatter(self):
        think, ans = gt.split_think(
            "<think>A.</think>\nSome preamble.\nThis is a locksmith.\n"
            "Hope that helps!")
        self.assertEqual(ans, "This is a locksmith.")

    def test_unclosed_block_yields_no_think(self):
        # Must NOT return reasoning as the answer: has_think=False keeps the
        # row out of the SFT set instead of training on a malformed trace.
        think, _ = gt.split_think("<think>A implies B but never closes")
        self.assertEqual(think, "")

    def test_missing_tags_yields_no_think(self):
        think, ans = gt.split_think("This is a locksmith.")
        self.assertEqual(think, "")
        self.assertEqual(ans, "This is a locksmith.")


class TestClaudePrompt(unittest.TestCase):
    def test_prompt_demands_format_and_ends_with_scenario(self):
        p = gt.build_claude_prompt("A man waits. What do you make of them?")
        self.assertIn("<think>", p)
        self.assertIn("This is", p)
        self.assertEqual(p.count("EXAMPLE"), 2)   # both exemplars rendered
        self.assertTrue(p.rstrip().endswith("What do you make of them?"))

    def test_exemplars_put_answer_on_its_own_line(self):
        # The rendered demonstration must model the exact target shape.
        p = gt.build_claude_prompt("x")
        self.assertIn("</think>\n", p)
