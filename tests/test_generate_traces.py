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


class TestParseJudge(unittest.TestCase):
    """The LLM judge is now the keeper GATE, so a parser regression silently
    changes which traces enter the SFT set."""

    def test_yes_verdict_with_reason(self):
        v, reason = gt.parse_judge("VERDICT: YES\nREASON: same trade, other words.")
        self.assertTrue(v)
        self.assertEqual(reason, "same trade, other words.")

    def test_no_verdict(self):
        v, reason = gt.parse_judge("VERDICT: NO\nREASON: a different identity.")
        self.assertFalse(v)
        self.assertEqual(reason, "a different identity.")

    def test_lowercase_and_dash_separator(self):
        v, _ = gt.parse_judge("verdict - yes\nreason - close enough")
        self.assertTrue(v)

    def test_markdown_fence_tolerated(self):
        v, _ = gt.parse_judge("```\nVERDICT: YES\nREASON: ok\n```")
        self.assertTrue(v)

    def test_bare_yes_without_label(self):
        v, _ = gt.parse_judge("YES")
        self.assertTrue(v)

    def test_unparseable_fails_closed_as_none(self):
        # Must be None, not False: an unusable judgement has to be visible in the
        # data rather than silently counted as a rejection.
        v, reason = gt.parse_judge("I'm not sure how to grade this one.")
        self.assertIsNone(v)
        self.assertIn("not sure", reason)

    def test_empty_input_is_none(self):
        v, _ = gt.parse_judge("")
        self.assertIsNone(v)

    def test_verdict_word_in_reason_does_not_flip_it(self):
        v, _ = gt.parse_judge(
            "VERDICT: NO\nREASON: the answer says yes to a different trade.")
        self.assertFalse(v)


class TestJudgePrompt(unittest.TestCase):
    def test_prompt_carries_both_sides_and_demands_the_form(self):
        p = gt.build_judge_prompt("a beekeeper", "This is a beekeeper.")
        self.assertIn("a beekeeper", p)
        self.assertIn("VERDICT:", p)
        self.assertIn("REASON:", p)

    def test_rubric_states_the_generality_rule(self):
        # The coarser-but-consistent rule is the whole reason the judge exists.
        self.assertIn("MORE GENERAL", gt.JUDGE_SYSTEM)


class TestAgreementReport(unittest.TestCase):
    def test_counts_all_four_cells_and_unparsed(self):
        rows = [
            {"judge": True, "matched": True},    # agree keep
            {"judge": True, "matched": False},   # judge recovers
            {"judge": False, "matched": False},  # agree drop
            {"judge": None, "matched": False},   # unparseable
        ]
        out = gt.agreement_report(rows)
        self.assertIn("n=4", out)
        self.assertIn("unparseable judge replies: 1", out)
        self.assertIn("agreement: 2/4", out)
