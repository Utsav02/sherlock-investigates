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


class TestParseDisambig(unittest.TestCase):
    """Scenario disambiguation: a cue set admitting >1 equally-valid answer is a
    defective training item under any verifier."""

    _AMB = ("CANDIDATES:\n- a church organist\n- a drummer\n"
            "VERDICT: AMBIGUOUS\nBEST: NONE\n"
            "REASON: both explain limb independence equally.")
    _CLEAR = ("CANDIDATES:\n- a beekeeper\n"
              "VERDICT: CLEAR\nBEST: a beekeeper\nREASON: stings and smoke.")

    def test_ambiguous_verdict_and_candidates(self):
        amb, cands, best, reason = rs.parse_disambig(self._AMB)
        self.assertTrue(amb)
        self.assertEqual(cands, ["a church organist", "a drummer"])
        self.assertEqual(best, "")          # BEST: NONE normalises to empty
        self.assertIn("equally", reason)

    def test_clear_verdict(self):
        amb, cands, best, _ = rs.parse_disambig(self._CLEAR)
        self.assertFalse(amb)
        self.assertEqual(cands, ["a beekeeper"])
        self.assertEqual(best, "a beekeeper")

    def test_markdown_fence_and_asterisk_bullets(self):
        amb, cands, _, _ = rs.parse_disambig(
            "```\nCANDIDATES:\n* a tailor\n* a bookbinder\nVERDICT: AMBIGUOUS\n```")
        self.assertTrue(amb)
        self.assertEqual(cands, ["a tailor", "a bookbinder"])

    def test_unparseable_fails_closed_as_none(self):
        amb, _, _, _ = rs.parse_disambig("Hard to say without more detail.")
        self.assertIsNone(amb)

    def test_candidate_list_stops_at_verdict_line(self):
        # VERDICT/BEST/REASON must never be swallowed as candidates.
        _, cands, _, _ = rs.parse_disambig(self._AMB)
        self.assertNotIn("VERDICT: AMBIGUOUS", cands)
        self.assertEqual(len(cands), 2)


class TestDisambigPrompt(unittest.TestCase):
    def test_prompt_contains_cues_and_never_the_ground_truth(self):
        cues = ["stings on the forearms", "a sweet smoky smell"]
        p = rs.build_disambig_prompt(cues)
        for c in cues:
            self.assertIn(c, p)
        # The check must be blind to the answer or it rationalises toward it.
        self.assertNotIn("beekeeper", p.lower())


class TestParseDisambigRealFailures(unittest.TestCase):
    """Replies that actually broke the first parser (2026-08-15 run). BEST is
    authoritative: the verdict WORD proved unreliable on its own."""

    def test_tie_verdict_word_counts_as_ambiguous(self):
        # Model used "TIE", outside the requested CLEAR|AMBIGUOUS vocabulary;
        # the first parser returned None and hid a genuinely ambiguous scenario.
        raw = ("CANDIDATES:\n- Watchmaker\n- Jeweler\n- Gemologist\n"
               "VERDICT: TIE\nBEST: NONE\n"
               "REASON: all seat a loupe and pinch tiny objects.")
        amb, cands, best, _ = rs.parse_disambig(raw)
        self.assertTrue(amb)
        self.assertEqual(best, "")
        self.assertEqual(len(cands), 3)

    def test_clear_but_best_none_is_ambiguous(self):
        # Self-contradictory reply seen twice: the CLASS is clear but no single
        # candidate wins (violinist/violist). No single best => ambiguous.
        raw = ("CANDIDATES:\n- Violinist\n- Violist\nVERDICT: CLEAR\nBEST: NONE\n"
               "REASON: violin and viola are held and fingered identically.")
        amb, _, best, _ = rs.parse_disambig(raw)
        self.assertTrue(amb)
        self.assertEqual(best, "")

    def test_clear_with_named_best_is_not_ambiguous(self):
        amb, _, best, _ = rs.parse_disambig(
            "CANDIDATES:\n- Organist\n- Drummer\nVERDICT: CLEAR\nBEST: Organist\n"
            "REASON: only the organist explains column-reading and pedal wear.")
        self.assertFalse(amb)
        self.assertEqual(best, "Organist")

    def test_na_best_normalises_to_ambiguous(self):
        amb, _, best, _ = rs.parse_disambig(
            "CANDIDATES:\n- a\n- b\nVERDICT: CLEAR\nBEST: N/A")
        self.assertTrue(amb)
        self.assertEqual(best, "")

    def test_best_line_alone_is_enough_to_decide(self):
        # No VERDICT line at all, but a named BEST => readable, not None.
        amb, _, best, _ = rs.parse_disambig(
            "CANDIDATES:\n- Beekeeper\nBEST: Beekeeper")
        self.assertFalse(amb)
        self.assertEqual(best, "Beekeeper")

    def test_neither_verdict_nor_best_is_none(self):
        amb, _, _, _ = rs.parse_disambig("I can't tell from these cues.")
        self.assertIsNone(amb)
