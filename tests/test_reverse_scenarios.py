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


class TestIdentityPool(unittest.TestCase):
    """Identity-pool expansion: parsing, dedup, and the style filter."""

    def test_parses_plain_lines(self):
        got = rs.parse_identity_list(
            "a chimney sweep\na farrier\nsomeone who has just left the navy\n")
        self.assertEqual(got, ["a chimney sweep", "a farrier",
                               "someone who has just left the navy"])

    def test_strips_bullets_numbering_and_trailing_punctuation(self):
        got = rs.parse_identity_list("1. a farrier.\n- an oyster shucker;\n* a drayman")
        self.assertEqual(got, ["a farrier", "an oyster shucker", "a drayman"])

    def test_drops_commentary_and_headers(self):
        # Preamble, headers and prose must not enter the pool as identities.
        got = rs.parse_identity_list(
            "Here are 3 answers:\nCATEGORY:\na farrier\nThese all rely on cues.")
        self.assertEqual(got, ["a farrier"])

    def test_exact_duplicate_dropped_ignoring_article_and_case(self):
        self.assertEqual(rs.dedup_identities(["A Farrier"], ["a farrier"]), [])

    def test_near_duplicate_dropped(self):
        self.assertEqual(
            rs.dedup_identities(["a professional concert violinist"],
                                ["a concert violinist professional"]), [])

    def test_distinct_trades_survive(self):
        # Two similar-but-different trades must BOTH survive: each scenario is
        # disambiguated on its own cues, so the pool need not be semantically
        # spread, only non-redundant.
        got = rs.dedup_identities(["a watchmaker"], ["a diamond setter"])
        self.assertEqual(got, ["a watchmaker"])

    def test_dedups_within_the_new_batch(self):
        got = rs.dedup_identities(["a farrier", "a farrier", "a drayman"], [])
        self.assertEqual(got, ["a farrier", "a drayman"])


class TestClassifyRow(unittest.TestCase):
    """Terminal status of a ledger row — the report reads only this."""

    def _row(self, **kw):
        base = {"parse_ok": True, "leak": False, "ambiguous": False}
        base.update(kw)
        return base

    def test_usable(self):
        self.assertEqual(rs.classify_row(self._row()), "usable")

    def test_error_outranks_everything(self):
        self.assertEqual(
            rs.classify_row(self._row(error="RuntimeError: boom")), "error")

    def test_badfmt_before_leak(self):
        self.assertEqual(rs.classify_row(self._row(parse_ok=False, leak=True)),
                         "badfmt")

    def test_leak(self):
        self.assertEqual(rs.classify_row(self._row(leak=True)), "leak")

    def test_ambiguous(self):
        self.assertEqual(rs.classify_row(self._row(ambiguous=True)), "ambiguous")

    def test_unreadable_disambig_fails_closed(self):
        # ambiguous=None must NOT be silently kept — it gets its own status.
        self.assertEqual(rs.classify_row(self._row(ambiguous=None)),
                         "unparsed_disambig")


class TestResumeAndSummary(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Path(self.tmp.name) / "ledger.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_done_missing_file_is_empty(self):
        self.assertEqual(rs.load_done(self.p), set())

    def test_load_done_reads_every_attempted_identity(self):
        # Dropped rows count as DONE — otherwise a resume re-pays for every
        # identity the gates already rejected.
        self.p.write_text(
            '{"ground_truth": "a farrier", "status": "usable"}\n'
            '{"ground_truth": "a drayman", "status": "leak"}\n')
        self.assertEqual(rs.load_done(self.p), {"a farrier", "a drayman"})

    def test_load_done_survives_a_torn_final_line(self):
        # A kill mid-append leaves a partial line; it must cost one identity,
        # not the whole resume.
        self.p.write_text('{"ground_truth": "a farrier"}\n{"ground_tru')
        self.assertEqual(rs.load_done(self.p), {"a farrier"})

    def test_summary_rates_use_the_right_denominators(self):
        rows = [{"status": "usable"}, {"status": "usable"},
                {"status": "ambiguous"}, {"status": "leak"},
                {"status": "badfmt"}]
        s = rs.summarize(rows)
        self.assertEqual(s["tried"], 5)
        self.assertEqual(s["usable"], 2)
        # Ambiguity rate is over scenarios that REACHED the check (3), not all 5:
        # a leaked or malformed scenario was never disambiguated.
        self.assertEqual(s["reached_disambig"], 3)
        self.assertAlmostEqual(s["ambiguous_rate"], 1 / 3)
        self.assertAlmostEqual(s["overall_yield"], 2 / 5)

    def test_summary_counts_regenerations_and_rescues(self):
        rows = [{"status": "usable", "attempts": 2},
                {"status": "ambiguous", "attempts": 2},
                {"status": "usable", "attempts": 1}]
        s = rs.summarize(rows)
        self.assertEqual(s["regenerated"], 2)
        self.assertEqual(s["regen_rescued"], 1)

    def test_summary_empty_is_safe(self):
        s = rs.summarize([])
        self.assertIsNone(s["ambiguous_rate"])
        self.assertIsNone(s["overall_yield"])


class TestLeakVariantMatching(unittest.TestCase):
    """Regression tests for the prefix rule that rejected clean scenarios.

    Measured on a real generation (2026-08-15): a blacksmith scenario that never
    names the trade was rejected because a cue said "blackened" creases and the
    old rule asked whether any TEXT token starts with the answer's first four
    letters ('blac'). The direction of the test was wrong.
    """

    def test_blackened_does_not_leak_blacksmith(self):
        leak, terms = rs.detect_leak(
            "a blacksmith",
            ["Deep, blackened creases in the pads and knuckles",
             "Forearms noticeably thicker and more corded"],
            "His knuckles are etched with dark lines. What do you make of them?")
        self.assertFalse(leak, f"false positive on {terms}")

    def test_lock_still_leaks_locksmith(self):
        # A signature tool naming the answer's first component must still trip.
        leak, terms = rs.detect_leak("a locksmith", ["holds a lockpick"],
                                     "He fiddles with a lock. What do you make of them?")
        self.assertTrue(leak)
        self.assertIn("locksmith", terms)

    def test_violin_still_leaks_violinist(self):
        leak, terms = rs.detect_leak("a professional concert violinist",
                                     ["a violin case in one hand"],
                                     "They carry it close. What do you make of them?")
        self.assertTrue(leak)
        self.assertIn("violinist", terms)

    def test_morphological_variant_leaks(self):
        # 'gardening' ~ 'gardener' via the stem rule.
        leak, _ = rs.detect_leak("a professional gardener",
                                 ["soil under the nails from gardening"],
                                 "What do you make of them?")
        self.assertTrue(leak)

    def test_unrelated_long_word_sharing_a_prefix_is_clean(self):
        # 'ministering' must not leak 'a coal miner'.
        leak, terms = rs.detect_leak(
            "a coal miner", ["a stooped, ministering posture toward his wife"],
            "He stoops as he walks. What do you make of them?")
        self.assertFalse(leak, f"false positive on {terms}")


class TestDriftFlags(unittest.TestCase):
    """Answer-drift review flag: BEST vs the seed label, computed offline."""

    def _row(self, gt, best, **kw):
        r = {"ground_truth": gt, "status": "usable", "disambig_best": best}
        r.update(kw)
        return r

    def test_matching_best_is_not_flagged(self):
        d = rs.drift_flags([self._row("a watchmaker", "a watchmaker")])
        self.assertEqual(d["n_flagged"], 0)

    def test_divergent_best_is_flagged(self):
        # The observed case: regeneration narrowed the answer away from the seed.
        d = rs.drift_flags([self._row(
            "a professional concert violinist",
            "Concertmaster / orchestra leader", attempts=2)])
        self.assertEqual(d["n_flagged"], 1)
        self.assertEqual(d["regen_flagged"], 1)
        self.assertEqual(d["regen_total"], 1)

    def test_only_usable_rows_are_considered(self):
        d = rs.drift_flags([{"ground_truth": "a tanner", "status": "leak",
                             "disambig_best": "a wholly different trade"}])
        self.assertEqual(d["n_flagged"], 0)

    def test_missing_best_is_not_flagged(self):
        # No BEST recorded is absence of evidence, not evidence of drift.
        d = rs.drift_flags([self._row("a cooper", "")])
        self.assertEqual(d["n_flagged"], 0)

    def test_counts_regenerated_rows_even_when_unflagged(self):
        d = rs.drift_flags([self._row("a farrier", "a farrier", attempts=2)])
        self.assertEqual(d["regen_total"], 1)
        self.assertEqual(d["regen_flagged"], 0)
