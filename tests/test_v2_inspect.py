"""
Pure-logic tests for the v2 Stage A inspection helpers.

No network, no source files required: these exercise the measurement code that
`data_inspection.md`'s numbers depend on — the distribution summary, the
transcript parser, the id normalisation that the 15-minute release needs, the
PII screens, and the candidate-length-unit counting used to identify Inverse
Turing Bench's filter. Run with `make test`.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2" / "scripts"))

import inspect_three_party as insp


class TestSummarize(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(insp.summarize([]), {"n": 0})

    def test_single_value_has_zero_sd(self):
        out = insp.summarize([7.0])
        self.assertEqual(out["n"], 1)
        self.assertEqual(out["sd"], 0.0)
        self.assertEqual(out["median"], 7.0)

    def test_quartiles_are_nearest_rank(self):
        out = insp.summarize([float(v) for v in range(1, 101)])
        self.assertEqual(out["min"], 1.0)
        self.assertEqual(out["p25"], 25.0)
        self.assertEqual(out["median"], 50.0)
        self.assertEqual(out["p90"], 90.0)
        self.assertEqual(out["max"], 100.0)

    def test_percentile_rejects_empty(self):
        with self.assertRaises(ValueError):
            insp.percentile([], 50)


class TestTranscriptParsing(unittest.TestCase):
    def test_roles_and_text(self):
        turns = insp.parse_transcript("I: hello there\nW: hi\nI: who are you")
        self.assertEqual([role for role, _ in turns], ["I", "W", "I"])
        self.assertEqual(turns[0][1], "hello there")

    def test_unprefixed_line_continues_previous_turn(self):
        # A message containing a newline must not be counted as a second turn.
        turns = insp.parse_transcript("I: first line\nstill the same message\nW: reply")
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0][1], "first line\nstill the same message")

    def test_blank_lines_ignored(self):
        self.assertEqual(len(insp.parse_transcript("\n\nI: a\n\nW: b\n")), 2)

    def test_strip_role_prefixes_drops_markers(self):
        stripped = insp.strip_role_prefixes("I: hello there\nW: hi")
        self.assertEqual(stripped, "hello there hi")
        self.assertEqual(insp.word_count("I: hello there\nW: hi"), 5)
        self.assertEqual(insp.word_count(stripped), 3)


class TestNormId(unittest.TestCase):
    def test_float_written_ids_collapse(self):
        # The 15-minute release writes tt_witness.user_id as '109819.0'.
        self.assertEqual(insp.norm_id("109819.0"), "109819")
        self.assertEqual(insp.norm_id("109819"), "109819")

    def test_non_integral_and_text_left_alone(self):
        self.assertEqual(insp.norm_id("1.5"), "1.5")
        self.assertEqual(insp.norm_id("abc.0"), "abc.0")
        self.assertEqual(insp.norm_id(None), "")

    def test_blank_covers_r_missing_markers(self):
        for value in ("", " ", "NA", "NaN", "nan", "NULL", "None"):
            self.assertTrue(insp.is_blank(value), value)
        self.assertFalse(insp.is_blank("0"))
        self.assertFalse(insp.is_blank("not applicable"))


class TestPiiScreens(unittest.TestCase):
    def test_screens_fire_on_their_own_shapes(self):
        self.assertIn("email", insp.scan_pii("write to a.b+c@example.co.uk"))
        self.assertIn("url", insp.scan_pii("see https://example.com/x"))
        self.assertIn("anonymization_placeholder", insp.scan_pii("hi <NAME>, ok"))

    def test_clean_text_fires_nothing(self):
        self.assertEqual(insp.scan_pii("just chatting about the weather"), [])


class TestCandidateLengthUnits(unittest.TestCase):
    """The counting rule behind the 'length >= 50' identification."""

    def _rows(self, a: str, b: str, game: str = "1"):
        return [
            {"game_id": game, "conversation_label": "A", "transcript": a},
            {"game_id": game, "conversation_label": "B", "transcript": b},
        ]

    def test_both_sides_must_pass(self):
        long_side = "I: " + " ".join(["word"] * 60)
        short_side = "I: short"
        out = insp.candidate_length_units(
            self._rows(long_side, short_side), threshold=50
        )
        unit = out["whitespace_tokens_incl_role_prefixes"]
        self.assertEqual(unit["both_sides_pass"], 0)
        self.assertEqual(unit["either_side_passes"], 1)
        self.assertEqual(unit["sum_of_sides_passes"], 1)

    def test_role_prefixes_change_the_count(self):
        # 49 content words + one 'I:' prefix == 50 tokens with prefixes counted.
        side = "I: " + " ".join(["word"] * 49)
        out = insp.candidate_length_units(self._rows(side, side), threshold=50)
        self.assertEqual(out["whitespace_tokens_incl_role_prefixes"]["both_sides_pass"], 1)
        self.assertEqual(out["whitespace_tokens_excl_role_prefixes"]["both_sides_pass"], 0)

    def test_duplicate_rows_do_not_inflate_counts(self):
        side = "I: " + " ".join(["word"] * 60)
        rows = self._rows(side, side) + self._rows(side, side)
        out = insp.candidate_length_units(rows, threshold=50)
        self.assertEqual(out["games_with_two_transcripts"], 1)
        self.assertEqual(out["whitespace_tokens_incl_role_prefixes"]["both_sides_pass"], 1)

    def test_incomplete_games_are_excluded(self):
        rows = [{"game_id": "9", "conversation_label": "A",
                 "transcript": "I: " + " ".join(["word"] * 60)}]
        out = insp.candidate_length_units(rows, threshold=50)
        self.assertEqual(out["games_with_two_transcripts"], 0)


if __name__ == "__main__":
    unittest.main()
