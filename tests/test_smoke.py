"""
Smoke tests for the pure logic in the data-prep pipeline.

No network, no Ollama, no GPU: these exercise the text cleaning, paragraph
chunking, label parsing, framing selection, and cache-key versioning that the
pipeline scripts share. Run with `make test`.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data_prep"))

import augment_corpus
import classify_chunks
import gutenberg_utils
from chunk_stories import split_paragraphs


class TestGutenbergCleaning(unittest.TestCase):
    def test_strip_boilerplate_keeps_only_body(self):
        text = (
            "Project Gutenberg legal preamble\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK A STUDY IN SCARLET ***\n"
            "Chapter I.\n"
            "Mr. Sherlock Holmes\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK A STUDY IN SCARLET ***\n"
            "More legal text\n"
        )
        body = gutenberg_utils.strip_boilerplate(text)
        self.assertEqual(body, "Chapter I.\nMr. Sherlock Holmes")

    def test_normalize_whitespace_collapses_blank_runs(self):
        raw = "\n\nfirst line   \n\n\n\n\nsecond line\n\n\n"
        self.assertEqual(
            gutenberg_utils.normalize_whitespace(raw),
            "first line\n\n\nsecond line\n",
        )


class TestChunking(unittest.TestCase):
    def test_split_paragraphs_on_blank_lines_only(self):
        text = "para one\nstill para one\n\npara two\n\n\npara three"
        self.assertEqual(
            split_paragraphs(text),
            ["para one\nstill para one", "para two", "para three"],
        )


class TestLabelParsing(unittest.TestCase):
    def test_parses_well_formed_response(self):
        raw = (
            "LABEL: central\n"
            "JUSTIFICATION: Holmes explains his inference chain step by step."
        )
        label = classify_chunks.LABEL_RE.search(raw)
        just = classify_chunks.JUSTIFICATION_RE.search(raw)
        self.assertEqual(label.group(1).lower(), "central")
        self.assertEqual(
            just.group(1).strip(),
            "Holmes explains his inference chain step by step.",
        )

    def test_label_match_is_case_insensitive(self):
        match = classify_chunks.LABEL_RE.search("label: Minor")
        self.assertEqual(match.group(1).lower(), "minor")

    def test_rejects_unknown_label(self):
        self.assertIsNone(classify_chunks.LABEL_RE.search("LABEL: dramatic"))


class TestFramingSelection(unittest.TestCase):
    def test_long_central_chunk_gets_all_framings(self):
        chunk = {"label": "central", "word_count": 150}
        self.assertEqual(
            augment_corpus.framings_for(chunk),
            ["VERBATIM", "WATSON", "QA", "CHAIN", "REVERSE"],
        )

    def test_short_central_chunk_skips_reverse(self):
        chunk = {"label": "central", "word_count": 50}
        self.assertEqual(
            augment_corpus.framings_for(chunk),
            ["VERBATIM", "WATSON", "QA", "CHAIN"],
        )

    def test_short_minor_chunk_is_verbatim_only(self):
        chunk = {"label": "minor", "word_count": 12}
        self.assertEqual(augment_corpus.framings_for(chunk), ["VERBATIM"])


class TestCacheVersioning(unittest.TestCase):
    def test_same_inputs_same_key(self):
        a = augment_corpus._cache_key("qwen2.5:7b", "QA", "some passage")
        b = augment_corpus._cache_key("qwen2.5:7b", "QA", "some passage")
        self.assertEqual(a, b)

    def test_bumping_augment_version_invalidates_key(self):
        # AUGMENT_VERSION must gate the cache, as the module docstring claims.
        before = augment_corpus._cache_key("qwen2.5:7b", "QA", "some passage")
        original = augment_corpus.AUGMENT_VERSION
        try:
            augment_corpus.AUGMENT_VERSION = original + "-bumped"
            after = augment_corpus._cache_key("qwen2.5:7b", "QA", "some passage")
        finally:
            augment_corpus.AUGMENT_VERSION = original
        self.assertNotEqual(before, after)

    def test_classify_prompt_version_gates_key(self):
        before = classify_chunks.cache_key("qwen2.5:7b", "some passage")
        original = classify_chunks.PROMPT_VERSION
        try:
            classify_chunks.PROMPT_VERSION = original + "-bumped"
            after = classify_chunks.cache_key("qwen2.5:7b", "some passage")
        finally:
            classify_chunks.PROMPT_VERSION = original
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
