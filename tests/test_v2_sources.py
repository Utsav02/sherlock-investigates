"""
Tests for the v2 source layer: the OSF fetcher's pure logic, and — when the
source has been downloaded — that the immutable copy still matches its manifest.

The manifest checks skip themselves if `v2/data/sources/jones_bergen_2025/` is
absent, so `make test` stays network-free and green on a fresh clone. When the
data IS present they are the guard that "read-only on the downloaded files" is
still true. Run with `make test`.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2" / "scripts"))

import fetch_osf_source as fetcher

SOURCE_DIR = REPO_ROOT / "v2" / "data" / "sources" / "jones_bergen_2025"
MANIFEST = SOURCE_DIR / "MANIFEST.json"

REQUIRED_MANIFEST_FIELDS = (
    "source_name",
    "osf_node",
    "osf_page_url",
    "download_date_utc",
    "file_count",
    "total_bytes",
    "files",
)


class TestFileSelection(unittest.TestCase):
    def setUp(self):
        self.files = [
            {"path": "img/plot.png"},
            {"path": "data/tt_game.csv"},
            {"path": "codebook.md"},
        ]

    def test_img_excluded_by_default(self):
        fetch, skip = fetcher.select_files(self.files, include_img=False)
        self.assertEqual([f["path"] for f in skip], ["img/plot.png"])
        self.assertEqual(len(fetch), 2)

    def test_include_img_keeps_everything(self):
        fetch, skip = fetcher.select_files(self.files, include_img=True)
        self.assertEqual(len(fetch), 3)
        self.assertEqual(skip, [])


class TestSha256(unittest.TestCase):
    def test_hashes_file_contents(self):
        import hashlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.txt"
            path.write_bytes(b"three-party")
            self.assertEqual(
                fetcher.sha256_file(path),
                hashlib.sha256(b"three-party").hexdigest(),
            )


@unittest.skipUnless(MANIFEST.exists(), "source not downloaded (make v2-fetch-3p)")
class TestDownloadedSourceUnchanged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())

    def test_manifest_has_every_required_field(self):
        for field in REQUIRED_MANIFEST_FIELDS:
            self.assertIn(field, self.manifest)

    def test_every_file_still_matches_its_recorded_hash(self):
        for record in self.manifest["files"]:
            path = SOURCE_DIR / record["path"]
            with self.subTest(path=record["path"]):
                self.assertTrue(path.exists(), f"missing {path}")
                self.assertEqual(path.stat().st_size, record["size_local"])
                self.assertEqual(fetcher.sha256_file(path), record["sha256_local"])

    def test_local_hashes_matched_osf(self):
        mismatched = [
            r["path"] for r in self.manifest["files"]
            if r["sha256_matches_osf"] is False
        ]
        self.assertEqual(mismatched, [])

    def test_no_derived_data_written_into_the_source_tree(self):
        allowed_suffixes = {".csv", ".txt", ".md", ".Rmd", ".json"}
        for path in SOURCE_DIR.rglob("*"):
            if path.is_file():
                self.assertIn(path.suffix, allowed_suffixes, f"unexpected {path}")


if __name__ == "__main__":
    unittest.main()
