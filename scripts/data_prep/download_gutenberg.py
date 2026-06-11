#!/usr/bin/env python3
"""
Download A Study in Scarlet from Project Gutenberg.

Produces two files:
  data/raw/study_in_scarlet.txt      - untouched Gutenberg download
  data/processed/study_in_scarlet.txt - header/footer stripped, whitespace normalized

Run from the repo root:
  python scripts/data_prep/download_gutenberg.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gutenberg_utils import download, strip_boilerplate, normalize_whitespace  # noqa: E402

# Project Gutenberg ebook #244, UTF-8 plain text via the cache CDN.
# Note: ebook #1661 is "The Adventures of Sherlock Holmes" (short stories),
# downloaded separately by download_adventures.py.
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/244/pg244.txt"

RAW_PATH = Path("data/raw/study_in_scarlet.txt")
PROCESSED_PATH = Path("data/processed/study_in_scarlet.txt")


def main() -> None:
    raw = download(GUTENBERG_URL, RAW_PATH)

    print("Stripping Gutenberg header and footer...")
    content = strip_boilerplate(raw)
    print(f"  Extracted {len(content):,} chars of novel content")

    print("Normalizing whitespace...")
    cleaned = normalize_whitespace(content)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.write_text(cleaned, encoding="utf-8")
    word_count = len(cleaned.split())
    print(f"  Saved processed ({word_count:,} words) -> {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
