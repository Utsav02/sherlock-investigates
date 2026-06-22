#!/usr/bin/env python3
"""
Download the complete Sherlock Holmes canon from Project Gutenberg.

Produces:
  data/raw/canon/{slug}.txt            - untouched Gutenberg downloads
  data/processed/full_canon/{slug}.txt - header/footer stripped, whitespace normalized

Gutenberg IDs verified against the cache CDN as of June 2026.
The pilot corpus (A Study in Scarlet, Adventures) is already in data/raw/ and
data/processed/ — this script writes to separate canon/ subdirectories so the
pilot paths are unaffected.

Run from the repo root:
  python scripts/data_prep/download_full_canon.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gutenberg_utils import download, strip_boilerplate, normalize_whitespace  # noqa: E402

# All canonical Sherlock Holmes works by Conan Doyle available on Gutenberg.
# Format: (gutenberg_id, slug, human_readable_title)
# Novels and collections are all included; the extraction/augmentation pipeline
# will later filter for deductive-reasoning-heavy passages.
CANON = [
    # Novels
    (244,   "study_in_scarlet",          "A Study in Scarlet"),
    (2097,  "sign_of_the_four",          "The Sign of the Four"),
    (2852,  "hound_of_the_baskervilles", "The Hound of the Baskervilles"),
    (3289,  "valley_of_fear",            "The Valley of Fear"),
    # Short story collections
    (1661,  "adventures",                "The Adventures of Sherlock Holmes"),
    (834,   "memoirs",                   "The Memoirs of Sherlock Holmes"),
    (108,   "return",                    "The Return of Sherlock Holmes"),
    (2350,  "his_last_bow",              "His Last Bow"),
    (69070, "case_book",                 "The Case-Book of Sherlock Holmes"),
]

RAW_DIR       = Path("data/raw/canon")
PROCESSED_DIR = Path("data/processed/full_canon")

CACHE_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    total_words = 0
    failed = []

    for gid, slug, title in CANON:
        url = CACHE_URL.format(id=gid)
        raw_path = RAW_DIR / f"{slug}.txt"
        proc_path = PROCESSED_DIR / f"{slug}.txt"

        if proc_path.exists():
            words = len(proc_path.read_text(encoding="utf-8").split())
            print(f"  [skip] {title} already at {proc_path} ({words:,} words)")
            total_words += words
            continue

        print(f"\n--- {title} (#{gid}) ---")
        try:
            raw = download(url, raw_path)
        except Exception as e:
            print(f"  ERROR downloading {url}: {e}", file=sys.stderr)
            failed.append((gid, slug, title, str(e)))
            continue

        try:
            content = strip_boilerplate(raw)
        except SystemExit as e:
            print(f"  ERROR stripping boilerplate for {title}: {e}", file=sys.stderr)
            failed.append((gid, slug, title, str(e)))
            continue

        cleaned = normalize_whitespace(content)
        proc_path.write_text(cleaned, encoding="utf-8")
        words = len(cleaned.split())
        total_words += words
        print(f"  Processed -> {proc_path} ({words:,} words)")

    print(f"\n=== Done ===")
    print(f"Total words in full_canon/: {total_words:,}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for gid, slug, title, err in failed:
            print(f"  #{gid} {title}: {err}")
        print(
            "\nIf a Gutenberg ID failed, verify the correct ID at "
            "https://www.gutenberg.org/ebooks/search/?query=sherlock+holmes+doyle"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
