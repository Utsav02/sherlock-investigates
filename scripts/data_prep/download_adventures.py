#!/usr/bin/env python3
"""
Download The Adventures of Sherlock Holmes (Gutenberg ebook #1661),
split into individual stories, and save three specific stories to disk.

Outputs
-------
  data/raw/adventures_of_sherlock_holmes.txt       - untouched Gutenberg download
  data/processed/training/scandal_in_bohemia.txt   - A Scandal in Bohemia
  data/processed/training/red_headed_league.txt    - The Red-Headed League
  data/processed/heldout/speckled_band.txt         - The Adventure of the Speckled Band

Run from the repo root:
  python scripts/data_prep/download_adventures.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gutenberg_utils import download, strip_boilerplate, normalize_whitespace  # noqa: E402

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
RAW_PATH = Path("data/raw/adventures_of_sherlock_holmes.txt")

# Stories to extract and where to save them.
# Keys are normalized upper-case story titles as they appear in the Gutenberg heading.
TARGETS = {
    "A SCANDAL IN BOHEMIA": Path("data/processed/training/scandal_in_bohemia.txt"),
    "THE RED-HEADED LEAGUE": Path("data/processed/training/red_headed_league.txt"),
    "THE ADVENTURE OF THE SPECKLED BAND": Path("data/processed/heldout/speckled_band.txt"),
}

# Matches lines like:  I. A SCANDAL IN BOHEMIA
# The title is captured in group 2.
# Sub-section markers within stories (bare "I.", "II.") do not match
# because they have no title text after the period.
HEADING_RE = re.compile(
    r"^([IVXLCDM]+)\.\s+(\S.+)$",
    re.IGNORECASE,
)


def split_into_stories(text: str) -> dict[str, str]:
    """
    Parse the Adventures body text into a dict mapping upper-case story title
    to story body text (excluding the heading line itself).

    The Gutenberg text separates stories with lines of the form:
        ADVENTURE I. A SCANDAL IN BOHEMIA
    Within each story, Roman-numeral section markers appear as bare "I.", "II."
    lines and do NOT match the heading pattern.
    """
    lines = text.splitlines()

    # First pass: locate every heading line and record (title, line_index).
    boundaries: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line.strip())
        if m:
            title = m.group(2).strip().upper()
            boundaries.append((title, i))

    if not boundaries:
        sys.exit(
            "ERROR: No story headings found. "
            "The file format may differ from expected. "
            "Inspect data/raw/adventures_of_sherlock_holmes.txt for actual heading style."
        )

    # The Gutenberg text has a table-of-contents section that lists each title
    # on a single line in the same heading format.  Those TOC lines produce tiny
    # "bodies" (often empty) between consecutive headings.  The actual story bodies
    # always appear as the LAST occurrence of each title in the file, so iterating
    # forward and letting later entries overwrite earlier ones in the dict gives us
    # only the real story bodies.
    print(f"  Detected {len(boundaries)} heading lines "
          f"({len({t for t, _ in boundaries})} unique titles, TOC + body):")
    for title, lineno in boundaries:
        print(f"    line {lineno:5d}: {title}")

    # Second pass: slice the text between consecutive headings.
    # For duplicate titles, the later (actual-body) entry overwrites the earlier (TOC) entry.
    stories: dict[str, str] = {}
    for idx, (title, start) in enumerate(boundaries):
        end = boundaries[idx + 1][1] if idx + 1 < len(boundaries) else len(lines)
        # Skip the heading line itself (start), include everything up to next heading.
        body_lines = lines[start + 1 : end]
        stories[title] = "\n".join(body_lines)

    return stories


def main() -> None:
    raw = download(GUTENBERG_URL, RAW_PATH)

    print("Stripping Gutenberg header and footer...")
    content = strip_boilerplate(raw)
    print(f"  Extracted {len(content):,} chars")

    print("Splitting into individual stories...")
    stories = split_into_stories(content)

    print("\nExtracting target stories:")
    for target_title, dest_path in TARGETS.items():
        if target_title not in stories:
            # Try a substring match as a fallback — handles slight title variations.
            candidates = [t for t in stories if target_title in t or t in target_title]
            if len(candidates) == 1:
                target_title_actual = candidates[0]
                print(f"  [fuzzy match] '{target_title}' -> '{target_title_actual}'")
            else:
                available = ", ".join(stories.keys())
                sys.exit(
                    f"ERROR: Could not find '{target_title}' in detected stories.\n"
                    f"Available: {available}"
                )
        else:
            target_title_actual = target_title

        body = normalize_whitespace(stories[target_title_actual])
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(body, encoding="utf-8")
        word_count = len(body.split())
        print(f"  {dest_path}  ({word_count:,} words)")


if __name__ == "__main__":
    main()
