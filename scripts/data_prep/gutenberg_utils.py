"""
Shared utilities for downloading and cleaning Project Gutenberg plain-text files.

Functions
---------
download(url, dest)
    Fetch a URL and write the response body to dest, returning the text.

strip_boilerplate(text)
    Remove the Gutenberg legal header and footer, returning the content body.

normalize_whitespace(text)
    Strip trailing spaces and collapse long runs of blank lines.
"""

import re
import sys
from pathlib import Path

import requests

# Gutenberg wraps every book with these sentinel lines.
START_RE = re.compile(r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.IGNORECASE)
END_RE   = re.compile(r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK",   re.IGNORECASE)


def download(url: str, dest: Path) -> str:
    print(f"Downloading {url}")
    resp = requests.get(
        url,
        headers={"User-Agent": "sherlock-investigates research (non-commercial, one-time fetch)"},
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.text
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    print(f"  Saved raw ({len(text):,} chars) -> {dest}")
    return text


def strip_boilerplate(text: str) -> str:
    lines = text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if start is None and START_RE.search(line):
            start = i + 1
        elif start is not None and END_RE.search(line):
            end = i
            break
    if start is None:
        sys.exit("ERROR: Could not find Gutenberg START marker. The file format may have changed.")
    if end is None:
        sys.exit("ERROR: Could not find Gutenberg END marker. The file format may have changed.")
    return "\n".join(lines[start:end])


def normalize_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]

    # Collapse runs of more than 2 consecutive blank lines to exactly 2.
    normalized: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                normalized.append(line)
        else:
            blank_run = 0
            normalized.append(line)

    while normalized and normalized[0] == "":
        normalized.pop(0)
    while normalized and normalized[-1] == "":
        normalized.pop()

    return "\n".join(normalized) + "\n"
