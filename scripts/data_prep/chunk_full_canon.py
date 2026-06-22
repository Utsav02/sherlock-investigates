"""
Chunk all 9 full-canon works into data/processed/full_canon_chunks.jsonl.

Collection files (Adventures, Memoirs, Return, His Last Bow, Case-Book) are
split into individual stories first so the held-out story (Speckled Band) can
be excluded cleanly.  Novel files are chunked as single continuous texts.

Outputs data/processed/full_canon_chunks.jsonl — fed into classify_chunks.py
via its --input flag.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FULL_CANON_DIR = ROOT / "data/processed/full_canon"
OUTPUT         = ROOT / "data/processed/full_canon_chunks.jsonl"
MIN_WORDS      = 10

# Stories to exclude from training (held-out evaluation set).
HELD_OUT_TITLES = {"THE ADVENTURE OF THE SPECKLED BAND"}

# Roman-numeral story heading as used in all Gutenberg collection files.
# Matches lines like:  "I. A SCANDAL IN BOHEMIA"  or  "XI. THE ADVENTURE OF..."
_HEADING_RE = re.compile(r"^([IVXLCDM]+)\.\s+(\S.+)$", re.IGNORECASE)


def _split_into_stories(text: str) -> dict[str, str]:
    """Split a collection file into {UPPER_TITLE: body_text} pairs.

    Duplicate headings (TOC + body) are resolved by keeping the LAST occurrence,
    which is always the actual story body (same logic as download_adventures.py).
    """
    lines = text.splitlines()
    boundaries: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line.strip())
        if m:
            boundaries.append((m.group(2).strip().upper(), i))

    if not boundaries:
        return {}

    stories: dict[str, str] = {}
    for idx, (title, start) in enumerate(boundaries):
        end = boundaries[idx + 1][1] if idx + 1 < len(boundaries) else len(lines)
        stories[title] = "\n".join(lines[start + 1 : end])
    return stories


def _split_paragraphs(text: str) -> list[str]:
    return re.split(r"\n{2,}", text)


def _chunk_text(text: str, source_name: str, start_id: int) -> list[dict]:
    chunks = []
    cid = start_id
    for para_idx, raw in enumerate(_split_paragraphs(text)):
        content = raw.strip()
        if not content:
            continue
        wc = len(content.split())
        if wc < MIN_WORDS:
            continue
        chunks.append({
            "chunk_id":       cid,
            "source_story":   source_name,
            "paragraph_index": para_idx,
            "content":        content,
            "word_count":     wc,
        })
        cid += 1
    return chunks


# Files that are story collections (split by heading); everything else is a novel.
COLLECTION_FILES = {
    "adventures.txt",
    "memoirs.txt",
    "return.txt",
    "his_last_bow.txt",
    "case_book.txt",
}


def main() -> None:
    if not FULL_CANON_DIR.exists():
        sys.exit(f"ERROR: {FULL_CANON_DIR} not found — run download_full_canon.py first")

    all_chunks: list[dict] = []
    skipped_stories: list[str] = []

    for filepath in sorted(FULL_CANON_DIR.glob("*.txt")):
        text = filepath.read_text(encoding="utf-8")
        slug = filepath.name

        if slug in COLLECTION_FILES:
            stories = _split_into_stories(text)
            if not stories:
                print(f"  WARNING: no story headings found in {slug}, chunking as novel")
                chunks = _chunk_text(text, slug.replace(".txt", ""), len(all_chunks))
                all_chunks.extend(chunks)
                continue

            for title, body in stories.items():
                if title in HELD_OUT_TITLES:
                    skipped_stories.append(title)
                    continue
                source_name = f"{slug.replace('.txt', '')}:{title.lower()[:40]}"
                chunks = _chunk_text(body, source_name, len(all_chunks))
                all_chunks.extend(chunks)
                print(f"  {slug}  story={title[:50]:<50}  chunks={len(chunks)}")
        else:
            # Novel — chunk as one continuous text
            source_name = slug.replace(".txt", "")
            chunks = _chunk_text(text, source_name, len(all_chunks))
            all_chunks.extend(chunks)
            print(f"  {slug}  (novel)  chunks={len(chunks)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks → {OUTPUT.relative_to(ROOT)}")
    if skipped_stories:
        print(f"Excluded (held-out): {skipped_stories}")


if __name__ == "__main__":
    main()
