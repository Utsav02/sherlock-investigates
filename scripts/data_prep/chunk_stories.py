"""
Split the three training stories into paragraph-level chunks and write
data/processed/chunks.jsonl.

Each line of the JSONL has:
  chunk_id       int   — monotonic across all stories
  source_story   str   — "study_in_scarlet" | "scandal_in_bohemia" | "red_headed_league"
  paragraph_index int  — raw paragraph index within the story file (0-based,
                         counting all paragraphs including filtered ones so this
                         is a stable positional reference back to the source file)
  content        str   — the paragraph text
  word_count     int   — number of whitespace-separated tokens

Paragraphs shorter than MIN_WORDS are dropped (section headings, short
dialogue fragments). The threshold is conservative so single-sentence
deductions like "You have been in Afghanistan, I perceive." survive.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAINING_FILES = [
    (ROOT / "data/processed/study_in_scarlet.txt",           "study_in_scarlet"),
    (ROOT / "data/processed/training/scandal_in_bohemia.txt", "scandal_in_bohemia"),
    (ROOT / "data/processed/training/red_headed_league.txt",  "red_headed_league"),
]

OUTPUT = ROOT / "data/processed/chunks.jsonl"

MIN_WORDS = 10


def split_paragraphs(text: str) -> list[str]:
    """Split text on one or more consecutive blank lines."""
    return re.split(r"\n{2,}", text)


def main() -> None:
    chunks: list[dict] = []
    chunk_id = 0

    for filepath, story_name in TRAINING_FILES:
        if not filepath.exists():
            sys.exit(f"ERROR: training file not found: {filepath}")

        text = filepath.read_text(encoding="utf-8")
        raw_paragraphs = split_paragraphs(text)

        for para_idx, raw in enumerate(raw_paragraphs):
            content = raw.strip()
            if not content:
                continue
            word_count = len(content.split())
            if word_count < MIN_WORDS:
                continue

            chunks.append({
                "chunk_id":       chunk_id,
                "source_story":   story_name,
                "paragraph_index": para_idx,
                "content":        content,
                "word_count":     word_count,
            })
            chunk_id += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(chunks)} chunks to {OUTPUT.relative_to(ROOT)}\n")
    print("=== First 3 chunks ===")
    for chunk in chunks[:3]:
        print(f"\nchunk_id={chunk['chunk_id']}  source={chunk['source_story']}  "
              f"para_idx={chunk['paragraph_index']}  words={chunk['word_count']}")
        print(chunk["content"][:200] + ("..." if len(chunk["content"]) > 200 else ""))

    print(f"\n=== Total chunks: {len(chunks)} ===")


if __name__ == "__main__":
    main()
