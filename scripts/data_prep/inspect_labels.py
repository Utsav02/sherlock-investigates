"""
Spot-check labelled chunks by category.

Usage:
    python3 scripts/data_prep/inspect_labels.py               # all three labels, 5 each
    python3 scripts/data_prep/inspect_labels.py --label central
    python3 scripts/data_prep/inspect_labels.py --label central --n 10
    python3 scripts/data_prep/inspect_labels.py --seed 7      # different random sample

Reads data/processed/chunks_labeled.jsonl
"""

import argparse
import json
import random
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
LABELS_FILE = ROOT / "data/processed/chunks_labeled.jsonl"

VALID_LABELS = ("none", "minor", "central")
SEPARATOR = "─" * 72


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def print_chunk(chunk: dict, rank: int, total: int) -> None:
    label  = chunk["label"]
    story  = chunk["source_story"]
    wc     = chunk["word_count"]
    cid    = chunk["chunk_id"]
    just   = chunk.get("justification", "—")

    print(f"\n[{rank}/{total}]  chunk_id={cid}  label={label}  story={story}  words={wc}")
    print(f"  justification: {just}")
    print()
    # Wrap content at 80 chars with 2-space indent
    for line in chunk["content"].splitlines():
        if line.strip():
            print(textwrap.fill(line, width=80, initial_indent="  ", subsequent_indent="  "))
        else:
            print()
    print(SEPARATOR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect labeled chunks by category.")
    parser.add_argument("--label",  choices=[*VALID_LABELS, "all"], default="all",
                        help="Which label to inspect (default: all three)")
    parser.add_argument("--n",      type=int, default=5,
                        help="Number of chunks to sample per label (default: 5)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    if not LABELS_FILE.exists():
        print(f"ERROR: {LABELS_FILE} not found — run classify_chunks.py first")
        return

    chunks  = load_chunks(LABELS_FILE)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_label[chunk["label"]].append(chunk)

    # Summary
    total = len(chunks)
    print(f"=== chunks_labeled.jsonl: {total} chunks ===")
    model = chunks[0].get("model", "unknown") if chunks else "unknown"
    print(f"    model: {model}\n")
    for lbl in (*VALID_LABELS, "error", "parse_error"):
        count = len(by_label.get(lbl, []))
        pct   = count / total * 100 if total else 0
        bar   = "#" * int(pct / 2)
        print(f"  {lbl:12s}: {count:4d}  ({pct:5.1f}%)  {bar}")
    print()

    target_labels = VALID_LABELS if args.label == "all" else (args.label,)
    rng = random.Random(args.seed)

    for lbl in target_labels:
        pool = by_label.get(lbl, [])
        if not pool:
            print(f"\n=== {lbl.upper()} — no chunks with this label ===")
            continue

        n = min(args.n, len(pool))
        sample = rng.sample(pool, n)
        print(f"\n{'='*72}")
        print(f"  Label: {lbl.upper()}  — showing {n} of {len(pool)} chunks  (seed={args.seed})")
        print(f"{'='*72}")
        for rank, chunk in enumerate(sample, 1):
            print_chunk(chunk, rank, n)


if __name__ == "__main__":
    main()
