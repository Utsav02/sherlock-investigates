"""
Full classification pass over all chunks using a chosen Ollama model.

Usage:
    python3 scripts/data_prep/classify_chunks.py --model phi4-mini
    python3 scripts/data_prep/classify_chunks.py --model qwen2.5:7b

Reads  data/processed/chunks.jsonl
Writes data/processed/chunks_labeled.jsonl
Cache  data/processed/.cache/<sha256>.json  (keyed on model + content + prompt version)

Re-running the same command with the same prompt version is free: all chunks
that were already classified are served from cache. Change PROMPT_VERSION to
invalidate the cache and redo the classification.

Each line of chunks_labeled.jsonl extends the input chunk fields with:
    label          str  — "none" | "minor" | "central" | "error"
    justification  str  — one-sentence rationale from the model
    model          str  — model name used for classification
    prompt_version str  — version tag for the prompt
    cached         bool — True if the result came from the on-disk cache
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from tqdm import tqdm

ROOT        = Path(__file__).resolve().parents[2]
CHUNKS_FILE = ROOT / "data/processed/chunks.jsonl"
OUTPUT      = ROOT / "data/processed/chunks_labeled.jsonl"
CACHE_DIR   = ROOT / "data/processed/.cache"

OLLAMA_URL = "http://localhost:11434/api/chat"

# Bump if prompt wording changes — old cache entries use the old version string
# in their key so the new prompt will get fresh calls.
PROMPT_VERSION = "v1"

SYSTEM_MSG = (
    "You classify passages from Sherlock Holmes stories. "
    "Respond ONLY with a LABEL line and a JUSTIFICATION line. No other text."
)


def user_msg(content: str) -> str:
    return (
        f"Passage:\n{content}\n\n"
        "Does this passage contain Holmes's deductive reasoning — "
        "his process of observing details and inferring conclusions from them?\n\n"
        "Labels:\n"
        "  none    — No deductive reasoning. Pure narrative, action, atmosphere, "
        "or dialogue without deduction.\n"
        "  minor   — Deduction is mentioned or briefly referenced but is not the "
        "passage's focus.\n"
        "  central — Holmes's deductive process is the primary focus: he explains "
        "observations and inferences, or actively demonstrates deduction.\n\n"
        "Respond in exactly this format (nothing else):\n"
        "LABEL: <none|minor|central>\n"
        "JUSTIFICATION: <one sentence explaining your choice>"
    )


LABEL_RE         = re.compile(r"LABEL\s*:\s*(none|minor|central)", re.IGNORECASE)
JUSTIFICATION_RE = re.compile(r"JUSTIFICATION\s*:\s*(.+)", re.IGNORECASE)


def cache_key(model: str, content: str) -> str:
    raw = f"{model}|||{PROMPT_VERSION}|||{content}"
    return hashlib.sha256(raw.encode()).hexdigest()


def load_from_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def save_to_cache(key: str, result: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(result, ensure_ascii=False))


def call_ollama(model: str, content: str, timeout: int = 120) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": user_msg(content)},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
    except Exception as e:
        return {"label": "error", "justification": str(e), "raw_response": ""}

    label_m = LABEL_RE.search(raw)
    just_m  = JUSTIFICATION_RE.search(raw)
    return {
        "label":         label_m.group(1).lower() if label_m else "parse_error",
        "justification": just_m.group(1).strip() if just_m else raw,
        "raw_response":  raw,
    }


def classify_chunk(model: str, chunk: dict) -> tuple[dict, bool]:
    """Return (classification_result, was_cached)."""
    key = cache_key(model, chunk["content"])
    cached = load_from_cache(key)
    if cached is not None:
        return cached, True

    result = call_ollama(model, chunk["content"])
    save_to_cache(key, result)
    return result, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify all chunks with a local Ollama model.")
    parser.add_argument("--model", required=True, choices=["phi4-mini", "qwen2.5:7b"],
                        help="Ollama model to use")
    parser.add_argument("--input",  default=str(CHUNKS_FILE),
                        help="Input chunks JSONL (default: data/processed/chunks.jsonl)")
    parser.add_argument("--output", default=str(OUTPUT),
                        help="Output labeled JSONL (default: data/processed/chunks_labeled.jsonl)")
    args = parser.parse_args()
    model = args.model

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(f"ERROR: {input_path} not found")

    chunks = [json.loads(l) for l in input_path.read_text().splitlines() if l.strip()]
    total  = len(chunks)
    print(f"Classifying {total} chunks with {model}  (prompt {PROMPT_VERSION})\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts: Counter = Counter()
    cached_count = 0

    pbar = tqdm(
        chunks,
        unit="chunk",
        desc=f"classify [{model}]",
        dynamic_ncols=True,
    )

    with output_path.open("w", encoding="utf-8") as out_f:
        for i, chunk in enumerate(pbar, 1):
            result, was_cached = classify_chunk(model, chunk)
            label = result["label"]
            label_counts[label] += 1
            if was_cached:
                cached_count += 1

            labeled = {
                **chunk,
                "label":          label,
                "justification":  result.get("justification", ""),
                "model":          model,
                "prompt_version": PROMPT_VERSION,
                "cached":         was_cached,
            }
            out_f.write(json.dumps(labeled, ensure_ascii=False) + "\n")

            pbar.set_postfix({
                "none":    label_counts.get("none",    0),
                "minor":   label_counts.get("minor",   0),
                "central": label_counts.get("central", 0),
                "cached":  cached_count,
            })

            # Periodic stdout checkpoint every 50 chunks (captured in log file)
            if i % 50 == 0 or i == total:
                tqdm.write(
                    f"  checkpoint {i:4d}/{total}  "
                    f"none={label_counts.get('none',0)}  "
                    f"minor={label_counts.get('minor',0)}  "
                    f"central={label_counts.get('central',0)}  "
                    f"cached={cached_count}"
                )

    print(f"\nWrote {total} labeled chunks to {output_path}")
    print(f"Cache hits: {cached_count}/{total}\n")

    print("=== Label distribution ===")
    for label in ("none", "minor", "central", "error", "parse_error"):
        count = label_counts.get(label, 0)
        pct   = count / total * 100
        bar   = "#" * int(pct / 2)
        print(f"  {label:12s}: {count:4d}  ({pct:5.1f}%)  {bar}")

    by_story: dict[str, Counter] = {}
    for chunk in chunks:
        story = chunk["source_story"]
        if story not in by_story:
            by_story[story] = Counter()

    for line in output_path.read_text().splitlines():
        row = json.loads(line)
        by_story[row["source_story"]][row["label"]] += 1

    print("\n=== Distribution by story ===")
    for story, counts in by_story.items():
        total_s = sum(counts.values())
        central = counts.get("central", 0)
        minor   = counts.get("minor", 0)
        print(f"  {story:<26}: {total_s:4d} chunks  "
              f"central={central}  minor={minor}  none={counts.get('none', 0)}")


if __name__ == "__main__":
    main()
