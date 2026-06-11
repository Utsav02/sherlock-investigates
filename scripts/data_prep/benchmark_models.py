"""
Benchmark phi4-mini vs qwen2.5:7b on 20 randomly sampled chunks.

Reads  data/processed/chunks.jsonl
Writes results/pilot/local_model_comparison.json
Prints  a side-by-side table when done.

The sample is drawn with SEED=42 so results are reproducible.
The prompt version string is embedded in the saved JSON so that if the
prompt changes the saved file is clearly from a different run.
"""

import json
import random
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = ROOT / "data/processed/chunks.jsonl"
OUTPUT      = ROOT / "results/pilot/local_model_comparison.json"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELS     = ["phi4-mini", "qwen2.5:7b"]
SAMPLE_SIZE = 20
SEED        = 42

# Bump this string if you change the prompt wording; it travels with saved results.
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


def classify(model: str, content: str, timeout: int = 120) -> dict:
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
        "label":        label_m.group(1).lower() if label_m else "parse_error",
        "justification": just_m.group(1).strip() if just_m else raw,
        "raw_response": raw,
    }


def main() -> None:
    if not CHUNKS_FILE.exists():
        sys.exit(f"ERROR: {CHUNKS_FILE} not found — run chunk_stories.py first")

    chunks = [json.loads(l) for l in CHUNKS_FILE.read_text().splitlines() if l.strip()]
    rng = random.Random(SEED)
    sample = rng.sample(chunks, SAMPLE_SIZE)
    print(f"Sampled {SAMPLE_SIZE} chunks (seed={SEED}) from {len(chunks)} total\n")

    results = []
    for i, chunk in enumerate(sample, 1):
        row: dict = {
            "chunk_id":    chunk["chunk_id"],
            "source_story": chunk["source_story"],
            "word_count":  chunk["word_count"],
            "content":     chunk["content"],
        }
        for model in MODELS:
            print(f"  [{i:02d}/{SAMPLE_SIZE}] {model:15s} chunk {chunk['chunk_id']} ...", end=" ", flush=True)
            t0 = time.time()
            row[model] = classify(model, chunk["content"])
            elapsed = time.time() - t0
            print(f"{row[model]['label']:12s} ({elapsed:.1f}s)")
        results.append(row)

    output_data = {
        "metadata": {
            "sample_size":   SAMPLE_SIZE,
            "seed":          SEED,
            "prompt_version": PROMPT_VERSION,
            "models":        MODELS,
        },
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUTPUT.relative_to(ROOT)}\n")

    # Side-by-side table
    w = 18
    print(f"{'chunk_id':>8}  {'story':<22}  {'wc':>5}  "
          f"{'phi4-mini':<{w}}  {'qwen2.5:7b':<{w}}")
    print("-" * 85)
    for row in results:
        phi  = row["phi4-mini"]["label"]
        qwen = row["qwen2.5:7b"]["label"]
        agree = "" if phi == qwen else " <<<"
        print(f"{row['chunk_id']:>8}  {row['source_story']:<22}  {row['word_count']:>5}  "
              f"{phi:<{w}}  {qwen:<{w}}{agree}")

    # Agreement rate
    agreements = sum(1 for r in results if r["phi4-mini"]["label"] == r["qwen2.5:7b"]["label"])
    print(f"\nAgreement: {agreements}/{SAMPLE_SIZE} ({agreements/SAMPLE_SIZE*100:.0f}%)")

    print("\n=== Detailed justifications ===")
    for row in results:
        phi_r  = row["phi4-mini"]
        qwen_r = row["qwen2.5:7b"]
        if phi_r["label"] != qwen_r["label"]:
            marker = " [DISAGREE]"
        else:
            marker = ""
        print(f"\nchunk {row['chunk_id']} ({row['source_story']}, {row['word_count']} words){marker}")
        snippet = row["content"][:120].replace("\n", " ")
        print(f"  content: {snippet}...")
        print(f"  phi4-mini:  [{phi_r['label']:7s}] {phi_r['justification']}")
        print(f"  qwen2.5:7b: [{qwen_r['label']:7s}] {qwen_r['justification']}")


if __name__ == "__main__":
    main()
