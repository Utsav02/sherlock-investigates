"""
Augment the labeled chunk corpus into training examples.

Framing matrix (from data/augmented/augmentation_spec.md):
  VERBATIM → central + minor   (no generation; emit chunk as-is)
  WATSON   → central + minor   (word_count >= 30)
  QA       → central only
  CHAIN    → central only
  REVERSE  → central only      (word_count >= 100)

The design doc calls for 3× oversampling of central chunks relative to minor.
That weighting is left to the training script — this script emits each example
once and records the source label so the training script can apply weights.

Usage:
    python3 scripts/data_prep/augment_corpus.py
    python3 scripts/data_prep/augment_corpus.py --model qwen2.5:7b --temperature 0.7

Reads  data/processed/chunks_labeled.jsonl
Writes data/augmented/train.jsonl
       data/augmented/manifest.json
Cache  data/augmented/.cache/<sha256>.txt  (keyed on model + framing + content)

Re-running is cheap: every generated example is cached on disk. Change
AUGMENT_VERSION to invalidate cache and regenerate from scratch.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

ROOT      = Path(__file__).resolve().parents[2]
INPUT     = ROOT / "data/processed/chunks_labeled.jsonl"
OUTPUT    = ROOT / "data/augmented/train.jsonl"
MANIFEST  = ROOT / "data/augmented/manifest.json"
CACHE_DIR = ROOT / "data/augmented/.cache"

OLLAMA_URL = "http://localhost:11434/api/chat"

AUGMENT_VERSION = "v1"

MIN_WORDS_WATSON  = 30
MIN_WORDS_REVERSE = 100

# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

_QA_SYSTEM = (
    "You write companion documents for Victorian detective fiction. "
    "Given a story passage, produce a first-person answer in the detective's voice "
    "explaining exactly how he reasoned from his observations to his conclusion. "
    "Be specific about each observation and the inference it supports. "
    "Maintain Victorian register and formal sentence structure. "
    "Respond with only the answer text — no preamble, no meta-commentary."
)

_WATSON_SYSTEM = (
    "You write from the perspective of Dr. John Watson. "
    "Given a scene from a Victorian detective story, write a retrospective account "
    "in Watson's voice: past tense, 3–5 sentences. "
    "Emphasise what Holmes noticed, what Watson initially missed, and what the "
    "experience revealed about Holmes's method. "
    "Respond with only Watson's account — no preamble, no meta-commentary."
)

_CHAIN_SYSTEM = (
    "You produce structured analyses of Victorian detective reasoning. "
    "Given a passage, extract the observable facts and the inferences drawn from "
    "them, then state the final conclusion. "
    "Respond with numbered OBSERVATIONS, numbered INFERENCES, and a CONCLUSION line. "
    "Start your response directly with '1.' for the first observation — no preamble."
)

_REVERSE_SYSTEM = (
    "You restate Victorian detective conclusions. "
    "Given a passage, identify the detective's main conclusion and state it as a "
    "single direct sentence prefixed exactly 'CONCLUSION: '. "
    "Then on the next line write exactly this question: "
    "'What observations would you expect a skilled detective to have made in order "
    "to reach this conclusion?' "
    "Nothing else — no explanation, no preamble, no passage reproduction."
)


def _qa_user(content: str) -> str:
    return (
        f"PASSAGE:\n{content}\n\n"
        "Q: Given the observations described above, what does the detective conclude, "
        "and how does he arrive at that conclusion?\n\n"
        "Write only the answer — begin directly with the detective's reasoning:"
    )


def _watson_user(content: str) -> str:
    return (
        f"PASSAGE:\n{content}\n\n"
        "Write Watson's retrospective account of this scene:"
    )


def _chain_user(content: str) -> str:
    return (
        f"PASSAGE:\n{content}\n\n"
        "Extract the reasoning chain. Fill in the structured analysis, "
        "beginning with the first observable fact:\n"
        "1."
    )


def _reverse_user(content: str) -> str:
    return (
        f"PASSAGE:\n{content}\n\n"
        "State the detective's main conclusion, then write the question:"
    )


def _qa_assemble(model_output: str, content: str) -> str:
    return (
        "Below is a passage from a Victorian detective story, followed by "
        "a question and answer based on it.\n\n"
        f"PASSAGE:\n{content}\n\n"
        "Q: Given the observations described above, what does the detective "
        "conclude, and how does he arrive at that conclusion?\n\n"
        f"A: {model_output.strip()}"
    )


def _watson_assemble(model_output: str, content: str) -> str:
    return (
        "Below is a passage from a Victorian detective story, followed by "
        "Watson's retrospective account of the same scene.\n\n"
        f"PASSAGE:\n{content}\n\n"
        f"WATSON'S ACCOUNT:\n{model_output.strip()}"
    )


def _chain_assemble(model_output: str, content: str) -> str:
    return (
        "Below is a passage from a Victorian detective story. "
        "Read it carefully, then complete the structured analysis.\n\n"
        f"PASSAGE:\n{content}\n\n"
        "OBSERVATIONS (what can be directly seen or measured):\n"
        f"1.{model_output.strip()}"
    )


def _reverse_assemble(model_output: str, content: str) -> str:
    return f"{model_output.strip()}\n\nORIGINAL PASSAGE:\n{content}"


# (system_msg, user_msg_fn, assemble_fn)
FRAMINGS = {
    "QA":      (_QA_SYSTEM,      _qa_user,      _qa_assemble),
    "WATSON":  (_WATSON_SYSTEM,  _watson_user,  _watson_assemble),
    "CHAIN":   (_CHAIN_SYSTEM,   _chain_user,   _chain_assemble),
    "REVERSE": (_REVERSE_SYSTEM, _reverse_user, _reverse_assemble),
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(model: str, framing: str, content: str) -> str:
    raw = f"{model}|||{AUGMENT_VERSION}|||{framing}|||{content}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.txt"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


def _save_cache(key: str, text: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Ollama call + generation
# ---------------------------------------------------------------------------

def _call_ollama(model: str, system: str, user: str,
                 temperature: float, timeout: int = 120) -> str | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception:
        return None


def generate_framing(model: str, framing: str, content: str,
                     temperature: float) -> tuple[str | None, bool]:
    """Return (training_text, was_cached). Returns (None, False) on error."""
    system_msg, user_fn, assemble_fn = FRAMINGS[framing]
    key    = _cache_key(model, framing, content)
    cached = _load_cache(key)
    if cached is not None:
        return cached, True

    raw = _call_ollama(model, system_msg, user_fn(content), temperature)
    if raw is None:
        return None, False

    text = assemble_fn(raw, content)
    _save_cache(key, text)
    return text, False


# ---------------------------------------------------------------------------
# Framing selection per chunk
# ---------------------------------------------------------------------------

def framings_for(chunk: dict) -> list[str]:
    label = chunk["label"]
    words = chunk["word_count"]
    result = ["VERBATIM"]

    if words >= MIN_WORDS_WATSON:
        result.append("WATSON")

    if label == "central":
        result.append("QA")
        result.append("CHAIN")
        if words >= MIN_WORDS_REVERSE:
            result.append("REVERSE")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment labeled Sherlock corpus into CLM training examples."
    )
    parser.add_argument(
        "--model", default="qwen2.5:7b",
        help="Ollama model for generation (default: qwen2.5:7b)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--oversample-central", type=int, default=1, dest="oversample_central",
        help="Repeat each central chunk's examples N times (default: 1). "
             "Use 3 to match the design doc's 3× central oversample target.",
    )
    args = parser.parse_args()

    if not INPUT.exists():
        sys.exit(f"ERROR: {INPUT} not found — run classify_chunks.py first")

    chunks   = [json.loads(l) for l in INPUT.read_text().splitlines() if l.strip()]
    eligible = [c for c in chunks if c["label"] in ("central", "minor")]
    n_central = sum(1 for c in eligible if c["label"] == "central")
    n_minor   = len(eligible) - n_central

    print(
        f"Loaded {len(chunks)} chunks, {len(eligible)} eligible "
        f"(central={n_central}, minor={n_minor})\n"
    )

    tasks = [
        (chunk, framing)
        for chunk in eligible
        for framing in framings_for(chunk)
        for _ in range(args.oversample_central if chunk["label"] == "central" else 1)
    ]
    n_verbatim  = sum(1 for _, f in tasks if f == "VERBATIM")
    n_generated = len(tasks) - n_verbatim
    oversample_note = (
        f"  (central ×{args.oversample_central})" if args.oversample_central > 1 else ""
    )
    print(
        f"Examples to produce: {len(tasks)} total  "
        f"({n_verbatim} VERBATIM, {n_generated} generated){oversample_note}\n"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    by_label_framing: dict[str, Counter] = defaultdict(Counter)
    error_count  = 0
    cached_count = 0
    total_words  = 0

    pbar = tqdm(tasks, unit="ex", desc="augment", dynamic_ncols=True)

    with OUTPUT.open("w", encoding="utf-8") as out_f:
        for chunk, framing in pbar:
            content = chunk["content"]

            if framing == "VERBATIM":
                training_text = content
                was_cached    = False
            else:
                training_text, was_cached = generate_framing(
                    args.model, framing, content, args.temperature
                )

            if training_text is None:
                error_count += 1
                tqdm.write(
                    f"  ERROR  chunk_id={chunk['chunk_id']}  framing={framing}"
                )
                continue

            if was_cached:
                cached_count += 1

            wc = len(training_text.split())
            total_words += wc
            by_label_framing[chunk["label"]][framing] += 1

            record = {
                "chunk_id":       chunk["chunk_id"],
                "source_story":   chunk["source_story"],
                "original_label": chunk["label"],
                "framing":        framing,
                "word_count":     wc,
                "text":           training_text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total_written = sum(sum(v.values()) for v in by_label_framing.values())
            pbar.set_postfix({
                "written":  total_written,
                "errors":   error_count,
                "cached":   cached_count,
                "~tokens":  f"{int(total_words * 1.33 / 1000)}K",
            })

    total_written = sum(sum(v.values()) for v in by_label_framing.values())
    approx_tokens = int(total_words * 1.33)

    print(f"\nWrote {total_written} examples → {OUTPUT.relative_to(ROOT)}")
    print(f"Errors: {error_count}   Cache hits: {cached_count}\n")

    print("=== Examples by label × framing ===")
    for label in ("central", "minor"):
        fc = by_label_framing.get(label, Counter())
        print(
            f"  {label:<8}: {sum(fc.values()):4d} total   "
            + "  ".join(
                f"{f}={fc.get(f, 0)}"
                for f in ("VERBATIM", "QA", "WATSON", "CHAIN", "REVERSE")
            )
        )

    target_low, target_high = 240_000, 400_000
    print(f"\nEstimated tokens: {approx_tokens:,}")
    if approx_tokens < target_low:
        shortfall = target_low - approx_tokens
        print(
            f"  WARNING: {shortfall:,} tokens below 240K lower bound. "
            "Consider a 6th framing or adding curated narrative 'none' chunks."
        )
    elif approx_tokens > target_high:
        print(
            f"  NOTE: above 400K upper bound — consider dropping minor WATSON "
            "if training budget is tight."
        )
    else:
        print("  Within the 240K–400K target range.")

    manifest = {
        "augment_version":   AUGMENT_VERSION,
        "model":             args.model,
        "temperature":       args.temperature,
        "oversample_central": args.oversample_central,
        "input_file":      str(INPUT.relative_to(ROOT)),
        "output_file":     str(OUTPUT.relative_to(ROOT)),
        "total_examples":  total_written,
        "approx_tokens":   approx_tokens,
        "error_count":     error_count,
        "by_label_framing": {k: dict(v) for k, v in by_label_framing.items()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest → {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
