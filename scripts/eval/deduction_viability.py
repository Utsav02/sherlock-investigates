#!/usr/bin/env python3
"""Viability probe for the SFT-on-reasoning-traces pivot.

QUESTION: can the BASE model, few-shot-prompted with Holmes-style deduction
exemplars, produce CONFIDENT DEDUCTIVE think blocks on held-out prompts? If yes,
self-distillation has clean source traces to SFT on (per the standing rule that
generated reasoning must come from the base model itself). If it still hedges,
we'd need a stronger generator (a new provenance decision).

Runs LOCALLY against Ollama (deepseek-r1:7b) — no GPU cost, fast iteration. For
each held-out deduction prompt it generates two think blocks:
  A (plain)          : the prompt alone — the baseline (we already know it hedges)
  B (Holmes few-shot): a deduction instruction + 2 exemplars, then the prompt

The exemplars are PROMPTING demonstrations that condition generation; they are
NOT training data. The traces you would later distill are the model's own B
outputs. No overlap between exemplars and the held-out prompts.

    python scripts/eval/deduction_viability.py --limit 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OLLAMA = "http://localhost:11434/api/chat"

# Deduction instruction for condition B.
SYSTEM = (
    "You observe people the way Sherlock Holmes does. Given a few observations "
    "about a stranger, reason from each specific detail to what it must imply, "
    "chain the clues together, and then commit to a single confident, specific "
    "conclusion about who the person is and their recent history. Be decisive: "
    "state what the evidence shows, do not list vague possibilities or hedge."
)

# Two exemplars in the target shape (observations -> stepwise reasoning ->
# confident conclusion), compactly derived from canonical Holmes deductions.
# Demonstrations for elicitation, not training data.
EXEMPLARS = [
    {"role": "user", "content":
        "Observations: a gentleman of medical air but military bearing; his face "
        "is darkened as by sun though his wrists are fair; he holds his left arm "
        "stiffly; his face is haggard, as from hardship and illness. What do you "
        "make of him?"},
    {"role": "assistant", "content":
        "<think>The medical air joined to a military bearing means an army "
        "doctor. The face is dark but the wrists are fair, so the tan was got "
        "abroad in a hot climate, not at home. The left arm is held stiffly — a "
        "recent injury. Hardship and illness together say a hard campaign in "
        "which he was both wounded and taken ill. An army doctor, lately abroad "
        "in the tropics, wounded and sick — that points to one place.</think>"
        "He is an army doctor, newly returned from a campaign in Afghanistan, "
        "where he was wounded and fell ill."},
    {"role": "user", "content":
        "Observations: a man whose right cuff is shiny and worn, with a callus "
        "on the side of his right little finger and a smear of ink along the "
        "edge of that hand. What do you make of him?"},
    {"role": "assistant", "content":
        "<think>The right cuff is worn shiny where a writer rests the forearm on "
        "the desk. The callus on the side of the little finger is where the hand "
        "drags across the page, and the ink on that same edge confirms it. This "
        "is not the mark of the occasional letter but of hours daily with a "
        "pen.</think>He earns his living by the pen — a clerk or copyist who "
        "writes for many hours every day."},
]


def ollama_chat(messages: list[dict], num_predict: int, seed: int) -> tuple[str, str]:
    """Return (think, answer). Handles Ollama's separate `thinking` field and
    inline <think> tags."""
    r = requests.post(OLLAMA, json={
        "model": "deepseek-r1:7b", "messages": messages, "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.6, "seed": seed},
    }, timeout=600)
    r.raise_for_status()
    msg = r.json().get("message", {})
    think = (msg.get("thinking") or "").strip()
    content = msg.get("content") or ""
    if not think and "<think>" in content:
        m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if m:
            think = m.group(1).strip()
            content = content[m.end():]
    return think, content.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompts", default="data/probes/probe_set_v1.jsonl")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/analysis")
    args = ap.parse_args()

    ppath = ROOT / args.prompts if not Path(args.prompts).is_absolute() else Path(args.prompts)
    prompts = [json.loads(l) for l in ppath.read_text().splitlines() if l.strip()]
    ded = [p for p in prompts if p["category"] == "DEDUCTION_INVITING"][:args.limit]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tpath = out_dir / f"deduction_viability_{stamp}.md"
    tf = tpath.open("w", encoding="utf-8")

    def emit(s: str) -> None:
        print(s, flush=True)
        tf.write(s + "\n")
        tf.flush()

    emit(f"# Deduction viability probe — base deepseek-r1:7b (Ollama)")
    emit(f"seed {args.seed}, num_predict {args.num_predict}, {len(ded)} held-out "
         f"deduction prompts\n")

    for p in ded:
        emit(f"## [{p['id']}] {p['prompt']}\n")
        emit("### A — plain (baseline)")
        a_think, a_ans = ollama_chat([{"role": "user", "content": p["prompt"]}],
                                     args.num_predict, args.seed)
        emit("**think:**\n```\n" + (a_think or "(no think block)") + "\n```")
        emit("**answer:** " + (a_ans[:400] or "(none)") + "\n")

        emit("### B — Holmes instruction + 2 exemplars")
        b_msgs = [{"role": "system", "content": SYSTEM}] + EXEMPLARS + \
                 [{"role": "user", "content": p["prompt"]}]
        b_think, b_ans = ollama_chat(b_msgs, args.num_predict, args.seed)
        emit("**think:**\n```\n" + (b_think or "(no think block)") + "\n```")
        emit("**answer:** " + (b_ans[:400] or "(none)") + "\n")
        emit("---\n")

    tf.close()
    print(f"\nwrote {tpath}", flush=True)


if __name__ == "__main__":
    main()
