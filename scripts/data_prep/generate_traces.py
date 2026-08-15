#!/usr/bin/env python3
"""Trace generator + ground-truth filter — step 2 of the SFT pivot.

For each reverse-constructed scenario, few-shot-prompt the BASE model to deduce,
capture its <think> trace + answer, and KEEP the trace only if its answer reaches
the ground-truth identity. This is self-distillation (traces are the base model's
own reasoning — provenance-preserving, per the standing rule) with the automatic
rejection-sampling filter that reverse construction made possible.

The scenario is shown WITH few-shot exemplars to elicit deductive reasoning, but
the SFT example trains on the PLAIN scenario -> <think>trace</think> answer (the
exemplars are stripped), so the model learns to deduce zero-shot.

Runs locally against Ollama (deepseek-r1:7b). Multiple samples per scenario
(--samples) give the filter candidates to choose from.

    python scripts/data_prep/generate_traces.py \
        --scenarios data/sft/scenarios_seed_claude.jsonl --samples 2

Output: one row per (scenario, sample) with `matched` (keeper) + the think/answer.
Downstream: matched==True rows, chat-formatted, become the SFT set (+ OpenThoughts
format anchor).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data_prep"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from reverse_scenarios import detect_leak          # word-match; here a MATCH is GOOD
from deduction_viability import EXEMPLARS, ollama_chat  # base-model few-shot + extractor

# A hard "commit or nothing" instruction — the viability probe showed the base
# model defaults to hedged enumeration; this pushes it toward a single decisive
# conclusion. Kept separate from the (softer) viability SYSTEM on purpose.
TRACE_SYSTEM = (
    "You are Sherlock Holmes. You are shown a stranger and a few observed "
    "details. In your private reasoning, take each detail in turn and state what "
    "it implies, then combine them into ONE confident, specific conclusion about "
    "who this person is — their trade, station, or recent history. COMMIT. Do not "
    "list alternatives, do not hedge, never write 'maybe', 'could be', or 'might'. "
    "Reason concisely, then give the answer as a single decisive sentence "
    "beginning 'This is'."
)


def answer_matches(ground_truth: str, answer: str) -> tuple[bool, list[str]]:
    """A keeper's answer must name the identity. Reuses the leak-detector's
    word-match (a ground-truth content word / variant appearing in the text) —
    the SAME computation, opposite intent: a hit in a scenario is a leak (bad);
    a hit in an answer is a correct deduction (good)."""
    return detect_leak(ground_truth, [], answer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenarios", default="data/sft/scenarios_seed_claude.jsonl")
    ap.add_argument("--samples", type=int, default=2, help="generations per scenario")
    ap.add_argument("--limit", type=int, default=None, help="first N scenarios")
    ap.add_argument("--num-predict", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spath = ROOT / args.scenarios if not Path(args.scenarios).is_absolute() else Path(args.scenarios)
    scenarios = [json.loads(l) for l in spath.read_text().splitlines() if l.strip()]
    scenarios = [s for s in scenarios if s.get("usable", True)]
    if args.limit:
        scenarios = scenarios[:args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "data" / "sft" / f"traces_{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w", encoding="utf-8")

    total = keepers = 0
    scen_with_keeper = 0
    for s in scenarios:
        gt, prompt = s["ground_truth"], s["scenario_prompt"]
        hit_this = False
        for k in range(args.samples):
            seed = args.seed + k
            think, answer = ollama_chat(
                [{"role": "system", "content": TRACE_SYSTEM}] + EXEMPLARS +
                [{"role": "user", "content": prompt}],
                args.num_predict, seed)
            matched, terms = answer_matches(gt, answer)
            has_think = bool(think and think.strip())
            keep = matched and has_think
            row = {"scenario_id": s.get("id"), "ground_truth": gt,
                   "scenario_prompt": prompt, "sample": k, "seed": seed,
                   "think": think, "answer": answer,
                   "has_think": has_think, "matched": matched,
                   "matched_terms": terms, "keeper": keep,
                   "generator": "deepseek-r1:7b (self-distilled)"}
            f.write(json.dumps(row) + "\n")
            f.flush()
            total += 1
            keepers += keep
            hit_this = hit_this or keep
            tag = "KEEP" if keep else ("no-match" if has_think else "no-think")
            print(f"  [{s.get('id'):>2}.{k}] {tag:<9} gt={gt[:34]:<34} "
                  f"ans={answer[:44].replace(chr(10),' ')}", flush=True)
        scen_with_keeper += hit_this
    f.close()
    print(f"\n{keepers}/{total} traces kept | "
          f"{scen_with_keeper}/{len(scenarios)} scenarios have >=1 keeper -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
