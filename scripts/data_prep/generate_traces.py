#!/usr/bin/env python3
"""Trace generator + ground-truth filter — step 2 of the SFT pivot.

For each reverse-constructed scenario, prompt a TEACHER to deduce, capture its
<think> trace + answer, and KEEP the trace only if its answer reaches the
ground-truth identity — the automatic rejection-sampling filter that reverse
construction made possible.

Two backends:

  --backend ollama   base deepseek-r1:7b, i.e. SELF-distillation. Measured
                     2026-08-15: **0/10 keepers** — the base commits ("This
                     is...") to confident-WRONG generic answers (clerk, office
                     worker) whatever the cues. It cannot bootstrap a skill it
                     lacks. Retained as the baseline arm, not a data source.

  --backend claude   the Claude Code CLI as a stronger teacher — STaR /
                     rejection-sampling distillation, the mainstream recipe our
                     own base (R1-Distill) was made with (Decision Log
                     2026-08-15). Claude does NOT emit <think> natively, so the
                     R1 output format is specified and demonstrated explicitly;
                     see `_CLAUDE_FORMAT` and `split_think`.

The scenario is shown WITH few-shot exemplars to elicit deductive reasoning, but
the SFT example trains on the PLAIN scenario -> <think>trace</think> answer (the
exemplars are stripped), so the model learns to deduce zero-shot.

`--samples` gives the filter multiple candidates per scenario to choose from.

    python scripts/data_prep/generate_traces.py --backend claude \
        --scenarios data/sft/scenarios_seed_claude.jsonl --samples 1

CAVEAT the filter cannot see (2026-08-15): `matched` checks the ANSWER only. A
trace can reach the right answer through hollow or templated reasoning, and it is
the REASONING that gets SFT'd into the student. The think blocks must be READ on
a sample of every batch before it is used as training data.

Output: one row per (scenario, sample) with `matched` (keeper) + the think/answer.
Downstream: matched==True rows, chat-formatted, become the SFT set (+ OpenThoughts
format anchor).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data_prep"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from reverse_scenarios import detect_leak, claude_cli  # word-match; here a MATCH is GOOD
from deduction_viability import EXEMPLARS, ollama_chat  # base-model few-shot + extractor

# Provenance stamped on every row — which model actually authored the reasoning.
GENERATORS = {
    "ollama": "deepseek-r1:7b (self-distilled)",
    "claude": "claude-code-cli (STaR teacher; stronger-model distillation)",
}

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


# --- Claude (teacher) backend -------------------------------------------------
# Claude does NOT emit R1-style <think> blocks by default, so the format is
# specified explicitly and demonstrated. The output has to land in the STUDENT's
# format (R1: <think>reasoning</think> then the answer), because these traces are
# SFT targets for deepseek-r1-7b — a trace in any other shape is unusable.
_CLAUDE_FORMAT = """
OUTPUT FORMAT — follow it exactly, with nothing before or after:
<think>
Step-by-step deduction: take each observed cue in turn, state what it implies, \
and chain the implications into ONE confident conclusion.
</think>
This is <the single decisive conclusion>.

The reply must begin with the literal characters <think> and end with the \
sentence beginning "This is". Write no preamble, no headings, no commentary, no \
markdown code fences. Reason inside the <think> block only; the line after it is \
the spoken answer.
"""


def _render_exemplars() -> str:
    """The same EXEMPLARS the Ollama backend passes as chat turns, flattened to
    text for the single-prompt CLI — one source of truth for the demonstrations,
    so the two backends stay in the same target shape."""
    out = []
    for i in range(0, len(EXEMPLARS), 2):
        user, asst = EXEMPLARS[i]["content"], EXEMPLARS[i + 1]["content"]
        body = asst.replace("</think>", "</think>\n")  # answer on its own line
        out.append(f"EXAMPLE {i // 2 + 1}\n{user}\n{body}")
    return "\n\n".join(out)


def build_claude_prompt(scenario: str) -> str:
    return (f"{TRACE_SYSTEM}\n{_CLAUDE_FORMAT}\n"
            f"Here are two worked examples in exactly the required shape.\n\n"
            f"{_render_exemplars()}\n\nNow the real case.\n\n{scenario}")


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def split_think(raw: str) -> tuple[str, str]:
    """Split a raw completion into (think, answer).

    Tolerates the two ways the format degrades in practice: a stray markdown
    fence around the whole reply, and a block that opens <think> but never closes
    (the same unclosed-block failure the dose curve found in the fine-tuned
    student). With no closing tag the think field is empty, so the row is scored
    has_think=False and can never become a keeper; the unparsed text is left in
    `answer` for diagnosis rather than discarded.
    """
    text = re.sub(r"^\s*```[a-z]*\s*|\s*```\s*$", "", raw.strip())
    m = _THINK_RE.search(text)
    if not m:
        return "", text.strip()
    think = m.group(1).strip()
    rest = text[m.end():].strip()
    # Prefer the decisive sentence if the model added trailing chatter.
    for line in rest.splitlines():
        if line.strip().lower().startswith("this is"):
            return think, line.strip()
    return think, rest


def claude_trace(scenario: str, timeout: int = 300) -> tuple[str, str, str]:
    """Return (think, answer, raw) for one scenario from the Claude teacher."""
    raw = claude_cli(build_claude_prompt(scenario), timeout=timeout)
    think, answer = split_think(raw)
    return think, answer, raw


def answer_matches(ground_truth: str, answer: str) -> tuple[bool, list[str]]:
    """A keeper's answer must name the identity. Reuses the leak-detector's
    word-match (a ground-truth content word / variant appearing in the text) —
    the SAME computation, opposite intent: a hit in a scenario is a leak (bad);
    a hit in an answer is a correct deduction (good)."""
    return detect_leak(ground_truth, [], answer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenarios", default="data/sft/scenarios_seed_claude.jsonl")
    ap.add_argument("--backend", choices=("ollama", "claude"), default="ollama",
                    help="ollama = base deepseek-r1:7b self-distillation "
                         "(measured 0/10 keepers — it cannot deduce). "
                         "claude = the Claude Code CLI teacher, STaR-style "
                         "distillation from a stronger model (2026-08-15).")
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
    out = (Path(args.out) if args.out else
           ROOT / "data" / "sft" / f"traces_{args.backend}_{stamp}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w", encoding="utf-8")

    total = keepers = 0
    scen_with_keeper = 0
    for s in scenarios:
        gt, prompt = s["ground_truth"], s["scenario_prompt"]
        hit_this = False
        for k in range(args.samples):
            seed = args.seed + k
            raw = None
            if args.backend == "claude":
                # No seed parameter on the CLI — sampling is not reproducible
                # here, so `seed` is recorded as null rather than a fake value.
                think, answer, raw = claude_trace(prompt)
                seed = None
            else:
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
                   "generator": GENERATORS[args.backend]}
            if raw is not None:
                row["raw"] = raw
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
