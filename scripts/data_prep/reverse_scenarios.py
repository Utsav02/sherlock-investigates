#!/usr/bin/env python3
"""Reverse-construction scenario generator — step 1 of the SFT pivot.

Starts from a KNOWN identity/occupation/situation and asks the model to invent
concrete observable CUES that imply it (without naming it), then phrases a
forensic scenario. This gives what raw prompts could not (viability probe,
2026-08-14):
  - forensic flavor (cues -> hidden identity), the right reasoning type;
  - a CRISP answer by construction, so confident deduction is appropriate (not
    over-reach) — the ambiguity that made the base model hedge is removed;
  - a GROUND TRUTH to filter generated traces against downstream (a trace is a
    keeper if its conclusion matches the seed identity) — automatic rejection
    sampling.

Fully self-sourced: our seed identities + our model's cues. No external dataset,
no license question (see docs/data_strategy.md — ART/ROCStories dropped from the
critical path for exactly this reason).

Runs LOCALLY against Ollama (deepseek-r1:7b, same family as the training base).

    python scripts/data_prep/reverse_scenarios.py --limit 5
    python scripts/data_prep/reverse_scenarios.py            # all seeds

Downstream: generate a Holmes-style deductive <think> trace for each
`scenario_prompt`, keep only traces whose conclusion matches `ground_truth`,
then SFT on the survivors (+ an OpenThoughts format anchor).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OLLAMA = "http://localhost:11434/api/chat"

# Seed identities — deliberately diverse (occupation, condition, situation).
# Public, trivially self-authored; expand freely. These are the ground truths.
SEED_IDENTITIES = [
    "a retired sergeant of the Royal Marines",
    "a night-shift hospital nurse coming off a long shift",
    "a long-haul lorry driver",
    "a professional concert violinist",
    "a practised pickpocket",
    "a deep-sea trawler fisherman",
    "a new mother of a very young infant, badly sleep-deprived",
    "a medical student in the week before final exams",
    "a bookbinder",
    "a watchmaker",
    "a farmhand at harvest time",
    "an amateur boxer",
    "a gambler on a long losing streak",
    "a man recently released from prison",
    "a head chef in a busy kitchen",
    "a tailor",
    "a mountaineer just back from a climb",
    "a heavy smoker who has just quit",
    "a professional gardener",
    "a competitive long-distance swimmer",
    "a locksmith",
    "a beekeeper",
    "a coal miner",
    "a forger of documents",
    "a widower in early mourning",
    "someone who has just emigrated from a hot country to a cold one",
    "a church organist",
    "a diamond setter (jeweller)",
    "a ballet dancer",
    "a bus conductor near the end of a double shift",
]

SYSTEM = (
    "You design observation puzzles in the style of Sherlock Holmes. You are "
    "given a HIDDEN ANSWER: a person's identity, occupation, or situation. Invent "
    "3 to 5 concrete, specific, OBSERVABLE cues — physical marks, wear on clothes "
    "or hands, posture, habits, small behaviours — that a keen observer could "
    "notice and that together point clearly to that answer.\n"
    "HARD RULES for the cues and the scenario:\n"
    "- INDIRECT ONLY. Never state or name the answer, the job title, or the "
    "profession's signature tools/instruments by name (e.g. for a violinist do "
    "NOT mention a violin or violin case; use the chin/jaw mark, calloused "
    "fingertips, the way they hold things). The reader must INFER it.\n"
    "- No dialogue that reveals the answer (do not have the person say what they "
    "do or where they work).\n"
    "- Each cue must be something physically visible, not an interpretation.\n"
    "Then write one short scenario (2-3 sentences) describing a stranger showing "
    "those cues, ending with the question 'What do you make of them?'. Output "
    "EXACTLY this format and nothing else:\n"
    "CUES:\n- <cue>\n- <cue>\n- <cue>\nSCENARIO: <2-3 sentences ending with 'What "
    "do you make of them?'>"
)


def ollama_chat(messages: list[dict], num_predict: int, seed: int) -> str:
    r = requests.post(OLLAMA, json={
        "model": "deepseek-r1:7b", "messages": messages, "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0.8, "seed": seed},
    }, timeout=600)
    r.raise_for_status()
    msg = r.json().get("message", {})
    content = msg.get("content") or ""
    if "<think>" in content:  # strip any inline reasoning
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


def claude_chat(system: str, user: str, timeout: int = 180) -> str:
    """Generate via the Claude Code CLI in headless/print mode — the same pattern
    other projects use to call `claude` non-interactively. The base 7B is too
    weak a scenario generator (30-seed run: 7/30 usable, code-switching, nonsense
    cues); a stronger model here is PROVENANCE-SAFE because scenarios are prompts
    (inputs), not the reasoning traces we distill (those stay base-model).

    Requires `claude` on PATH and authenticated (uses whatever model Claude Code
    is configured with). The prompt is piped on stdin to avoid arg-escaping on
    long text. NOTE: built but NOT executed in the build sandbox (no `claude`
    there) — verify the flags match your own working invocation.
    """
    prompt = f"{system}\n\nHIDDEN ANSWER context:\n{user}"
    r = subprocess.run(["claude", "-p", "--output-format", "text"],
                       input=prompt, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI failed ({r.returncode}): "
                           f"{r.stderr.strip()[:300]}")
    return r.stdout.strip()


def parse(out: str) -> tuple[list[str], str]:
    """Best-effort extraction of cue bullets and the scenario sentence."""
    cues = [re.sub(r"^[-*]\s*", "", ln).strip()
            for ln in out.splitlines() if re.match(r"^\s*[-*]\s+", ln)]
    m = re.search(r"SCENARIO:\s*(.+)", out, re.DOTALL | re.IGNORECASE)
    scenario = m.group(1).strip() if m else ""
    # keep only up to the question if the model rambled past it
    q = scenario.lower().find("what do you make of them?")
    if q != -1:
        scenario = scenario[:q + len("what do you make of them?")]
    return cues, scenario


# Generic words in the seed identities that are NOT giveaways — ignore them when
# checking for answer leakage.
_LEAK_STOP = {
    "professional", "amateur", "competitive", "head", "retired", "recently",
    "coming", "early", "badly", "long", "just", "very", "new", "who", "has",
    "off", "from", "hot", "cold", "country", "week", "before", "near", "end",
    "man", "woman", "person", "people", "someone", "their", "into", "over",
    # generic geographic/size descriptors that appear in labels but aren't the
    # giveaway (e.g. "deep-sea fisherman" — the answer is the trade, not "deep").
    "deep", "sea", "high", "low", "busy", "big", "small", "young",
}


def detect_leak(ground_truth: str, cues: list[str], scenario: str) -> tuple[bool, list[str]]:
    """Flag a scenario that GIVES AWAY its answer — the core curation check,
    since plain SFT has no good-vs-bad signal other than what we keep.

    A leak = a content word of the ground-truth identity (or a morphological
    variant of it) appearing in the cues/scenario. Catches the observed failures:
    'she mentions working night shifts' (night/shift), 'a violin case' (violin ~
    violinist). Returns (is_leak, leaked_terms). This flags the direct case;
    signature-tool leaks not derived from the answer string still need human
    curation, so `leak=False` is 'no OBVIOUS leak', not a guarantee.
    """
    content = [w for w in re.findall(r"[a-z]+", ground_truth.lower())
               if len(w) >= 4 and w not in _LEAK_STOP]
    text_tokens = set(re.findall(r"[a-z]+", (" ".join(cues) + " " + scenario).lower()))
    leaked = []
    for w in content:
        hit = w in text_tokens or (
            len(w) >= 5 and any(t.startswith(w[:4]) for t in text_tokens))
        if hit:
            leaked.append(w)
    return (bool(leaked), leaked)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="first N seeds")
    ap.add_argument("--backend", choices=("ollama", "claude"), default="ollama",
                    help="ollama = local deepseek-r1:7b (weak: ~7/30 usable). "
                         "claude = the Claude Code CLI (headless), far stronger "
                         "and provenance-safe for scenario PROMPTS.")
    ap.add_argument("--num-predict", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--flag", default=None,
                    help="re-score an EXISTING jsonl for leaks in place, then exit "
                         "(no generation). Use to apply the leak filter to a run "
                         "produced before the filter existed.")
    args = ap.parse_args()

    # --flag: curation-only pass over an existing file.
    if args.flag:
        p = Path(args.flag)
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        for r in rows:
            leak, terms = detect_leak(r["ground_truth"], r.get("cues", []),
                                      r.get("scenario_prompt", ""))
            r["leak"], r["leaked_terms"] = leak, terms
            r["usable"] = bool(r.get("parse_ok")) and not leak
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"re-scored {len(rows)}: {sum(r['leak'] for r in rows)} leaked, "
              f"{sum(r['usable'] for r in rows)} usable -> {p}", flush=True)
        return

    seeds = SEED_IDENTITIES[:args.limit] if args.limit else SEED_IDENTITIES
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "data" / "sft" / f"reverse_scenarios_{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w", encoding="utf-8")

    parsed = leaked = usable = 0
    for i, identity in enumerate(seeds):
        user = f"HIDDEN ANSWER: {identity}"
        if args.backend == "claude":
            raw = claude_chat(SYSTEM, user)
        else:
            raw = ollama_chat(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": user}],
                args.num_predict, args.seed)
        cues, scenario = parse(raw)
        ok = len(cues) >= 2 and scenario.endswith("What do you make of them?")
        leak, terms = detect_leak(identity, cues, scenario)
        use = ok and not leak
        row = {"id": i, "ground_truth": identity, "cues": cues,
               "scenario_prompt": scenario, "parse_ok": ok,
               "leak": leak, "leaked_terms": terms, "usable": use,
               "generator": args.backend, "raw": raw}
        f.write(json.dumps(row) + "\n")
        f.flush()
        parsed += ok
        leaked += leak
        usable += use
        tag = "USABLE" if use else ("LEAK  " if leak else "BADFMT")
        print(f"[{i:>2}] {tag} {identity}"
              + (f"  (leaked: {','.join(terms)})" if leak else ""), flush=True)
        if ok and not leak:
            print(f"      cues: {len(cues)} | {scenario[:88]}", flush=True)
    f.close()
    print(f"\n{parsed}/{len(seeds)} parsed, {leaked} leaked -> "
          f"{usable} USABLE  ->  {out}", flush=True)


if __name__ == "__main__":
    main()
