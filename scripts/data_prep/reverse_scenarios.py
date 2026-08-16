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
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
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


def resolve_claude_bin() -> str | None:
    """CLAUDE_BIN env -> PATH -> newest CLI bundled with the Claude desktop app.

    The desktop-app fallback is the load-bearing branch on this machine: the CLI
    is NOT on PATH here, so the original PATH-only lookup raised FileNotFoundError
    and no `--backend claude` call could ever have run. Mirrors the resolver in
    the French project's `journal/app/main.py`, which is the working invocation
    this pattern was copied from.
    """
    env_bin = os.environ.get("CLAUDE_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    bundled = glob.glob(str(
        Path.home() / "Library/Application Support/Claude/claude-code"
                      "/*/claude.app/Contents/MacOS/claude"))
    if bundled:
        # newest by numeric version, e.g. 2.1.229
        return max(bundled, key=lambda p: [
            int(x) if x.isdigit() else 0 for x in Path(p).parts[-4].split(".")])
    return None


def claude_cli(prompt: str, timeout: int = 300) -> str:
    """Run one headless `claude -p` generation and return stdout.

    Two deliberate choices, both measured on 2026-08-15:

    * **Runs in an empty temp cwd.** The CLI loads the project's CLAUDE.md as
      context when invoked inside the repo; this repo's is large enough that a
      trivial call cost $0.34 vs $0.072 from a clean directory — a ~4.7x saving
      per call, and the project context is irrelevant to trace generation anyway.
    * **Drops ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN**, so the CLI uses the
      interactive subscription auth rather than silently billing an API key that
      happens to be in the environment. Same guard as the French project.

    The prompt goes on stdin to avoid arg-escaping on long text.
    """
    claude_bin = resolve_claude_bin()
    if claude_bin is None:
        raise RuntimeError(
            "Claude Code CLI not found. Set CLAUDE_BIN, put `claude` on PATH, "
            "or install the desktop app.")
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    with tempfile.TemporaryDirectory() as cwd:
        r = subprocess.run([claude_bin, "-p", "--output-format", "text"],
                           input=prompt, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI failed ({r.returncode}): "
                           f"{r.stderr.strip()[:300]}")
    return r.stdout.strip()


def claude_chat(system: str, user: str, timeout: int = 300) -> str:
    """Scenario generation via the Claude Code CLI. The base 7B is too weak a
    scenario generator (30-seed run: 7/30 usable, code-switching, nonsense cues);
    a stronger model here is PROVENANCE-SAFE because scenarios are prompts
    (inputs). Trace provenance is a separate decision (2026-08-15: traces are
    also Claude-distilled, STaR-style).
    """
    return claude_cli(f"{system}\n\nHIDDEN ANSWER context:\n{user}", timeout)


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


# --- Scenario disambiguation check --------------------------------------------
# A scenario whose cues admit several equally-good answers is a defective
# training item under ANY verifier: the teacher can reason impeccably and still
# "fail", and if it passes, the student is taught to guess. Measured 2026-08-15 —
# 3 of 5 filter misses were this, not reasoning errors (organist/drummer,
# gambler/ruined-man, conductor/roundsman).
#
# The check sees the CUES ONLY. It must never see the ground truth, or it would
# rationalise toward the intended answer and report every scenario as clear.
DISAMBIG_SYSTEM = """You are shown only a list of physical cues observed about a \
stranger. Nobody has told you who they are.

List every identity — trade, station, or situation — that fits ALL of these cues \
about equally well. Then say whether one candidate is clearly the best \
explanation, or whether several are equally good.

Be strict. Two candidates are "equally good" only if each explains every cue as \
well as the other. If one explains the cues more completely or more specifically, \
it is clearly best.

Reply in EXACTLY this form, nothing else. VERDICT must be the single word CLEAR \
or AMBIGUOUS — no other word. The two lines must agree: CLEAR requires a named \
BEST, and AMBIGUOUS requires BEST: NONE.
CANDIDATES:
- <identity>
- <identity>
VERDICT: CLEAR
BEST: <the single best identity, or NONE if several tie>
REASON: <one short line>"""


def build_disambig_prompt(cues: list[str]) -> str:
    bullets = "\n".join(f"- {c}" for c in cues)
    return f"{DISAMBIG_SYSTEM}\n\nOBSERVED CUES:\n{bullets}"


# Measured 2026-08-15: the verdict WORD is not reliable on its own. The model
# used "TIE" (outside the requested vocabulary) once, and on two scenarios wrote
# "VERDICT: CLEAR" together with "BEST: NONE" — meaning the *class* was clear but
# no single candidate won (violinist/violist; immigrant/tourist/student/refugee).
# The operational question the prompt actually asks is "is there ONE clearly best
# identity?", so **BEST is authoritative** and the verdict word corroborates it.
# This reads the model's own answer more faithfully; it does not move the
# criterion to recover particular scenarios.
_DIS_VERDICT_RE = re.compile(
    r"VERDICT\s*[:\-]\s*(CLEAR|AMBIGUOUS|TIE|UNCLEAR|EQUAL)\b", re.I)
_DIS_BEST_RE = re.compile(r"BEST\s*[:\-]\s*(.+)")
_DIS_REASON_RE = re.compile(r"REASON\s*[:\-]\s*(.+)")
_AMBIGUOUS_WORDS = {"AMBIGUOUS", "TIE", "UNCLEAR", "EQUAL"}


def parse_disambig(raw: str) -> tuple[bool | None, list[str], str, str]:
    """Parse a disambiguation reply into (ambiguous, candidates, best, reason).

    `ambiguous` is None only when NEITHER a verdict word nor a BEST line can be
    found, so an unreadable reply FAILS CLOSED — the scenario is neither silently
    kept nor silently dropped, it is visibly `null` and gets looked at.
    """
    text = re.sub(r"^\s*```[a-z]*\s*|\s*```\s*$", "", (raw or "").strip())
    cands: list[str] = []
    in_block = False
    for line in text.splitlines():
        st = line.strip()
        if re.match(r"^CANDIDATES\s*[:\-]", st, re.I):
            in_block = True
            continue
        if in_block:
            if re.match(r"^[-*]\s+", st):
                cands.append(re.sub(r"^[-*]\s+", "", st).strip())
                continue
            if st:                      # any non-bullet line ends the list
                in_block = False

    b = _DIS_BEST_RE.search(text)
    has_best_line = b is not None
    best = b.group(1).strip() if b else ""
    if re.match(r"^(NONE|N/?A)\b", best, re.I):
        best = ""

    m = _DIS_VERDICT_RE.search(text)
    word_says_ambiguous = m.group(1).upper() in _AMBIGUOUS_WORDS if m else False

    if not m and not has_best_line:
        return None, cands, "", text[:200].replace("\n", " ").strip()

    # No single best identity => ambiguous, whatever the verdict word claimed.
    ambiguous = word_says_ambiguous or not best
    r = _DIS_REASON_RE.search(text)
    return ambiguous, cands, best, (r.group(1).strip() if r else "")


def disambiguate(cues: list[str],
                 timeout: int = 180) -> tuple[bool | None, list[str], str, str, str]:
    """Return (ambiguous, candidates, best, reason, raw) for one scenario's cues."""
    raw = claude_cli(build_disambig_prompt(cues), timeout=timeout)
    ambiguous, cands, best, reason = parse_disambig(raw)
    return ambiguous, cands, best, reason, raw


def disambiguate_file(src: Path, out: Path | None) -> None:
    """Annotate a scenarios file with BOTH scenario-defect checks.

    Writes an annotated COPY; the input is never modified, so the seed set cannot
    be destroyed by a check run against it.

    Two independent defects, measured 2026-08-15 — the second was NOT anticipated
    and the ambiguity check alone does not catch it:

    1. **ambiguous** — the cues admit several equally-good identities, so no
       verifier can fairly grade an answer (violinist/violist; immigrant vs
       tourist vs refugee).
    2. **cues_miss_gt** — the cues point clearly at something OTHER than the seed
       identity. The gambler scenario is "clear" and its best answer is "a man
       hiding financial hardship": nothing in the cues says *gambling*, so the
       seed label is unreachable and every trace on it is graded against an
       answer the evidence does not support. Detected by judging the check's own
       `best` against the ground truth, reusing the trace judge.

    Either defect makes the item unusable.
    """
    from generate_traces import judge_answer   # late: generate_traces imports us

    src = src if src.is_absolute() else ROOT / src
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    out = out or src.with_name(src.stem + "_disambig.jsonl")
    out = out if out.is_absolute() else ROOT / out
    f = out.open("w", encoding="utf-8")
    for r in rows:
        amb, cands, best, reason, raw = disambiguate(r.get("cues", []))
        r["ambiguous"], r["candidates"] = amb, cands
        r["disambig_best"], r["disambig_reason"], r["disambig_raw"] = \
            best, reason, raw
        # Does the cues' own best answer actually reach the seed identity?
        if best:
            v, vreason, _ = judge_answer(r["ground_truth"], best)
            r["best_reaches_gt"], r["best_reaches_gt_reason"] = v, vreason
        else:
            r["best_reaches_gt"], r["best_reaches_gt_reason"] = None, ""
        r["cues_miss_gt"] = r["best_reaches_gt"] is False
        r["usable"] = (bool(r.get("parse_ok")) and not r.get("leak")
                       and amb is False and not r["cues_miss_gt"])
        f.write(json.dumps(r) + "\n")
        f.flush()
        if amb:
            tag = "AMBIGUOUS"
        elif r["cues_miss_gt"]:
            tag = "CUES!=GT "
        elif amb is None:
            tag = "UNPARSED "
        else:
            tag = "ok       "
        print(f"  [{r.get('id'):>2}] {tag} gt={r['ground_truth'][:34]:<34} "
              f"cands={len(cands)}"
              + (f" | best: {best[:38]}" if best else " | best: NONE"), flush=True)
        if amb:
            print(f"       tie between: {', '.join(cands[:4])}", flush=True)
        if r["cues_miss_gt"]:
            print(f"       cues point elsewhere: {r['best_reaches_gt_reason'][:88]}",
                  flush=True)
    f.close()
    amb_n = sum(r["ambiguous"] is True for r in rows)
    miss_n = sum(r["cues_miss_gt"] for r in rows)
    print(f"\n{amb_n}/{len(rows)} AMBIGUOUS | {miss_n}/{len(rows)} CUES!=GT | "
          f"{sum(r['usable'] for r in rows)}/{len(rows)} usable -> {out}", flush=True)


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
    ap.add_argument("--disambiguate", default=None, metavar="SCENARIOS.JSONL",
                    help="run the Claude disambiguation check over an EXISTING "
                         "scenarios file, then exit (no generation). Flags "
                         "scenarios whose cues admit >1 equally-valid answer. "
                         "Writes an annotated copy; input is never modified.")
    args = ap.parse_args()

    if args.disambiguate:
        disambiguate_file(Path(args.disambiguate),
                          Path(args.out) if args.out else None)
        return

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
