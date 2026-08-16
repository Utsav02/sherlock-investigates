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

VERIFIER (2026-08-15): the gate is the LLM judge (`--judge`, V-STaR style), not
the keyword match. The lexical `answer_matches` scored correct-but-differently-
worded answers as no-match ("former soldier turned commissionaire" vs ground
truth "a retired sergeant of the Royal Marines"); it is retained per row as a
cheap cross-check and the two are reported as a confusion table.

    python scripts/data_prep/generate_traces.py --backend claude --judge \
        --scenarios data/sft/scenarios_seed_claude.jsonl --samples 1
    python scripts/data_prep/generate_traces.py --rejudge data/sft/traces_x.jsonl

CAVEAT NEITHER verifier can see: both check the ANSWER only. A trace can reach
the right answer through hollow or templated reasoning, and it is the REASONING
that gets SFT'd into the student. The think blocks must be READ on a sample of
every batch before it is used as training data.

Output: one row per (scenario, sample) with `keeper`, `matched` (keyword
cross-check), `judge`/`judge_reason` when judged, and the think/answer.
Downstream: keeper==True rows, chat-formatted, become the SFT set (+ OpenThoughts
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
    a hit in an answer is a correct deduction (good).

    NOTE (2026-08-15): retained as a cheap CROSS-CHECK, no longer the gate. It is
    a lexical test on a semantic question and measured too strict — it scored
    "This is a former soldier turned commissionaire..." as no-match against
    ground truth "a retired sergeant of the Royal Marines". `judge_answer` is the
    gate. This function is deliberately NOT loosened to recover such cases:
    widening a matcher until it accepts the answers you already believe are right
    is the test-set-fitting that ended the `t_think_07` regex patching.
    """
    return detect_leak(ground_truth, [], answer)


# --- LLM-judge verifier (V-STaR style) ----------------------------------------
# The rubric is written ONCE, from the principle, and deliberately not retuned
# against the 18-scenario validation set. The distinction it must draw:
#   * a correct answer stated at a coarser grain is CORRECT ("an old soldier" for
#     "a retired sergeant of the Royal Marines" — the cues cannot encode a
#     specific service, so demanding it would penalise sound deduction);
#   * a DIFFERENT identity is INCORRECT ("a professional drummer" for "a church
#     organist"), however well-argued.
JUDGE_SYSTEM = """You are grading one answer to an observation puzzle.

You are given the TRUE identity of the person and a CANDIDATE answer written by \
someone who saw only physical cues. Decide whether the candidate identified the \
same person.

Judge SUBSTANCE, not wording. Apply these rules:
- Correct if the candidate names the same trade, station, or situation, even in \
different words (e.g. "lorry driver" and "long-haul truck driver").
- Correct if the candidate is MORE GENERAL than the truth but consistent with it \
(e.g. "an old soldier" for "a retired sergeant of the Royal Marines"). The cues \
cannot encode every specific, so a correct-but-coarser answer counts.
- INCORRECT if the candidate names a different identity, however plausibly \
argued (e.g. "a professional drummer" for "a church organist").
- INCORRECT if the candidate only describes a mood, a circumstance, or a generic \
person without reaching the trade or situation.
- Ignore differences of gender, era, or literary flourish.

Reply in EXACTLY this form, two lines, nothing else:
VERDICT: YES
REASON: <one short line>"""


def build_judge_prompt(ground_truth: str, answer: str) -> str:
    return (f"{JUDGE_SYSTEM}\n\nTRUE IDENTITY: {ground_truth}\n"
            f"CANDIDATE ANSWER: {answer}")


_VERDICT_RE = re.compile(r"VERDICT\s*[:\-]\s*(YES|NO)\b", re.I)
_REASON_RE = re.compile(r"REASON\s*[:\-]\s*(.+)", re.I)


def parse_judge(raw: str) -> tuple[bool | None, str]:
    """Parse a judge reply into (verdict, reason).

    Returns verdict None when no VERDICT line can be found, so an unparseable
    judgement FAILS CLOSED (the row cannot become a keeper) and is visible in the
    data as `judge=null` rather than being silently counted either way. Tolerates
    markdown fences, a bare leading YES/NO, and case variation.
    """
    text = re.sub(r"^\s*```[a-z]*\s*|\s*```\s*$", "", (raw or "").strip())
    m = _VERDICT_RE.search(text)
    if m:
        verdict = m.group(1).upper() == "YES"
    else:
        # bare "YES"/"NO" on the first line, no VERDICT: label
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        if re.fullmatch(r"(YES|NO)\b[.!]?", first, re.I):
            verdict = first.upper().startswith("YES")
        else:
            return None, text[:200].replace("\n", " ").strip()
    r = _REASON_RE.search(text)
    reason = r.group(1).strip() if r else ""
    return verdict, reason


def judge_answer(ground_truth: str, answer: str,
                 timeout: int = 180) -> tuple[bool | None, str, str]:
    """Ask the Claude judge whether `answer` identifies `ground_truth`.
    Returns (verdict, reason, raw)."""
    raw = claude_cli(build_judge_prompt(ground_truth, answer), timeout=timeout)
    verdict, reason = parse_judge(raw)
    return verdict, reason, raw


def agreement_report(rows: list[dict]) -> str:
    """Keyword-match vs judge confusion, printed after any judged run. The
    disagreement cells are the interesting ones: judge-YES/keyword-NO are the
    semantic answers the lexical filter cannot see."""
    j_yes_k_yes = sum(r.get("judge") is True and r.get("matched") for r in rows)
    j_yes_k_no = sum(r.get("judge") is True and not r.get("matched") for r in rows)
    j_no_k_yes = sum(r.get("judge") is False and r.get("matched") for r in rows)
    j_no_k_no = sum(r.get("judge") is False and not r.get("matched") for r in rows)
    unparsed = sum(r.get("judge") is None for r in rows)
    n = len(rows)
    agree = j_yes_k_yes + j_no_k_no
    return (
        f"\n  judge vs keyword-match (n={n})\n"
        f"                     keyword YES   keyword NO\n"
        f"    judge YES        {j_yes_k_yes:>11}   {j_yes_k_no:>10}   <- recovered by the judge\n"
        f"    judge NO         {j_no_k_yes:>11}   {j_no_k_no:>10}\n"
        f"    unparseable judge replies: {unparsed}\n"
        f"    agreement: {agree}/{n}"
        + (f" ({agree / n:.0%})" if n else ""))


def rejudge_file(src: Path, out: Path | None) -> None:
    """Re-score an existing traces file with the LLM judge, without regenerating.

    Writes an annotated COPY — the input is never modified, so a judged rescore
    can never destroy the traces it grades.
    """
    src = src if src.is_absolute() else ROOT / src
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    out = out or src.with_name(src.stem + "_judged.jsonl")
    out = out if out.is_absolute() else ROOT / out
    f = out.open("w", encoding="utf-8")
    for r in rows:
        verdict, reason, raw = judge_answer(r["ground_truth"], r["answer"])
        r["judge"], r["judge_reason"], r["judge_raw"] = verdict, reason, raw
        r["keeper"] = bool(verdict) and bool(r.get("has_think"))
        f.write(json.dumps(r) + "\n")
        f.flush()
        flag = "  <- keyword MISSED it" if verdict and not r.get("matched") else ""
        vs = {True: "YES", False: "NO ", None: "???"}[verdict]
        print(f"  [{r.get('scenario_id'):>2}] judge={vs} keyword="
              f"{'YES' if r.get('matched') else 'NO '} "
              f"gt={r['ground_truth'][:36]:<36}{flag}", flush=True)
    f.close()
    keep = sum(r["keeper"] for r in rows)
    print(agreement_report(rows), flush=True)
    print(f"\n{keep}/{len(rows)} keepers by JUDGE -> {out}", flush=True)


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
    ap.add_argument("--judge", action="store_true",
                    help="gate keepers on the Claude LLM judge instead of the "
                         "keyword match. The keyword match is still recorded per "
                         "row as a cross-check and their agreement is reported.")
    ap.add_argument("--rejudge", default=None, metavar="TRACES.JSONL",
                    help="re-score an EXISTING traces file with the judge, then "
                         "exit (no generation). Writes an annotated copy; the "
                         "input file is never modified.")
    args = ap.parse_args()

    if args.rejudge:
        rejudge_file(Path(args.rejudge), Path(args.out) if args.out else None)
        return

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
    judged: list[dict] = []
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
            row = {"scenario_id": s.get("id"), "ground_truth": gt,
                   "scenario_prompt": prompt, "sample": k, "seed": seed,
                   "think": think, "answer": answer,
                   "has_think": has_think, "matched": matched,
                   "matched_terms": terms,
                   "generator": GENERATORS[args.backend]}
            if args.judge:
                # The judge is the gate; `matched` stays as a cross-check.
                verdict, reason, jraw = judge_answer(gt, answer)
                row["judge"], row["judge_reason"], row["judge_raw"] = \
                    verdict, reason, jraw
                keep = bool(verdict) and has_think
            else:
                keep = matched and has_think
            row["keeper"] = keep
            if raw is not None:
                row["raw"] = raw
            f.write(json.dumps(row) + "\n")
            f.flush()
            judged.append(row)
            total += 1
            keepers += keep
            hit_this = hit_this or keep
            tag = "KEEP" if keep else ("no-match" if has_think else "no-think")
            print(f"  [{s.get('id'):>2}.{k}] {tag:<9} gt={gt[:34]:<34} "
                  f"ans={answer[:44].replace(chr(10),' ')}", flush=True)
        scen_with_keeper += hit_this
    f.close()
    if args.judge:
        print(agreement_report(judged), flush=True)
    gate = "JUDGE" if args.judge else "keyword match"
    print(f"\n{keepers}/{total} traces kept by {gate} | "
          f"{scen_with_keeper}/{len(scenarios)} scenarios have >=1 keeper -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
