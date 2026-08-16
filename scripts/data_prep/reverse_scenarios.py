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


_SUFFIXES = ("ists", "ist", "ings", "ing", "ers", "er", "ed", "es", "s")


def _stem(w: str) -> str:
    """Crude suffix strip, only for matching morphological variants."""
    for suf in _SUFFIXES:
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _variant_hit(w: str, text_tokens: set[str]) -> bool:
    """Is ground-truth word `w` present in the text, allowing variants?

    Three rules, deliberately narrow (see the false positive that motivated the
    rewrite, below):

    1. exact token match                     — 'nurse' in 'nurse'
    2. a text token is a PREFIX of `w`       — 'lock' gives away 'locksmith',
       'violin' gives away 'violinist'
    3. equal stems                           — 'gardening' ~ 'gardener'

    The original rule was `any(t.startswith(w[:4]))`, which asks whether any text
    token merely *begins with* the first four letters of the answer. Measured on
    a real generation (2026-08-15): a genuinely clean blacksmith scenario —
    which never names the trade — was rejected because a cue said "blackened"
    creases, and 'blackened'.startswith('blac'). Rule 2 reverses the direction of
    the test (the text token must be a prefix of the ANSWER, not the other way
    round), which keeps 'lock'/'locksmith' while dropping
    'blackened'/'blacksmith'. The bias mattered at scale: it silently discarded
    scenarios for every identity sharing a four-letter prefix with a common
    descriptive word, and a discarded scenario is invisible in the output.
    """
    if w in text_tokens:
        return True
    ws = _stem(w)
    for t in text_tokens:
        if len(t) >= 4 and len(t) < len(w) and w.startswith(t):
            return True
        if len(ws) >= 4 and _stem(t) == ws:
            return True
    return False


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
    leaked = [w for w in content if _variant_hit(w, text_tokens)]
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
    """Annotate a scenarios file with the scenario-defect check.

    Writes an annotated COPY; the input is never modified, so the seed set cannot
    be destroyed by a check run against it.

    **ambiguous** — the cues admit several equally-good identities, so no
    verifier can fairly grade an answer (violinist/violist; immigrant vs tourist
    vs refugee). Measured 2026-08-15: 6/18, four of them never suspected by eye.
    An ambiguous item is unusable.

    Removed 2026-08-15 (same day it was added): a second check, `cues_miss_gt`,
    judged the check's own `best` against the ground truth to catch a seed label
    the cues cannot reach. It had **12 opportunities and fired 0 times** — the
    gambler case it was built for turned out to be *ambiguous*, so `best` was
    NONE and the check never ran on it. It cost ~1 extra judge call per
    unambiguous scenario and bought nothing measurable, so it is dropped rather
    than carried into the scaled run. The defect it targets is real in principle;
    if a scaled batch produces a scenario whose cues clearly point elsewhere and
    the ambiguity check passes it, reinstate it from git history.
    """
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
        r["usable"] = (bool(r.get("parse_ok")) and not r.get("leak")
                       and amb is False)
        f.write(json.dumps(r) + "\n")
        f.flush()
        if amb:
            tag = "AMBIGUOUS"
        elif amb is None:
            tag = "UNPARSED "
        else:
            tag = "ok       "
        print(f"  [{r.get('id'):>2}] {tag} gt={r['ground_truth'][:34]:<34} "
              f"cands={len(cands)}"
              + (f" | best: {best[:38]}" if best else " | best: NONE"), flush=True)
        if amb:
            print(f"       tie between: {', '.join(cands[:4])}", flush=True)
    f.close()
    amb_n = sum(r["ambiguous"] is True for r in rows)
    print(f"\n{amb_n}/{len(rows)} AMBIGUOUS | "
          f"{sum(r['usable'] for r in rows)}/{len(rows)} usable -> {out}", flush=True)


# --- Identity-pool expansion --------------------------------------------------
# SEED_IDENTITIES is ~30, which caps a scaled run far below the ~100-150 SFT
# examples a pilot needs once the two gates (leak, disambiguation) and the
# downstream keeper filter have taken their cut. Measured end-to-end yield is
# 50% at the scenario stage alone (2026-08-15), so the identity pool has to be
# roughly 2x the target scenario count before traces are ever generated.
#
# Identities are GROUND TRUTHS, i.e. labels — not model reasoning — so sourcing
# them from Claude raises no provenance question (same argument as scenarios:
# they are inputs, and the 2026-08-15 provenance entry scopes the concern to
# the reasoning traces).

# Rotated per batch so one category cannot dominate the pool. Diversity here is
# load-bearing: a pool of 200 near-identical manual trades would yield cue sets
# that collide with each other and inflate the ambiguity drop rate.
IDENTITY_CATEGORIES = [
    "manual trades and crafts, where the hands and clothes carry the work",
    "professions, clerical and indoor work",
    "sport, physical training and outdoor pursuits",
    "medical conditions, physiological states and recent injuries",
    "life situations and recent transitions (moves, losses, releases, arrivals)",
    "transport, sea and travel occupations",
    "habits, vices, and recently-broken habits",
    "performing arts and music",
    "service, food and hospitality work",
    "rural, agricultural and animal-handling work",
]

IDENTITY_SYSTEM = """You are building a pool of HIDDEN ANSWERS for \
Sherlock-Holmes-style observation puzzles.

Each answer is a person's identity, occupation, or situation that a keen observer \
could deduce from PHYSICAL, OBSERVABLE traces alone: marks and callouses on the \
hands, wear on clothing, posture, gait, skin, small involuntary habits.

HARD RULES:
- Each answer must be deducible from visible physical traces. Reject anything \
whose only evidence would be speech, documents, or a possession that names it.
- Each answer must be DISTINCT from every other one you write and from every \
entry on the AVOID list.
- Write each as a short lowercase noun phrase beginning "a", "an", or "someone \
who", in exactly this style:
    a deep-sea trawler fisherman
    someone who has just emigrated from a hot country to a cold one
- Output ONE PER LINE. No numbering, no bullets, no blank lines, no commentary, \
no preamble. Nothing but the answers."""


def normalize_identity(s: str) -> str:
    """Comparison key for dedup: lowercase, no article, no punctuation."""
    s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
    s = re.sub(r"^\s*(a|an|the)\s+", " ", " " + s + " ")
    return re.sub(r"\s+", " ", s).strip()


_IDENTITY_STOP = {"a", "an", "the", "of", "who", "has", "just", "in", "on",
                  "at", "to", "from", "and", "with", "very", "someone"}


def _identity_words(s: str) -> set[str]:
    return {w for w in normalize_identity(s).split() if w not in _IDENTITY_STOP}


def parse_identity_list(raw: str) -> list[str]:
    """Pull identity lines out of a model reply, tolerating bullets/numbering."""
    out = []
    for line in (raw or "").splitlines():
        s = line.strip()
        s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", s)
        s = re.sub(r"^\s*```[a-z]*\s*$", "", s).strip().strip('"').strip()
        if not s or len(s) < 6 or len(s) > 120:
            continue
        # Must read like the seed style; drops headers, commentary and prose.
        if not re.match(r"^(a|an|someone|somebody)\b", s, re.I):
            continue
        if s.endswith(":") or "\t" in s:
            continue
        out.append(re.sub(r"[.;,]+$", "", s).strip())
    return out


def dedup_identities(new: list[str], existing: list[str],
                     jaccard_max: float = 0.6) -> list[str]:
    """Drop exact and near-duplicate identities, preserving order.

    Near-duplicate = content-word Jaccard above `jaccard_max` (so "a watchmaker"
    survives alongside "a diamond setter" but a second "a professional concert
    violinist" does not). Near-dupes are dropped to avoid paying two CLI calls
    for one scenario, NOT because two similar trades in the pool would be
    invalid — each scenario is disambiguated on its own cues.
    """
    keys = {normalize_identity(e) for e in existing}
    words = [_identity_words(e) for e in existing]
    kept: list[str] = []
    for cand in new:
        k = normalize_identity(cand)
        if not k or k in keys:
            continue
        cw = _identity_words(cand)
        if not cw:
            continue
        dup = False
        for w in words:
            union = cw | w
            if union and len(cw & w) / len(union) > jaccard_max:
                dup = True
                break
        if dup:
            continue
        keys.add(k)
        words.append(cw)
        kept.append(cand)
    return kept


def expand_identities(target: int, out_path: Path, per_call: int = 25,
                      max_calls: int = 40) -> list[str]:
    """Generate `target` NEW identities beyond SEED_IDENTITIES, appending as
    produced so an interrupted expansion keeps everything it had.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pool: list[str] = []
    if out_path.exists():                       # resume: reuse what is on disk
        pool = [json.loads(l)["identity"]
                for l in out_path.read_text().splitlines() if l.strip()]
        print(f"resuming identity pool: {len(pool)} already on disk", flush=True)
    fh = out_path.open("a", encoding="utf-8")
    calls = 0
    while len(pool) < target and calls < max_calls:
        cat = IDENTITY_CATEGORIES[calls % len(IDENTITY_CATEGORIES)]
        avoid = "\n".join(f"- {x}" for x in (SEED_IDENTITIES + pool))
        prompt = (f"{IDENTITY_SYSTEM}\n\nCATEGORY for this batch: {cat}\n"
                  f"Give exactly {per_call} answers in that category.\n\n"
                  f"AVOID (already in the pool, do not repeat or paraphrase):\n"
                  f"{avoid}")
        calls += 1
        try:
            raw = claude_cli(prompt, timeout=300)
        except Exception as exc:                # one bad call must not end it
            print(f"  ! batch {calls} failed: {str(exc)[:160]}", flush=True)
            continue
        fresh = dedup_identities(parse_identity_list(raw),
                                 SEED_IDENTITIES + pool)
        for ident in fresh:
            if len(pool) >= target:
                break
            pool.append(ident)
            fh.write(json.dumps({"identity": ident, "category": cat,
                                 "batch": calls}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"  batch {calls} [{cat[:34]}]: +{len(fresh)} new "
              f"-> pool {len(pool)}/{target}", flush=True)
    fh.close()
    print(f"\nidentity pool: {len(pool)} new identities in {calls} calls "
          f"-> {out_path}", flush=True)
    return pool


# --- Gated, resumable scenario generation -------------------------------------
# The two gates were built as separate file-rewriting passes, which is fine at
# n=18 and wrong at n=200+: a mid-run interruption loses everything, and the
# ambiguous scenarios are only discovered after every generation call is already
# spent. Inline gating lets an AMBIGUOUS scenario be regenerated once while the
# identity is still in hand, and the append-only ledger makes the run resumable.

REGEN_NUDGE = """Your previous cue set for this hidden answer was AMBIGUOUS: an \
independent reader, shown the cues alone, judged these to be equally good \
explanations:
{cands}

Write a COMPLETELY NEW set of 3 to 5 cues for the SAME hidden answer. At least \
one cue must RULE OUT those other explanations. Same hard rules, same output \
format."""


def classify_row(row: dict) -> str:
    """Terminal status of a ledger row. Pure — the report reads only this."""
    if row.get("error"):
        return "error"
    if not row.get("parse_ok"):
        return "badfmt"
    if row.get("leak"):
        return "leak"
    amb = row.get("ambiguous")
    if amb is None:
        return "unparsed_disambig"
    if amb:
        return "ambiguous"
    return "usable"


def load_done(path: Path) -> set[str]:
    """Identities already attempted, read from the ledger itself.

    The ledger is the single source of truth for resume — deliberately, rather
    than a side-car state file. A separate state file can disagree with the
    append log if the process dies between the two writes; a row's presence
    cannot.
    """
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["ground_truth"])
        except (json.JSONDecodeError, KeyError):
            continue                     # a torn final line costs one identity
    return done


def summarize(rows: list[dict]) -> dict:
    """Counts by terminal status, plus the two drop rates worth reporting."""
    counts: dict[str, int] = {}
    for r in rows:
        st = r.get("status") or classify_row(r)
        counts[st] = counts.get(st, 0) + 1
    n = len(rows)
    # Denominator for the ambiguity rate is scenarios that REACHED the check:
    # a scenario dropped for bad format or a leak was never disambiguated.
    reached = sum(counts.get(k, 0)
                  for k in ("usable", "ambiguous", "unparsed_disambig"))
    return {
        "tried": n,
        "counts": counts,
        "usable": counts.get("usable", 0),
        "reached_disambig": reached,
        "ambiguous_rate": (counts.get("ambiguous", 0) / reached) if reached else None,
        "overall_yield": (counts.get("usable", 0) / n) if n else None,
        "regenerated": sum(1 for r in rows if r.get("attempts", 1) > 1),
        "regen_rescued": sum(1 for r in rows if r.get("attempts", 1) > 1
                             and (r.get("status") or classify_row(r)) == "usable"),
    }


def print_summary(rows: list[dict], out: Path, clean: Path) -> None:
    s = summarize(rows)
    print("\n" + "=" * 66, flush=True)
    print(f"identities tried      : {s['tried']}", flush=True)
    for k in ("badfmt", "leak", "ambiguous", "unparsed_disambig", "error"):
        if s["counts"].get(k):
            print(f"  dropped [{k:<17}]: {s['counts'][k]}", flush=True)
    print(f"CLEAN scenarios kept  : {s['usable']}", flush=True)
    if s["ambiguous_rate"] is not None:
        print(f"ambiguity drop rate   : {s['counts'].get('ambiguous', 0)}/"
              f"{s['reached_disambig']} = {s['ambiguous_rate']:.1%} "
              f"(n=18 reference: 6/18 = 33.3%)", flush=True)
    if s["overall_yield"] is not None:
        print(f"scenario-stage yield  : {s['overall_yield']:.1%}", flush=True)
    print(f"regenerated once      : {s['regenerated']} "
          f"({s['regen_rescued']} rescued to usable)", flush=True)
    print(f"ledger -> {out}\nclean  -> {clean}", flush=True)


def generate_one(identity: str, backend: str, num_predict: int, seed: int,
                 extra: str = "") -> dict:
    """One generation call + the leak gate. No disambiguation (that costs a
    second call and is only worth spending on a scenario that got this far)."""
    user = f"HIDDEN ANSWER: {identity}"
    if extra:
        user += "\n\n" + extra
    if backend == "claude":
        raw = claude_chat(SYSTEM, user)
    else:
        raw = ollama_chat([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": user}],
                          num_predict, seed)
    cues, scenario = parse(raw)
    ok = len(cues) >= 2 and scenario.endswith("What do you make of them?")
    leak, terms = detect_leak(identity, cues, scenario)
    return {"cues": cues, "scenario_prompt": scenario, "parse_ok": ok,
            "leak": leak, "leaked_terms": terms, "raw": raw}


def run_gated(identities: list[str], out: Path, clean: Path, backend: str,
              num_predict: int, seed: int, regen: bool = True,
              max_consecutive_errors: int = 5) -> list[dict]:
    """Generate scenarios with BOTH gates applied inline, appending as produced.

    Every attempted identity gets a ledger row — including dropped ones — so the
    run is resumable (`load_done` reads the ledger) and the drop rates are
    auditable rather than inferred from what is missing.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [i for i in identities if i not in done]
    print(f"{len(identities)} identities | {len(done)} already done | "
          f"{len(todo)} to do", flush=True)

    fh = out.open("a", encoding="utf-8")
    cf = clean.open("a", encoding="utf-8")
    rows: list[dict] = []
    consecutive_errors = 0
    for n, identity in enumerate(todo, 1):
        row = {"id": len(done) + n - 1, "ground_truth": identity,
               "generator": backend, "attempts": 1,
               "ts": datetime.now().isoformat(timespec="seconds")}
        try:
            row.update(generate_one(identity, backend, num_predict, seed))
            # Gate 1 (leak / format) passed -> spend the second call on gate 2.
            if row["parse_ok"] and not row["leak"]:
                amb, cands, best, reason, raw_d = disambiguate(row["cues"])
                row.update({"ambiguous": amb, "candidates": cands,
                            "disambig_best": best, "disambig_reason": reason,
                            "disambig_raw": raw_d})
                if amb and regen:
                    row["attempts"] = 2
                    row["first_attempt"] = {
                        "cues": row["cues"],
                        "scenario_prompt": row["scenario_prompt"],
                        "candidates": cands}
                    nudge = REGEN_NUDGE.format(
                        cands="\n".join(f"- {c}" for c in cands[:5]))
                    row.update(generate_one(identity, backend, num_predict,
                                            seed + 1, extra=nudge))
                    if row["parse_ok"] and not row["leak"]:
                        amb, cands, best, reason, raw_d = \
                            disambiguate(row["cues"])
                        row.update({"ambiguous": amb, "candidates": cands,
                                    "disambig_best": best,
                                    "disambig_reason": reason,
                                    "disambig_raw": raw_d})
                    else:
                        row["ambiguous"] = None
            consecutive_errors = 0
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            consecutive_errors += 1
        row["status"] = classify_row(row)
        row["usable"] = row["status"] == "usable"
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        if row["usable"]:
            cf.write(json.dumps({k: row[k] for k in (
                "id", "ground_truth", "cues", "scenario_prompt",
                "disambig_best", "generator", "attempts")}) + "\n")
            cf.flush()
            os.fsync(cf.fileno())
        rows.append(row)

        tag = {"usable": "USABLE", "leak": "LEAK  ", "badfmt": "BADFMT",
               "ambiguous": "AMBIG ", "unparsed_disambig": "UNPARS",
               "error": "ERROR "}[row["status"]]
        star = "*" if row.get("attempts", 1) > 1 else " "
        print(f"[{n:>3}/{len(todo)}]{star}{tag} {identity[:58]}", flush=True)
        if row["status"] == "leak":
            print(f"        leaked: {','.join(row.get('leaked_terms', []))}",
                  flush=True)
        elif row["status"] == "ambiguous":
            print(f"        tie: {', '.join(row.get('candidates', [])[:4])}",
                  flush=True)
        elif row["status"] == "error":
            print(f"        {row['error'][:150]}", flush=True)
        elif row["status"] == "usable":
            print(f"        {row['scenario_prompt'][:86]}", flush=True)

        if consecutive_errors >= max_consecutive_errors:
            print(f"\nABORT: {consecutive_errors} consecutive errors — the CLI "
                  f"or auth is likely down. Re-run the same command to resume.",
                  flush=True)
            break
    fh.close()
    cf.close()
    return rows


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
    ap.add_argument("--expand-identities", type=int, default=None, metavar="N",
                    help="generate N NEW identities beyond SEED_IDENTITIES via "
                         "the Claude CLI, dedup'd, appended as produced, then "
                         "exit. Resumes from --identities-out if it exists.")
    ap.add_argument("--identities-out", default="data/sft/identity_pool.jsonl",
                    help="identity-pool file written/resumed by "
                         "--expand-identities and read by --identities.")
    ap.add_argument("--identities", default=None, metavar="POOL.JSONL",
                    help="generate scenarios for the identities in this pool "
                         "file (plus SEED_IDENTITIES unless --no-seeds), with "
                         "BOTH gates inline. Resumable: re-running the same "
                         "command skips identities already in the ledger.")
    ap.add_argument("--no-seeds", action="store_true",
                    help="with --identities, exclude SEED_IDENTITIES.")
    ap.add_argument("--no-regen", action="store_true",
                    help="drop AMBIGUOUS scenarios outright instead of "
                         "regenerating them once with a discriminating nudge.")
    ap.add_argument("--report", default=None, metavar="LEDGER.JSONL",
                    help="print the drop-rate summary for an existing gated "
                         "ledger, then exit. Read-only.")
    args = ap.parse_args()

    if args.report:
        p = Path(args.report)
        p = p if p.is_absolute() else ROOT / p
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        print_summary(rows, p, p.with_name(p.stem.replace("_all", "") + "_clean.jsonl"))
        return

    if args.expand_identities:
        po = Path(args.identities_out)
        expand_identities(args.expand_identities,
                          po if po.is_absolute() else ROOT / po)
        return

    if args.identities:
        p = Path(args.identities)
        p = p if p.is_absolute() else ROOT / p
        pool = [json.loads(l)["identity"]
                for l in p.read_text().splitlines() if l.strip()]
        idents = (pool if args.no_seeds else SEED_IDENTITIES + pool)
        if args.limit:
            idents = idents[:args.limit]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.out:
            out = Path(args.out)
        else:
            out = ROOT / "data" / "sft" / f"scenarios_gated_{stamp}_all.jsonl"
        out = out if out.is_absolute() else ROOT / out
        clean = out.with_name(out.stem.replace("_all", "") + "_clean.jsonl")
        rows = run_gated(idents, out, clean, args.backend, args.num_predict,
                         args.seed, regen=not args.no_regen)
        # Summarise over the WHOLE ledger, not just this invocation's rows, so a
        # resumed run reports the run's totals rather than the tail.
        allrows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        print_summary(allrows, out, clean)
        return

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
