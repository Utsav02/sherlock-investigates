#!/usr/bin/env python3
"""Build a read-only HTML page for a human to scan the LLM judge's KEEPERS.

Why this exists
---------------
The LLM judge (`generate_traces.judge_answer`) is the keeper GATE for the SFT
set and is itself UNAUDITED. Judge and teacher are the same model family, so a
shared blind spot could wave a wrong identity into the training data — the
`t_think_07` failure mode (an instrument nobody scored, precision 0.185).

For a PILOT the only failure that damages training DATA is a **false accept**:
the judge says YES to an answer that does not identify a person of the ground
truth. False rejects only cost yield, which the 2x-scenarios plan absorbs. Every
false accept lives in the keepers, so scanning the keepers is the complete check
for the one error that matters, and it costs a human ~5 minutes.

This is deliberately NOT the rigorous 30-label precision/recall audit (which
needs rejects too, and a blind protocol). That is deferred to just before the
final scaled run.

Design decisions that matter
----------------------------
1. **Read-only.** No buttons, no localStorage, no export. The human reads and
   calls out row numbers in chat. Building a labelling UI for 14 rows would cost
   more than the scan.
2. **Four fields only** — scenario_prompt, ground_truth, ANSWER, judge REASON.
   The `<think>` block is omitted on purpose: whether the *reasoning* is genuine
   was settled by reading traces on 2026-08-15; this scan asks the narrower
   question "does the ANSWER identify a person of the GROUND TRUTH?", and adding
   ~1.1K chars of reasoning per row would turn a 5-minute scan into a 30-minute
   one.
3. **The keyword `matched` flag is NOT shown.** It disagrees with the judge on
   3/18 rows and was measured wrong in both directions; showing it would anchor
   the reader toward the lexical verdict the judge exists to replace.
4. **All keepers on one scrollable page**, in file order. A card-at-a-time flow
   optimises for careful labelling; a scan optimises for throughput and for
   noticing a pattern across rows.

Usage
-----
    python scripts/eval/build_keeper_scan.py
    open results/analysis/keeper_scan.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_keepers(path: Path) -> tuple[list[dict], int]:
    """Return (keeper rows, total rows). A keeper is a row the judge accepted."""
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    keepers = [r for r in rows if r.get("judge") is True]
    return keepers, len(rows)


def build_html(keepers: list[dict], total: int, source: str) -> str:
    cards = []
    for n, r in enumerate(keepers, 1):
        cards.append(_CARD.format(
            n=n,
            sid=html.escape(str(r.get("scenario_id", "?"))),
            prompt=html.escape(r.get("scenario_prompt", "")),
            gt=html.escape(r.get("ground_truth", "")),
            answer=html.escape(r.get("answer", "")),
            reason=html.escape(r.get("judge_reason", "") or "(no reason given)"),
        ))
    return (_TEMPLATE
            .replace("__CARDS__", "\n".join(cards))
            .replace("__N__", str(len(keepers)))
            .replace("__TOTAL__", str(total))
            .replace("__SOURCE__", html.escape(source)))


_CARD = """
<article class="card">
  <div class="hd"><span class="num">{n}</span>
    <span class="meta">scenario_id {sid}</span></div>
  <div class="row"><div class="lbl">Scenario</div>
    <div class="val scen">{prompt}</div></div>
  <div class="row"><div class="lbl gt">Ground truth</div>
    <div class="val gtv">{gt}</div></div>
  <div class="row"><div class="lbl ans">Teacher's answer</div>
    <div class="val ansv">{answer}</div></div>
  <div class="row"><div class="lbl jr">Judge's reason</div>
    <div class="val jrv">{reason}</div></div>
</article>
"""


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Keeper scan — judge false-accept check</title>
<style>
:root{
  --bg:#fbfaf8; --panel:#fff; --ink:#1a1a1a; --muted:#6b6b6b; --line:#e4e0da;
  --accent:#7c4dff; --ok:#0a7c4a; --warn:#b45309; --ans:#1d4ed8;
  --shadow:0 1px 3px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.05);
}
@media (prefers-color-scheme:dark){
  :root{--bg:#141416;--panel:#1c1c1f;--ink:#ececec;--muted:#9a9a9a;--line:#2e2e33;
        --accent:#a78bfa;--ok:#34d399;--warn:#fbbf24;--ans:#7dabff;
        --shadow:0 1px 3px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  padding:24px 20px 80px;}
.wrap{max-width:880px;margin:0 auto}
header{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px}
.pill{display:inline-block;padding:1px 9px;border-radius:99px;font-size:11px;
  background:var(--line);color:var(--muted);font-weight:600}
.task{background:color-mix(in srgb,var(--accent) 9%,transparent);
  border:1px solid color-mix(in srgb,var(--accent) 35%,transparent);
  border-radius:11px;padding:14px 16px;margin:20px 0 6px;font-size:13.5px}
.task b{color:var(--ink)}
.task .q{font-size:15px;font-weight:600;margin:8px 0 6px}
.note{color:var(--muted);font-size:12.5px;margin-top:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:20px 22px;margin:18px 0;box-shadow:var(--shadow)}
.hd{display:flex;align-items:center;gap:11px;margin-bottom:14px;
  padding-bottom:11px;border-bottom:1px solid var(--line)}
.num{display:inline-flex;align-items:center;justify-content:center;
  min-width:26px;height:26px;border-radius:8px;background:var(--accent);
  color:#fff;font-weight:700;font-size:13px;padding:0 7px}
.meta{font-size:11px;color:var(--muted);
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.02em}
.row{display:grid;grid-template-columns:118px 1fr;gap:14px;margin:11px 0;
  align-items:start}
@media (max-width:620px){.row{grid-template-columns:1fr;gap:3px}}
.lbl{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);padding-top:3px}
.lbl.gt{color:var(--ok)} .lbl.ans{color:var(--ans)} .lbl.jr{color:var(--muted)}
.val{font-size:14.5px}
.scen{color:var(--muted)}
.gtv{font-weight:650;color:var(--ok);font-size:15px}
.ansv{font-weight:500;color:var(--ink);font-size:15.5px;line-height:1.5}
.jrv{color:var(--muted);font-size:13px;font-style:italic}
footer{margin-top:34px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12.5px}
</style></head><body><div class="wrap">

<header>
  <div>
    <h1>Keeper scan — judge false-accept check</h1>
    <div class="sub">PILOT scan of the LLM judge's accepts — sherlock-investigates</div>
  </div>
  <div class="sub"><span class="pill">__N__ keepers of __TOTAL__</span></div>
</header>

<div class="task">
  <div class="q">For each row, ask one question:</div>
  <b>Does the teacher's ANSWER correctly identify a person of the GROUND TRUTH?</b>
  <div class="note">
    A coarser answer is fine (“a soldier” for “a retired sergeant of the Royal
    Marines”). A <b>contradictory</b> answer is a false accept (“professional
    boxer” for “amateur boxer”). Call out the row numbers you'd flag and why —
    nothing here records your judgement, this page is read-only.
  </div>
  <div class="note">
    This is a <b>PILOT</b> keeper scan, not the rigorous 30-label precision/recall
    audit. It checks only for false accepts — the one error that puts bad data
    into the SFT set. False rejects (yield loss) are out of scope by design.
  </div>
</div>

__CARDS__

<footer>
  Source: <code>__SOURCE__</code> · keeper = <code>judge == true</code> ·
  the keyword <code>matched</code> flag is deliberately not shown (it disagrees
  with the judge on 3/18 rows and was wrong in both directions).
</footer>

</div></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traces", default="data/sft/traces_claude_validation_judged.jsonl")
    ap.add_argument("--out", default="results/analysis/keeper_scan.html")
    args = ap.parse_args()

    src = Path(args.traces)
    src = src if src.is_absolute() else ROOT / src
    if not src.exists():
        raise SystemExit(f"no such traces file: {src}")

    keepers, total = load_keepers(src)
    if not keepers:
        raise SystemExit(f"no keepers (judge == true) in {src}")

    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        build_html(keepers, total, str(src.relative_to(ROOT))), encoding="utf-8")

    print(f"{len(keepers)} keepers of {total} rows -> {out_path}")
    print(f"\n  open {out_path}")
    print("  scan (~5 min), then call out any row where the ANSWER does not "
          "identify a person of the GROUND TRUTH")


if __name__ == "__main__":
    main()
