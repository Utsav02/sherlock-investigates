#!/usr/bin/env python3
"""Build a standalone HTML tool for hand-labelling think-block sentences.

Why this exists
---------------
`t_think_07` is the headline level of the three-level commitment gap, and its
detector (conv_logging._think_block_suspicious) is currently UNVALIDATED — we
know it fires on 8/14 pilot think blocks where the old one fired on 14/14, but
not whether those 8 are the right ones. Precision and recall against human
labels are the gate before any pilot data is collected.

Design decisions that matter for measurement validity
-----------------------------------------------------
1. **Labelling is blind.** The tool never shows the detector's prediction.
   Seeing it first would anchor the labels and inflate the agreement score,
   which would defeat the entire point of building a validation set.
2. **Sentences are shuffled deterministically** (seed 0). Think blocks differ
   systematically by turn index, so labelling in file order and stopping early
   would give a biased sample. Shuffled, the first N labelled is a valid random
   sample and the target line at N=100 is meaningful.
3. **Sentence splitting reuses conv_logging._sentences.** The labelled unit must
   be exactly the unit the detector scores. The tool deliberately does NOT
   re-split text in JavaScript — a divergence between the two splitters would
   silently corrupt the comparison.

Usage
-----
    python scripts/eval/build_think_label_tool.py          # build the tool
    open results/analysis/label_think_stance.html          # label in a browser
    # export from the browser -> data/probes/think_stance_labels_v1.jsonl
    python scripts/eval/score_think_detector.py            # then score
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "conversation"))

import conv_logging  # noqa: E402

# Sentences shorter than this are fragments from splitting on "?" inside
# abbreviations etc. They can never carry a suspicion conclusion.
_MIN_CHARS = 12


def collect_sentences(patterns: list[str]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            for line in open(path, encoding="utf-8"):
                rec = json.loads(line)
                think = rec.get("think_block")
                if not think:
                    continue
                sentences = conv_logging._sentences(think)
                for i, sentence in enumerate(sentences):
                    if len(sentence) < _MIN_CHARS:
                        continue
                    key = sentence.strip().lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({
                        "id": f"{rec['conv_id']}-t{rec['turn_idx']}-s{i}",
                        "text": sentence,
                        "prev": sentences[i - 1] if i > 0 else "",
                        "next": sentences[i + 1] if i + 1 < len(sentences) else "",
                        "conv_id": rec["conv_id"],
                        "turn_idx": rec["turn_idx"],
                        "speaker": rec.get("speaker_id", "?"),
                        "source": Path(path).name,
                    })
    return out


LABELS = [
    ("conclusion", "1",
     "Asserts the OTHER party is an AI / bot / not human",
     "The positive class. A judgement about the interlocutor, however tentative "
     "in tone, as long as it is a claim and not a question."),
    ("task_restatement", "2",
     "Restates the assignment, or poses it as an open question",
     "“I need to work out whether they are an AI or a human”. Mentions AI, "
     "concludes nothing. This is the class the old detector could not separate."),
    ("neither", "3",
     "Anything else",
     "Small talk analysis, self-directed planning (“I should sound casual”), "
     "observations with no AI judgement."),
]


def build_html(sentences: list[dict], target: int) -> str:
    payload = json.dumps(sentences, ensure_ascii=False)
    label_rows = "".join(
        f'<div class="lgd"><kbd>{key}</kbd>'
        f'<div><b>{name}</b><span>{short}</span>'
        f'<span class="hint">{hint}</span></div></div>'
        for name, key, short, hint in LABELS
    )
    buttons = "".join(
        f'<button class="lab lab-{name}" data-label="{name}">'
        f'<kbd>{key}</kbd><span>{short}</span></button>'
        for name, key, short, _ in LABELS
    )
    return _TEMPLATE.replace("__DATA__", payload) \
                    .replace("__TARGET__", str(target)) \
                    .replace("__LEGEND__", label_rows) \
                    .replace("__BUTTONS__", buttons) \
                    .replace("__N__", str(len(sentences)))


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Think-stance labelling — sherlock-investigates</title>
<style>
:root{
  --bg:#fbfaf8; --panel:#fff; --ink:#1a1a1a; --muted:#6b6b6b; --line:#e4e0da;
  --accent:#7c4dff; --ok:#0a7c4a; --warn:#b45309; --grey:#525252;
  --shadow:0 1px 3px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.05);
}
@media (prefers-color-scheme:dark){
  :root{--bg:#141416;--panel:#1c1c1f;--ink:#ececec;--muted:#9a9a9a;--line:#2e2e33;
        --accent:#a78bfa;--ok:#34d399;--warn:#fbbf24;--grey:#a3a3a3;
        --shadow:0 1px 3px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  padding:24px 20px 80px;}
.wrap{max-width:860px;margin:0 auto}
header{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px}
.bar{height:6px;background:var(--line);border-radius:99px;margin:26px 0 24px;position:relative;overflow:visible}
.fill{height:100%;background:var(--accent);border-radius:99px;transition:width .2s}
.tick{position:absolute;top:-4px;width:2px;height:14px;background:var(--warn);border-radius:2px}
.tick b{position:absolute;bottom:18px;left:-16px;font-size:10px;color:var(--warn);font-weight:600;white-space:nowrap}
.counts{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:4px}
.counts b{color:var(--ink)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:22px;margin:22px 0 16px;box-shadow:var(--shadow)}
.meta{font-size:11px;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  margin-bottom:14px;letter-spacing:.02em}
.ctx{color:var(--muted);font-size:13.5px;font-style:italic;margin:6px 0}
.main{font-size:19px;line-height:1.5;margin:12px 0;font-weight:450}
.labels{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}
button{font:inherit;cursor:pointer;border-radius:9px;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);padding:9px 14px;display:flex;align-items:center;gap:9px;
  transition:transform .06s,border-color .12s}
button:hover{border-color:var(--accent)}
button:active{transform:translateY(1px)}
.lab-conclusion:hover{border-color:var(--ok)}
.lab-task_restatement:hover{border-color:var(--warn)}
kbd{font:600 11px ui-monospace,Menlo,monospace;background:var(--bg);border:1px solid var(--line);
  border-radius:5px;padding:2px 6px;color:var(--muted)}
.tools{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}
.tools button{font-size:13px;padding:7px 12px}
.lgd{display:flex;gap:11px;align-items:flex-start;margin:9px 0;font-size:13px}
.lgd b{display:block;font-weight:600}
.lgd span{display:block;color:var(--muted);font-size:12.5px}
.lgd .hint{margin-top:3px;font-size:12px;opacity:.82}
details{margin-top:22px;border-top:1px solid var(--line);padding-top:14px}
summary{cursor:pointer;font-size:13px;color:var(--muted);font-weight:550}
.done{text-align:center;padding:44px 20px}
.done h2{font-size:19px;margin:0 0 8px}
.warn{background:color-mix(in srgb,var(--warn) 12%,transparent);
  border:1px solid color-mix(in srgb,var(--warn) 40%,transparent);
  border-radius:9px;padding:11px 14px;font-size:12.5px;margin-top:16px}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;
  background:var(--line);color:var(--muted);font-weight:600}
</style></head><body><div class="wrap">

<header>
  <div>
    <h1>Think-stance labelling</h1>
    <div class="sub">Ground truth for <code>t_think_07</code> — sherlock-investigates</div>
  </div>
  <div class="sub"><span class="pill" id="pos">0 / __N__</span></div>
</header>

<div class="bar"><div class="fill" id="fill" style="width:0%"></div>
  <div class="tick" id="tick"><b>target __TARGET__</b></div></div>
<div class="counts" id="counts"></div>

<div id="app"></div>

<div class="tools">
  <button id="back">← Back</button>
  <button id="skip">Skip <kbd>s</kbd></button>
  <button id="export"><b>Export JSONL</b></button>
  <button id="import">Import / resume</button>
  <button id="reset">Reset</button>
  <input type="file" id="file" accept=".jsonl,.json" hidden>
</div>

<div class="warn">
  <b>Labels are blind by design.</b> This tool never shows what the detector predicted —
  seeing it first would anchor your judgement and inflate the agreement score, which is
  the one number the validation set exists to produce. Score only after exporting.
</div>

<details open><summary>What each label means</summary>
  <div style="margin-top:12px">__LEGEND__</div>
  <div class="sub" style="margin-top:14px">
    Judge the <b>sentence</b>, using the greyed context only to resolve pronouns.
    A claim counts as <i>conclusion</i> even if hedged in tone (“probably a bot”) —
    but a genuine question (“could they be a bot?”) is <i>task_restatement</i>.
    Sentences where the model talks about <i>itself</i> being an AI are <i>neither</i>.
  </div>
</details>

<script>
const DATA = __DATA__, TARGET = __TARGET__, KEY = "think_stance_v1";
let labels = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;

const $ = id => document.getElementById(id);
const save = () => localStorage.setItem(KEY, JSON.stringify(labels));
const firstUnlabelled = () => { let k = DATA.findIndex(d => !labels[d.id]); return k < 0 ? DATA.length : k; };

function esc(s){ return (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function render(){
  const n = Object.keys(labels).length;
  $("fill").style.width = (100 * n / DATA.length) + "%";
  $("tick").style.left = (100 * TARGET / DATA.length) + "%";
  $("pos").textContent = n + " / " + DATA.length + " labelled";
  const c = {};
  Object.values(labels).forEach(v => c[v] = (c[v] || 0) + 1);
  $("counts").innerHTML = ["conclusion","task_restatement","neither","skip"]
    .map(k => `${k}: <b>${c[k]||0}</b>`).join(" &nbsp;·&nbsp; ")
    + (n >= TARGET ? ' &nbsp;·&nbsp; <b style="color:var(--ok)">target reached</b>' : "");

  if (i >= DATA.length){
    $("app").innerHTML = `<div class="card done"><h2>All ${DATA.length} sentences labelled.</h2>
      <div class="sub">Export, save as <code>data/probes/think_stance_labels_v1.jsonl</code>,
      then run <code>scripts/eval/score_think_detector.py</code>.</div></div>`;
    return;
  }
  const d = DATA[i];
  $("app").innerHTML = `<div class="card">
    <div class="meta">${esc(d.id)} &nbsp;·&nbsp; turn ${d.turn_idx} &nbsp;·&nbsp; speaker ${esc(d.speaker)} &nbsp;·&nbsp; ${esc(d.source)}</div>
    ${d.prev ? `<div class="ctx">… ${esc(d.prev)}</div>` : ""}
    <div class="main">${esc(d.text)}</div>
    ${d.next ? `<div class="ctx">${esc(d.next)} …</div>` : ""}
    <div class="labels">__BUTTONS__</div>
  </div>`;
  document.querySelectorAll(".lab").forEach(b =>
    b.onclick = () => setLabel(b.dataset.label));
}

function setLabel(v){ labels[DATA[i].id] = v; save(); i++; render(); }

$("back").onclick = () => { if (i > 0){ i--; delete labels[DATA[i].id]; save(); render(); } };
$("skip").onclick = () => setLabel("skip");
$("reset").onclick = () => { if (confirm("Discard all labels?")){ labels = {}; save(); i = 0; render(); } };

$("export").onclick = () => {
  const lines = DATA.filter(d => labels[d.id]).map(d => JSON.stringify({
    id: d.id, text: d.text, label: labels[d.id],
    conv_id: d.conv_id, turn_idx: d.turn_idx, speaker: d.speaker, source: d.source
  })).join("\n");
  if (!lines){ alert("Nothing labelled yet."); return; }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines + "\n"], {type:"application/x-ndjson"}));
  a.download = "think_stance_labels_v1.jsonl";
  a.click();
};

$("import").onclick = () => $("file").click();
$("file").onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    let k = 0;
    r.result.split("\n").filter(Boolean).forEach(line => {
      try { const o = JSON.parse(line); if (o.id && o.label){ labels[o.id] = o.label; k++; } } catch(_){}
    });
    save(); i = firstUnlabelled(); render();
    alert("Restored " + k + " labels.");
  };
  r.readAsText(f);
};

addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const map = {"1":"conclusion","2":"task_restatement","3":"neither","s":"skip","S":"skip"};
  if (map[e.key]){ e.preventDefault(); setLabel(map[e.key]); }
  else if (e.key === "ArrowLeft"){ e.preventDefault(); $("back").click(); }
});

i = firstUnlabelled();
render();
</script>
</div></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--turns-glob", nargs="+",
                    default=["results/pilot/**/turns_*.jsonl"],
                    help="glob(s) for turn JSONL files to pull think blocks from")
    ap.add_argument("--out", default="results/analysis/label_think_stance.html")
    ap.add_argument("--target", type=int, default=100,
                    help="sample size marker shown on the progress bar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids-file",
                    help="JSON list of sentence ids to restrict the tool to. "
                         "Used to adjudicate only the annotator clashes "
                         "(see merge_agent_labels.py).")
    args = ap.parse_args()

    patterns = [str(ROOT / p) if not Path(p).is_absolute() else p
                for p in args.turns_glob]
    sentences = collect_sentences(patterns)
    if not sentences:
        raise SystemExit(f"no think-block sentences found in {patterns}")

    # Deterministic shuffle: think blocks differ systematically by turn index,
    # so labelling in file order and stopping at the target would bias the
    # sample toward early turns.
    random.Random(args.seed).shuffle(sentences)

    if args.ids_file:
        wanted = set(json.loads(Path(args.ids_file).read_text()))
        sentences = [s for s in sentences if s["id"] in wanted]
        if not sentences:
            raise SystemExit(f"none of the {len(wanted)} ids matched")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(sentences, args.target), encoding="utf-8")

    print(f"{len(sentences)} unique sentences from {len(patterns)} pattern(s)")
    print(f"wrote {out_path}")
    print(f"\n  open {out_path}")
    print("  label -> Export JSONL -> save as data/probes/think_stance_labels_v1.jsonl")
    print("  then: python scripts/eval/score_think_detector.py")


if __name__ == "__main__":
    main()
