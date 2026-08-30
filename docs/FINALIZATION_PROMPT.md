# Sherlock Investigates finalization prompt

Copy the prompt below into a new Codex task rooted at the
`sherlock-investigates` repository.

---
Work in `/Users/utsavsingh/Desktop/Post-Uni/Projects/sherlock-investigates`.

## Objective

Finish the stopped project as a coherent methodological failure study. Produce
a tangible research report and a clean, understandable, reproducible repository
from the experiments that were actually run.

The report is the primary deliverable. Repository cleanup supports that
deliverable; it must not become a substitute for it.

Do not revive the active-investigation extension. Do not design or run Gate 2B,
create a counterfactual-fork benchmark, fine-tune a model, use a GPU, collect
human data, access the frozen Track A test split, or reinterpret the historical
Gate 2A PASS.

## Objective-alignment requirement

Before editing, write a short objective trace in your first progress update:

1. Original goal: train and evaluate an investigator that asks useful questions
   and updates calibrated human-versus-AI beliefs.
2. Evidence obtained: several V1 and v2 runs produced narrow positive, negative,
   corrected, and construct-invalid results.
3. Owner decision: stop the active extension and retain the methodological
   failure study.
4. Authorized deliverable: package the executed work and clean the repository
   without conducting new research.

If any proposed action would change the objective, construct, claim boundary,
success criterion, deliverable, or resource tier, stop and ask the owner to
choose before taking that action. Do not resume from the deepest existing draft.

## Read first

Read these files before making changes:

- `README.md`
- `STATUS.md`
- `CLAUDE.md`
- `v2/experiment_design.md`
- `v2/DECISIONS.md`
- `v2/D0_BRIDGE_VALIDITY_DECISION.md`
- `results/analysis/experiment_journey_20260814.md`
- `results/analysis/inference_correction_20260822.md`
- `v2/results/track_a/inference_correction_20260822.md`
- `v2/results/bridge/README.md`
- `v2/results/d0_gate2a/README.md`
- `v2/D0_GATE2A_PROTOCOL.md`
- `v2/results/d0_gate2a/result.json`
- `v2/results/d0_gate2a/inspection.json`

Treat later corrections and decisions as authoritative when an older document
conflicts with them. Do not silently rewrite historical frozen protocols or run
artifacts.

## Safety and working-tree rules

Start with a read-only audit:

```bash
git status --short
git diff --stat
git diff --name-only
git log --oneline --decorate -n 40
rg --files
```

The working tree contains unrelated work from other sessions. Preserve it. In
particular, do not modify, stage, delete, format, or overwrite these paths unless
the owner separately authorizes them:

- `scripts/data_prep/reverse_scenarios.py`
- `tests/test_reverse_scenarios.py`
- `brief.md`
- `data/sft/scenarios_pilot_all.jsonl`
- `data/sft/scenarios_pilot_clean.jsonl`
- `exploration_draft.md`
- `results/analysis/adjudicate_clashes.html`
- `results/analysis/label_stratified.html`

Preserve all frozen Gate 2A artifacts and hashes. Preserve append-only result and
decision records. Never use `git add .`, `git add -A`, destructive reset or
checkout commands, or broad recursive deletion. Do not commit or push unless the
owner explicitly requests it.

Before changing files, record SHA-256 hashes for frozen authoritative artifacts.
Verify the same hashes after cleanup. If a cleanup would alter an artifact needed
to reproduce a reported result, leave it unchanged and document why.

## Phase 1: repository audit

Produce a concise cleanup inventory before deleting or moving anything. Classify
each candidate as:

- active reproduction code;
- historical reproduction code;
- authoritative result or protocol;
- superseded documentation that must remain for provenance;
- generated or local-only material;
- proven dead code;
- ambiguous and requiring an owner decision.

Use evidence, not filename intuition. Check imports, command entry points,
Makefile targets, tests, documentation links, result provenance, and Git history.
A file is not dead merely because no current module imports it; historical run
reproduction may still require it.

Do not remove an ambiguous file. Put every proposed deletion in a deletion
manifest with its path, evidence that it is unused, recovery method, and effect
on reproducibility. Ask for approval before deleting material or moving large
groups of files.

## Phase 2: code and comment cleanup

Clean code only where the audit shows that doing so improves the retained
report's reproducibility or makes the repository easier to understand.

- Remove proven dead branches, unused imports, obsolete wrappers, and redundant
  helpers when tests and provenance show that removal is safe.
- Preserve historical runners required to reproduce reported results, even when
  they are no longer part of an active programme.
- Prefer small, reviewable edits over broad rewrites.
- Do not change algorithms, thresholds, random seeds, split membership,
  likelihood tables, reported measurements, or frozen protocol semantics.
- Do not convert cleanup into feature work.

Use comments sparingly:

- Add `IMPORTANT:` comments only for non-obvious scientific, safety, provenance,
  leakage, replay, or numerical invariants that a future maintainer could easily
  violate.
- Every source-code comment that you add or edit must occupy one physical line.
- Write comments as complete sentences with capitalization and punctuation.
- Explain why an invariant exists; do not narrate what readable code does.
- Remove stale, redundant, conversational, speculative, and commented-out code.
- If an explanation needs more than one line, put it in the relevant Markdown
  document and use a one-line code comment that links to that document.
- Do not compress a paragraph into an unreadable one-line comment.

Docstrings are not subject to the one-physical-line comment rule. Follow the
Google Python Style Guide: use a one-line docstring when it is sufficient, and
use a structured multiline docstring only for a public API, nontrivial function,
or non-obvious behavior. Do not add docstrings that repeat names or signatures.

Apply the official standards prospectively to files you edit; do not churn the
entire repository solely to restyle unchanged text:

- Google developer documentation style guide:
  `https://developers.google.com/style`
- Google Python Style Guide, comments and docstrings:
  `https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings`

For all edited prose and comments, use standard American English, active voice,
short direct sentences, consistent terminology, descriptive headings, and
specific claims. Avoid idioms, hype, rhetorical flourishes, vague pronouns,
anthropomorphism, and unnecessary future tense. Define abbreviations on first
use. Use one term for each concept.

## Phase 3: tangible research report

Create `docs/FINAL_RESEARCH_REPORT.md` as the main project artifact. Write it for
a technically literate reader who has not followed the repository history.

The report must stand alone and include:

1. Title, abstract, date, repository revision, and project status.
2. The original research question and why it mattered.
3. The planned causal or evidential chain from training to active investigation.
4. A methods overview for V1, Track A, the external-detector bridge, and D0 Gate
   2A.
5. An executed-run inventory that distinguishes training, inference, simulation,
   diagnostic, and literature/audit work.
6. A compact results table with exact values, sample units, uncertainty units,
   preregistration status, and the strongest supported interpretation.
7. A chronological narrative showing what each result changed.
8. Separate sections for genuine negative results, instrumentation failures,
   inference corrections, planning failures, and construct failures.
9. The Gate 2A formal PASS and its narrower post-run interpretation side by
   side, without retroactively changing either.
10. The 2026-08-29 owner decision to stop the active extension.
11. Limitations, non-claims, lessons for future LLM evaluation projects, and a
    concise conclusion.
12. Links to authoritative protocols, result files, code, and correction logs.

The report must state clearly:

- V1 performed real QLoRA training runs, but no D0 fine-tuning occurred.
- Historical V1 pooled repeated-measures p-values were withdrawn; do not reuse
  them as confirmatory evidence.
- Track A measured real-passive identity signal, not active questioning.
- The external detector's transferable signal appeared only after nested target
  calibration learned an inverse score relationship.
- Gate 2A integrated 16,384 trajectories and formally passed its frozen gate.
- Gate 2A selected one sequence per family and therefore demonstrated
  family-specific oracle prioritization, not response-conditioned adaptation.
- Gate 2B was never frozen or run and has no gate outcome.
- The Track A test split remained untouched.
- The project stopped before producing a trained adaptive investigator.

Do not manufacture a success narrative. Do not describe absence of evidence as
an impossibility result. Distinguish descriptive results from corrected
participant-clustered inference. Use exact numbers only when they can be traced
to an authoritative artifact.

Do not cite `experiment_journey_20260814.md` as final authority where later
correction documents supersede it. Use it for chronology, then reconcile every
claim against `STATUS.md`, `v2/DECISIONS.md`, and the correction artifacts.

## Phase 4: artifact and reproducibility package

Create `docs/ARTIFACT_MANIFEST.md` with one row per important artifact:

- path;
- stage and run date;
- artifact type;
- whether it is authoritative, corrected, superseded, or historical;
- sample or trajectory count;
- hash when available;
- reproduction command;
- required environment;
- claim supported;
- critical limitation.

Create `docs/REPRODUCIBILITY.md` containing:

- supported reproduction commands;
- expected runtime, hardware, and storage;
- required local or ignored inputs;
- which artifacts can be regenerated without network, model downloads, GPU, or
  protected data;
- which historical runs cannot be reproduced exactly and why;
- a warning that the frozen Track A test split must not be accessed;
- a minimal verification sequence for the retained report.

Update `README.md` so a new reader reaches the final report, artifact manifest,
reproducibility guide, current status, and principal result directories within
the first screen. Keep the README concise; link to detailed documents instead of
duplicating them.

If one or two figures materially clarify the results, generate them only from
existing authoritative artifacts. Record the source file, transformation, and
command. Do not create a new statistical analysis or use a figure to imply an
unsupported comparison.

## Phase 5: verification

Verify the cleanup and report with evidence:

1. Run the repository's relevant formatting and static checks.
2. Run the complete CPU test suite unless a documented environment limitation
   prevents it.
3. Run `git diff --check`.
4. Parse every edited JSON file.
5. Check local Markdown links in the new report and manifests.
6. Verify that frozen artifact hashes and row counts did not change.
7. Search for stale statements that authorize Gate 2B, D0 SFT, GPU work, human
   collection, or Track A test access.
8. Search edited source files for multiline block comments, commented-out code,
   and obsolete TODOs.
9. Confirm that unrelated working-tree paths are byte-identical to their initial
   state.

Do not claim that cleanup is complete if tests fail or frozen hashes change.
Report pre-existing failures separately from regressions introduced by cleanup.

## Final handoff

Lead with the tangible outcome. Report:

- the final report and manifest paths;
- the strongest supported project conclusion in two or three sentences;
- files removed, moved, or materially simplified, with recovery information;
- important invariants marked in code;
- test, link, JSON, hash, and diff-check evidence;
- unresolved ambiguities or pre-existing failures;
- all untouched unrelated working-tree changes; and
- whether anything was staged, committed, or pushed.

Do not propose another experiment as the default next step. The project is being
closed and packaged, not restarted.

---
