# Repository cleanup inventory

**Audit date:** 2026-08-29
**Base research revision:** `75a751e000387da9e5238789d810451688b0beeb` on `track-a-ablation`
**Closeout package revision:** The archival commit containing this file; see Git history for its hash.
**Decision:** Preserve all existing files. The audit found no deletion that is both useful to the retained report and proven safe for historical reproduction.

## Scope and method

This inventory supports the methodological failure study. It does not reopen the active-investigation extension. The audit checked the working tree, recent Git history, tracked-file status, Makefile entry points, Python imports, tests, documentation links, result provenance, artifact hashes, and row counts. A lack of current imports was not treated as proof that a historical runner was dead.

## Classification

| Candidate | Classification | Evidence | Cleanup decision |
|---|---|---|---|
| `v2/scripts/build_canonical.py`, `track_a_a0.py`, `track_a_rung2.py`, `track_a_rung4.py`, `track_a_a2.py` | Active reproduction code | Makefile targets, direct test imports, Track A writeup commands, and corrected JSON provenance name these scripts. | Keep unchanged. |
| `v2/scripts/track_a_bridge.py`, `scripts/setup_bridge_env.sh`, `scripts/doctor_bridge_env.sh` | Active reproduction code | The bridge README and Makefile provide supported score and analysis commands; bridge tests import the runner. | Keep unchanged. |
| `v2/scripts/d0_gate2a.py` and `v2/configs/d0_gate2a_v1.json` | Active reproduction code | Makefile targets, focused tests, the frozen protocol, and the result README identify the exact runner and config. | Keep unchanged. |
| Root `scripts/training/`, `scripts/eval/`, `scripts/conversation/`, training configs, and Kaggle notebooks | Historical reproduction code | V1 result JSON, writeups, tests, and the decision log identify the exact training, inference, and diagnostic paths. Several artifacts can be reconstructed only with these files. | Keep unchanged. |
| Root data-preparation scripts and tracked corpora | Historical reproduction code and input data | Makefile targets, manifests, configs, and V1 provenance depend on them. | Keep unchanged. |
| `v2/D0_GATE2A_PROTOCOL.md` and `v2/results/d0_gate2a/` | Authoritative protocol and results | Protocol was frozen in commit `77ee76e`; 16,384 trajectory rows and published hashes match. | Keep byte-identical. |
| Corrected Track A JSON and `v2/results/track_a/inference_correction_20260822.md` | Authoritative corrected results | Later correction replaces the earlier marginal-bootstrap intervals while retaining point predictions. | Keep byte-identical. |
| `results/analysis/inference_correction_20260822.md` | Authoritative correction | Withdraws V1 pooled repeated-measures p-values and narrows causal language. | Keep byte-identical. |
| `v2/results/bridge/` and `v2/BRIDGE_PROTOCOL.md` | Authoritative protocol and results | The bridge README records the frozen detector revision, nested calibration, corrected intervals, and raw-score hash. | Keep byte-identical. |
| Earlier V1 and Track A writeups and JSON files | Superseded documentation or historical artifacts that must remain | Correction documents explicitly preserve them as the audit trail and supersede only specified claims or intervals. | Keep; label status in the report and manifest. |
| `results/analysis/experiment_journey_20260814.md` | Superseded chronology that must remain | Useful for chronology, but later V1 inference correction withdraws its pooled Fisher/Wilson claims and causal attribution. | Keep; do not cite as final authority. |
| `v2/D0_GATE2B_PROTOCOL.md`, `v2/D0_GATE2B_ANALYTICAL_NOTE.md`, `v2/D0_GATE2B_PREFLIGHT.md`, `v2/configs/d0_gate2b_v1.json` | Historical unexecuted feasibility work | The stop decision explicitly requires preservation. The files are untracked, contain no runner or result, and state that Gate 2B was not frozen or run. | Keep untouched; do not execute or present as a gate outcome. |
| `scripts/data_prep/reverse_scenarios.py`, `tests/test_reverse_scenarios.py`, `brief.md`, two pilot scenario JSONL files, `exploration_draft.md`, and two HTML files named in the owner instructions | Unrelated in-flight work | They were modified or untracked before this task and are protected explicitly by the owner. Before and after SHA-256 values match in the preservation table below. | Keep byte-identical. |
| `docs/notes.docx`, `docs/PROJECT_LOG.md`, `docs/COWORK.md` | Ambiguous | Tracked files have no demonstrated effect on current reproduction, but history and possible human-authored content make deletion unsafe without owner review. | Keep. Owner decision required before removal. |
| `.gitkeep` placeholders in otherwise populated result directories | Ambiguous | They are harmless and tracked; deleting them offers no report or reproducibility benefit. | Keep. |
| Generated label tools and historical HTML under `results/analysis/` | Generated or historical local material | Some are tracked outputs of label/adjudication tooling and provide provenance for the instrument audit; two untracked files are protected unrelated work. | Keep. |
| Proven dead code | None established | Every plausible candidate was referenced by a historical run, test, command, decision, or provenance record, or remained ambiguous. | No removal. |

## Protected-file preservation

These values were computed at the read-only audit boundary and after closeout verification in the same session.

| Protected path | Before SHA-256 | After SHA-256 | Result |
|---|---|---|---|
| `scripts/data_prep/reverse_scenarios.py` | `8191f1cc14fdd3ea506cca5aee6a99e9c94633708ceb79883f4582b12e33329f` | `8191f1cc14fdd3ea506cca5aee6a99e9c94633708ceb79883f4582b12e33329f` | Byte-identical. |
| `tests/test_reverse_scenarios.py` | `1b1d7779ec96968198c7b8b463c3eea3a71a28109f6e9dafb0408ce551192211` | `1b1d7779ec96968198c7b8b463c3eea3a71a28109f6e9dafb0408ce551192211` | Byte-identical. |
| `brief.md` | `55e32ac6c23e20072e6257cab7b23975e1742616b27fd4a5917db62fc11a52fa` | `55e32ac6c23e20072e6257cab7b23975e1742616b27fd4a5917db62fc11a52fa` | Byte-identical. |
| `data/sft/scenarios_pilot_all.jsonl` | `8d8392c3f206052585edeac720c9db153067c6e3ae516959e55cde43b8174e60` | `8d8392c3f206052585edeac720c9db153067c6e3ae516959e55cde43b8174e60` | Byte-identical. |
| `data/sft/scenarios_pilot_clean.jsonl` | `23eee979cb39d2b1d168ac4525ee6c981d7b267329bdb2248c37e5e620507a57` | `23eee979cb39d2b1d168ac4525ee6c981d7b267329bdb2248c37e5e620507a57` | Byte-identical. |
| `exploration_draft.md` | `11aee3eff121e5d6786db86486b991459c662b80880cba50f932f3f1dd849f72` | `11aee3eff121e5d6786db86486b991459c662b80880cba50f932f3f1dd849f72` | Byte-identical. |
| `results/analysis/adjudicate_clashes.html` | `00f2c9fbf1682662447e40cad90db7bddef5ac4a2bb8e59f4ef334a74f2e8f27` | `00f2c9fbf1682662447e40cad90db7bddef5ac4a2bb8e59f4ef334a74f2e8f27` | Byte-identical. |
| `results/analysis/label_stratified.html` | `5cc6a0a7154592021e5f5ec8f77fa977d07ecea308fe1cd4e446e217be572f5d` | `5cc6a0a7154592021e5f5ec8f77fa977d07ecea308fe1cd4e446e217be572f5d` | Byte-identical. |

## Repository-level findings

- The repository mixes current reproduction code, historical reproduction code, immutable results, and local in-flight work. Removing files based only on present-day imports would damage the historical record.
- The Makefile remains useful but contains historical GPU and data-generation targets. The new documentation marks those commands as historical and unsupported for project closeout rather than deleting them.
- The untracked Gate 2B draft set is a provenance record, not an implementation backlog.
- No source-code edit is necessary to make the retained report reproducible. The highest-value cleanup is documentation that identifies authority, corrections, inputs, and safe commands.

## Deletion decision

No files are proposed for deletion or movement. See [`DELETION_MANIFEST.md`](DELETION_MANIFEST.md).
