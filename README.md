# Sherlock Investigates

A closed methodological failure study about training and evaluating a
language-model investigator. The original goal was an investigator that asks
useful questions and updates calibrated human-versus-AI beliefs. The project
stopped on 2026-08-29 before producing a trained adaptive investigator.

## Start here

- **Primary artifact:** [Final research report](docs/FINAL_RESEARCH_REPORT.md)
- [Artifact manifest](docs/ARTIFACT_MANIFEST.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Current status and stop boundary](STATUS.md)
- [Cleanup inventory](docs/CLEANUP_INVENTORY.md) and [deletion manifest](docs/DELETION_MANIFEST.md)
- Principal results: [V1 correction](results/analysis/inference_correction_20260822.md), [Track A correction](v2/results/track_a/inference_correction_20260822.md), [external bridge](v2/results/bridge/README.md), and [D0 Gate 2A](v2/results/d0_gate2a/README.md)

## Strongest supported conclusion

The project repeatedly obtained numerically positive results from measurements
or gates that were narrower than the motivating construct. Raw-prose QLoRA
produced no verified investigative reasoning shift; Track A found real passive
identity signal with severe prompt dependence; an external score transferred
only after nested calibration learned an inverse relationship; and Gate 2A
formally passed while selecting one fixed question sequence per scenario family.

Gate 2A remains a historical formal PASS for family-specific oracle
prioritization and exact ledger mechanics. It did not demonstrate
response-conditioned adaptation. Gate 2B was never frozen or run, D0 fine-tuning
did not occur, and the Track A final test split remained untouched.

## Safe verification

The retained CPU suite and syntax check require no network, model download, GPU,
or protected source data:

```bash
make lint
make test
venv/bin/python v2/scripts/d0_gate2a.py validate
```

See the [reproducibility guide](docs/REPRODUCIBILITY.md) before regenerating
results. It provides isolated commands that do not modify frozen artifacts.

## Repository map

```text
docs/                         Final report, artifact manifest, and reproduction guide
results/analysis/             V1 results, chronology, and authoritative correction
results/pilot/                Historical conversation shakedowns
scripts/                      V1 data, training, evaluation, and conversation code
configs/ and notebooks/       Historical QLoRA specifications and Kaggle drivers
v2/results/stage_a/           Literature, source, schema, and precision audits
v2/results/track_a/           Real-passive results and corrected inference
v2/results/bridge/            Pinned external-detector bridge
v2/results/d0_gate2a/         Frozen 16,384 sampled-policy-trajectory benchmark
v2/scripts/                   V2 reproduction and analysis code
```

## Authority and provenance

Later corrections and decisions control when they conflict with older files. In
particular:

- Historical V1 pooled repeated-measures p-values are withdrawn.
- Corrected Track A dyadic participant and component intervals replace earlier
  participant intervals.
- A2 was a probabilistic head evaluated for calibration, not a calibrated head.
- The claim that the corpus imposed a universal persona-transfer ceiling was
  withdrawn.
- Existing Gate 2B drafts are historical unexecuted feasibility work, not an
  active plan or gate outcome.

The V1 and V2 decision records remain in [CLAUDE.md](CLAUDE.md) and
[v2/DECISIONS.md](v2/DECISIONS.md). The frozen Gate 2A protocol and result
artifacts remain byte-identical.

## Project boundary

Do not resume the active-investigation extension from the deepest draft. No Gate
2B work, counterfactual-fork benchmark, D0 SFT, GPU run, human-data collection,
or Track A test access is authorized. Any future revival requires a new explicit
owner decision anchored to the original real-active objective.
