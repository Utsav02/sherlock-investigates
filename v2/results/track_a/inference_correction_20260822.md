# Track A inference correction — participant dyads

**Date:** 2026-08-22

**Status:** supersedes the `participant` confidence intervals in earlier Track A
JSON and writeups. Point predictions and point estimates are unchanged.

## Error

Earlier Track A runs bootstrapped games by interrogator and by human witness
separately, then labelled the wider marginal interval `participant`. That is not
a crossed-cluster method. It also loses participant identity across roles even
though many people appear as both interrogator and human witness.

The label "conservative stand-in" is withdrawn. Choosing the wider of two
marginal intervals has no general joint-coverage guarantee.

## Replacement

For additive per-game outcomes, the corrected code uses a dyadic
participant-cluster sandwich interval. Each game belongs to both endpoint
participants, and the same participant ID is retained across roles. If `u_g` is
the centered per-game contribution, the covariance meat is:

```text
sum over participants p of (sum of u_g for games incident to p)^2
    minus
sum over games g of u_g^2
```

This covers accuracy, game Brier, dialogue-average balanced accuracy, dialogue
Brier, dialogue log loss, and paired accuracy differences. For non-additive
metrics such as AUROC, the connected-component bootstrap is the reported
interval. Component intervals are also retained as a sensitivity analysis for
the additive outcomes; there are few components and one is large, so those
intervals can be much wider.

Both methods condition on the already fitted cross-fold predictions. They do not
include the additional variation from drawing a new training corpus and refitting
the detector, so they must not be read as unconditional algorithm-performance
intervals.

## Corrected headline intervals

All runs use the same 851 train+development games after the frozen empty-witness
policy. The final test split remains untouched.

| estimator / cut | point accuracy | dyadic participant 95% | component bootstrap 95% |
|---|---:|---:|---:|
| A0 TF-IDF, people-only | 0.9600 | [0.9497, 0.9704] | [0.9553, 0.9705] |
| A2 frozen representation, people-only | 0.8660 | [0.8436, 0.8885] | [0.8467, 0.8757] |
| A0 minimal→persona | 0.6038 | [0.5538, 0.6537] | [0.5575, 0.6695] |
| A0 persona→minimal | 0.4953 | [0.4401, 0.5505] | [0.4435, 0.5942] |
| A2 minimal→persona | 0.4929 | [0.4362, 0.5496] | [0.4351, 0.6195] |
| A2 persona→minimal | 0.4483 | [0.3780, 0.5186] | [0.2458, 0.5728] |

The component intervals show how unstable the smaller persona cuts are across
independent graph blocks. They do not rescue persona transfer: neither A2 cut
shows useful discrimination, while one A0 direction is modestly above chance and
the reverse is not.

## What changes and what does not

**Changes:** every earlier participant interval and any confidence attached to
its "wider marginal = conservative" rationale.

**Does not change:** the recorded point predictions, within-configuration
performance, side-recognition rates, or the narrow conclusion that both tested
in-corpus estimators fail to transfer reliably in both directions across the two
persona prompt families.

Corrected machine-readable outputs:

- `a0_baselines_20260822_202851_inference-correction-20260822.json`
- `a0_rung4_loso_20260822_204404_inference-correction-20260822.json`
- `a2_frozen_rep_20260822_203719.json`
