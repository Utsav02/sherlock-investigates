# Artifact manifest

**Report date:** 2026-08-29
**Base research revision:** `75a751e000387da9e5238789d810451688b0beeb`
**Closeout package revision:** The archival commit containing this file; see Git history for its hash.
**Status vocabulary:** authoritative = current source for the stated claim; corrected = later artifact that replaces specified inference or wording; superseded = retained but not current authority; historical = executed or planned record retained for provenance.

Commands are relative to the repository root. Historical GPU commands are documentation, not authorization to rerun them. The project is closed; see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for safe verification.

## V1 design, training, and evaluation

| Path | Stage and run date | Type and status | Count | SHA-256 | Reproduction command | Environment | Claim supported | Critical limitation |
|---|---|---|---:|---|---|---|---|---|
| `EXPERIMENT_DESIGN.md` | V1 design, 2026-06 to 2026-08 | Protocol; historical | Not applicable | `a5a63e9f…a8bf` | Not generated | Markdown | Original commitment-gap and training plan | Later decisions and corrections supersede active instructions. |
| `configs/kaggle_t4_validation.yaml` | Stage 0, 2026-07-28 | Training config; historical | 30 steps | `ec01c4f3…2217` | `python scripts/training/train_lora.py --config configs/kaggle_t4_validation.yaml` | Kaggle T4; historical GPU environment | Real QLoRA plumbing run | Weights are external/private; no D0 training occurred. |
| `configs/kaggle_t4_dosecurve.yaml` | Full-canon rank-32, 2026-08-06 to 2026-08-08 | Training config; historical | Up to 103 steps | `32a570cb…98d` | `python notebooks/kaggle_t4_dosecurve.py` | Kaggle T4, Unsloth/CUDA | Defines the full-canon checkpoint path | Several original weights and one original result file were lost. |
| `results/analysis/dose_curve_20260808_204827.json` | V1 closure curve, 2026-08-08 | Result; historical, inference superseded | 22 checkpoints × 8 fixed prompts | `5f566650…f268` | `python scripts/eval/dose_curve.py --help` | Historical GPU inference; JSON analysis is CPU-readable | Descriptive rank-32 closure path | Reconstructed from notebook output; pooled p-value and interval withdrawn. |
| `configs/kaggle_t4_confound_pilot103.yaml` | Pilot repeated trajectory, 2026-08-12 | Training config; historical | 110 steps | `778f3bc3…e138` | `python notebooks/kaggle_t4_confound.py` | Kaggle T4 | Defines the low-breadth training path | No independent seeds; causal factor attribution withdrawn. |
| `results/analysis/dose_curve_20260812_104622.json` | V1 closure curve, 2026-08-12 | Result; historical, inference superseded | 22 checkpoints × 8 fixed prompts | `14f52e5b…5f7b` | `python scripts/eval/dose_curve.py --help` | Historical GPU inference | Descriptive repeated-pilot closure path | Reconstructed tracked copy; repeated-measures inference invalid. |
| `results/analysis/confound_pilot103_vs_fullcanon.json` | V1 diagnostic, 2026-08-12 | Result; superseded inference | Four pooled contrasts | `a467a7dd…a083` | `python scripts/eval/confound_analysis.py --fullcanon results/analysis/dose_curve_20260808_204827.json --pilot results/analysis/dose_curve_20260812_104622.json` | CPU; current Python environment | Historical comparison arithmetic | Fisher tests and causal “steps not breadth” verdict are withdrawn. |
| `configs/kaggle_t4_lowrank_r8.yaml` | Rank-8 mitigation, 2026-08-14 | Training config; historical | 103 steps | `e07e325d…f940` | `python notebooks/kaggle_t4_lowrank.py` | Kaggle T4 | Defines the executed rank-8 QLoRA run | One training trajectory; not a general mitigation estimate. |
| `results/analysis/dose_curve_20260814_042400.json` | Rank-8 closure, 2026-08-14 | Result; historical, inference superseded | 21 checkpoints × 8 fixed prompts | `28ec8dee…fe20` | `python scripts/eval/dose_curve.py --help` | Historical GPU inference | Candidate format-preserving checkpoints | Reconstructed; pooled inference and `RESCUED` language withdrawn. |
| `results/analysis/effect_curve_20260814_065854.json` | Rank-8 perplexity, 2026-08-14 | Result; authoritative for descriptive PPL | Base, 21 checkpoints, final | `8300b1ad…edf3` | `python scripts/eval/effect_curve.py --help` | Historical GPU inference | Holmes and WikiText perplexity values | Proxy outcome; no sampling interval; weights are external/private. |
| `results/analysis/mitigation_lowrank_r8.json` | Rank-8 combined diagnostic, 2026-08-14 | Result; superseded verdict | 21 joined checkpoints | `c197fb13…70a4` | `python scripts/eval/mitigation_analysis.py --closure results/analysis/dose_curve_20260814_042400.json --effect results/analysis/effect_curve_20260814_065854.json` | CPU | Historical selection logic | Automated `RESCUED` verdict is withdrawn by later evidence. |
| `results/analysis/thinking_shift_20260814_171042_transcript.md` | Paired behavior, 2026-08-14 | Transcript; authoritative historical evidence | 30 paired prompts | `8c64eb4d…849d` | `python notebooks/kaggle_t4_thinking_shift.py` | Historical GPU inference | Direct base-versus-adapter reasoning comparison | Qualitative, unblinded, one checkpoint. |
| `results/analysis/thinking_shift_20260814_writeup.md` | Paired behavior, 2026-08-14 | Interpretation; authoritative historical | 30 paired prompts | `6503b507…a387` | Not separately generated | Markdown | No visible reasoning shift at step 50 | Does not prove no checkpoint or training method could shift behavior. |
| `results/pilot/gate2_n20/conversations_20260806_091441.jsonl` | Conversation shakedown, 2026-08-06 | Result; authoritative historical | 20 conversations | `9590fc97…a63` | `python scripts/conversation/run_pilot.py --help` | Local Ollama historical environment | 3/20 degeneracy and 2/20 accusation counts | Suspicion instrument invalid; not a commitment-gap result. |
| `results/pilot/gate2_n20/turns_20260806_091441.jsonl` | Conversation shakedown, 2026-08-06 | Result; authoritative historical | 439 turns | `1350cbc6…249` | Same as above | Local Ollama historical environment | Turn-level audit trail | Model/version availability and instrument failures limit exact rerun. |
| `results/analysis/inference_correction_20260822.md` | V1 correction, 2026-08-22 | Correction; authoritative | Not applicable | `7ab5d196…091c` | Not generated | Markdown | Withdraws pooled repeated-measures inference | Prompt-level covariance cannot be recovered from old aggregates. |

## V2 audit and Track A

| Path | Stage and run date | Type and status | Count | SHA-256 | Reproduction command | Environment | Claim supported | Critical limitation |
|---|---|---|---:|---|---|---|---|---|
| `v2/experiment_design.md` | V2.1 design, 2026-08-17 onward | Protocol; historical design authority | Not applicable | `7253abc3…fa8c` | Not generated | Markdown | Real-passive/synthetic-active decomposition | Later owner stop supersedes prospective work plan. |
| `v2/results/stage_a/literature_matrix.md` | Literature audit, 2026-08-18 | Audit; authoritative historical | 15 sources | `f5d49dd6…243a` | Manual documented review | Network required for source re-check | Literature coverage and external baselines | Some entries were abstract-level only. |
| `v2/results/stage_a/source_coverage.md` | Source audit, 2026-08-18 | Audit; authoritative historical | Candidate-source matrix | `8220c0db…30f8` | Manual documented review | Markdown | No source supplied all real-active requirements | Does not prove no qualifying source exists elsewhere. |
| `v2/results/track_a/a0_baselines_20260822_202851_inference-correction-20260822.json` | Track A correction, 2026-08-22 | Result; corrected and authoritative | 851 train+development games | `166884ec…4a` | `venv/bin/python v2/scripts/track_a_a0.py` | CPU; local ignored canonical data | A0 within-configuration point estimates and corrected intervals | Default command writes a new timestamped artifact; test split remains excluded. |
| `v2/results/track_a/a0_rung4_loso_20260822_204404_inference-correction-20260822.json` | Track A correction, 2026-08-22 | Result; corrected and authoritative | Persona/system held-out cells from 851-game pool | `5d3f7689…3acc` | `venv/bin/python v2/scripts/track_a_rung4.py` | CPU; local ignored canonical data | Corrected A0 persona-transfer inference | Within one corpus; few components; no final-test evaluation. |
| `v2/results/track_a/a2_frozen_rep_20260822_203719.json` | Track A correction, 2026-08-22 | Result; corrected and authoritative | 851-game people-only pool plus persona cells | `66907ef3…3bef` | `venv/bin/python v2/scripts/track_a_a2.py --embed-url http://127.0.0.1:51999` | CPU analysis plus local Qwen embedding server/cache | A2 point estimates and corrected intervals | No nested calibrator; one representation and head; requires local model/cache. |
| `v2/results/track_a/inference_correction_20260822.md` | Track A correction, 2026-08-22 | Correction; authoritative | Not applicable | `18f71f06…a097` | Not generated | Markdown | Defines dyadic and component interval authority | Conditional on fitted predictions; excludes refitting variation. |

## External bridge

| Path | Stage and run date | Type and status | Count | SHA-256 | Reproduction command | Environment | Claim supported | Critical limitation |
|---|---|---|---:|---|---|---|---|---|
| `v2/BRIDGE_PROTOCOL.md` | Bridge protocol, 2026-08-22 | Frozen protocol; authoritative | 850-game train+development scope | `1aca3591…d9ab` | Not generated | Markdown | Pre-registered nested calibration and gate | Does not establish real-active transfer. |
| `v2/results/bridge/openai_gpt2_detector_raw.jsonl` | Bridge scoring, 2026-08-22 | Raw result; authoritative | 1,700 unique dialogue scores | `2de37ddd…a7297c` | `.bridge-venv/bin/python -u v2/scripts/track_a_bridge.py score --resume` | Python 3.12.5; pinned bridge lock; model download/cache | Raw immutable detector scores and input hashes | Exact rerun needs model access/cache and ignored canonical text. |
| `v2/results/bridge/openai_gpt2_detector_bridge.json` | Bridge analysis, 2026-08-22 | Result; authoritative | 850 paired games | `9290325a…32c` | `venv/bin/python v2/scripts/track_a_bridge.py analyze --scores v2/results/bridge/openai_gpt2_detector_raw.jsonl --output /tmp/bridge.json --bootstrap 1000` | CPU; local canonical data | Frozen bridge PASS after inverse nested calibration | Only eight components; weak signal; raw detector direction fails. |
| `v2/results/bridge/openai_gpt2_detector_run_state.json` | Bridge scoring, 2026-08-22 | Run state; authoritative historical | 1,700 rows complete | `138dd14b…2636` | Produced by bridge score command | Pinned bridge environment | Environment, device, completion, and score hash | Runtime state, not scientific outcome. |

## D0 Gate 2A and stop decision

| Path | Stage and run date | Type and status | Count | SHA-256 | Reproduction command | Environment | Claim supported | Critical limitation |
|---|---|---|---:|---|---|---|---|---|
| `v2/D0_GATE2A_PROTOCOL.md` | Gate 2A freeze, 2026-08-22 | Frozen protocol; authoritative | 16 families; 256 episodes/family; 4 policies | `e08081a5…3767a` | Not generated | Markdown | Gate estimand, threshold, integrity rule, and claim boundary | No prospective adaptation-diversity criterion. |
| `v2/configs/d0_gate2a_v1.json` | Gate 2A config, 2026-08-22 | Frozen machine config; authoritative | 16 families; 12 questions | `0406be92…3436` | `venv/bin/python v2/scripts/d0_gate2a.py validate` | CPU; standard library | Exact likelihoods, seeds, surfaces, and fixed order | Constructed synthetic distributions. |
| `v2/results/d0_gate2a/trajectories.jsonl` | Gate 2A run, 2026-08-22 | Raw result; authoritative and frozen | 16,384 sampled policy trajectories | `207d9ecf…4a20` | Safe isolated command in `REPRODUCIBILITY.md` | CPU; approximately 43 MB output | Frozen policy ledgers with exact likelihood and posterior arithmetic | Four policies used matched sampled response schedules; this is not complete response-outcome enumeration. |
| `v2/results/d0_gate2a/result.json` | Gate 2A analysis, 2026-08-22 | Result; authoritative | 256 seeded sampled episodes/family; 8 held-out families | `6eb71f3e…11a7` | Safe isolated command in `REPRODUCIBILITY.md` | CPU | Formal PASS, sampled-episode averages, family intervals, and post-hoc diagnostic | Formal pass is narrower than active adaptation and does not cover the complete response-outcome space. |
| `v2/results/d0_gate2a/inspection.json` | Gate 2A audit, 2026-08-22 | Diagnostic; authoritative | 8 representative and 8 worst-BED trajectories | `8c1b61c7…a575` | Produced by isolated analysis command | CPU | Ledger inspection and tail-risk examples | Selected diagnostic subset, not a new statistical analysis. |
| `v2/results/d0_gate2a/run_state.json` | Gate 2A run, 2026-08-22 | Run state; authoritative historical | 16,384 complete IDs | `de886e67…c50a` | Produced by isolated run command | CPU | Completion and config binding | A no-op resume changes its timestamp; do not use it as the frozen outcome hash. |
| `v2/D0_BRIDGE_VALIDITY_DECISION.md` | Program stop, decided 2026-08-29 | Decision; authoritative | Not applicable | `ff27683a…0f2d` | Not generated | Markdown | Stops Gate 2B and the active extension without changing Gate 2A | Does not answer the original real-active research question. |

## Hash notation

The tables abbreviate hashes for readability. Full hashes are listed below and can be recomputed with:

```bash
shasum -a 256 PATH
```

```text
a5a63e9fde42aa0d8b2ce908eca4621f1c325040c29336e549bbb33297f9a8bf  EXPERIMENT_DESIGN.md
ec01c4f3cd420ff62212a50cc490d162e5d9250f35abd4657eb56059ebaa2217  configs/kaggle_t4_validation.yaml
32a570cbc397f615bd4cf82450585452b22147b363a087b2dbfa2e4a347c698d  configs/kaggle_t4_dosecurve.yaml
5f566650cde4c13e118c0cf2b809e22ad9ce13f1f388d000c0563dc77d27f268  results/analysis/dose_curve_20260808_204827.json
778f3bc33234f62d6e51efb6c406cd9903f07ea6d832aa18cd1862783efbe138  configs/kaggle_t4_confound_pilot103.yaml
14f52e5bbec5789c6a6101ebc1d5d1acb7c77cf5cd3ff86f697f3a18320f5f7b  results/analysis/dose_curve_20260812_104622.json
a467a7dd41dc9a189ee2ba9f388c291dcb1f2b2273e527bbb750b496e800a083  results/analysis/confound_pilot103_vs_fullcanon.json
e07e325dbebc0b5e8b8a5ff266cb2305b8b42932ceaf23566250f2776a835940  configs/kaggle_t4_lowrank_r8.yaml
28ec8deeff274fd0f91495ef52cb43b582e5c86aeecc7a3c9eeb69002799fe20  results/analysis/dose_curve_20260814_042400.json
8300b1ad1400e0ee17b0b55b4e0dce61b803e21d28b65b8dd60817c88ec0edf3  results/analysis/effect_curve_20260814_065854.json
c197fb13a73a144c7038d627ede6baac3a2fac8dbab5ad8be3b8f5488c8708a4  results/analysis/mitigation_lowrank_r8.json
8c64eb4d7fbebbd7ed439bfc5186b45c1e1b59bcc1fd846215c3451ea3a7849d  results/analysis/thinking_shift_20260814_171042_transcript.md
6503b507a6940533a4f92ff681fbd5e3d6bd2c6649356e13bc91dc991b41a387  results/analysis/thinking_shift_20260814_writeup.md
9590fc9702b2e90b0be9469331400c72f5b3657e79958fab04bbc01f14aeca63  results/pilot/gate2_n20/conversations_20260806_091441.jsonl
1350cbc613654fa136546ea33c38be78c5348e0c33d1ed42a1fd5b571e55c249  results/pilot/gate2_n20/turns_20260806_091441.jsonl
7ab5d1960b4b4381b6b32ec3a5526391a687dbd1921c15392dd82e989da5091c  results/analysis/inference_correction_20260822.md
7253abc3a660a1603f1d1930dc2eaccd0e0f543278cb060b0fdbc610bffbfa8c  v2/experiment_design.md
f5d49dd6be8196e561e9742bd104ad1e334837bfdaef9911810b1d719d6b243a  v2/results/stage_a/literature_matrix.md
8220c0db680d36dbfd63f7497a988fba209e001d64c580c631dfefea496030f8  v2/results/stage_a/source_coverage.md
166884eca27a9490e1db7648417fd1536832f6a19d0e65428c313336d05fbf4a  v2/results/track_a/a0_baselines_20260822_202851_inference-correction-20260822.json
5d3f7689faf6e6d8a510af82c4f5a364e7cf199e0475c1099115e009236b3acc  v2/results/track_a/a0_rung4_loso_20260822_204404_inference-correction-20260822.json
66907ef3ac39584f6dca5f467cae57f32a6bb965a81dc3230903236e2af3cbef  v2/results/track_a/a2_frozen_rep_20260822_203719.json
18f71f06d3fe60691a54ad11eea1ba336af034ab7ef2477711dc30936be4a097  v2/results/track_a/inference_correction_20260822.md
1aca35914cd1d6baaed81d26695905629a9e9ad02a66486f4273df3fc589d9ab  v2/BRIDGE_PROTOCOL.md
2de37dddcbb912d6b41e8031d9487faa69335b70ef9a9cdd98d97f72a8e7297c  v2/results/bridge/openai_gpt2_detector_raw.jsonl
9290325a05c8d851bec7a8d4679e09e93b16e01165958ef7e4b70a566723c32c  v2/results/bridge/openai_gpt2_detector_bridge.json
138dd14b6b3c553a9209f73967af6727f7569af3842fc70ce33e1802b1232636  v2/results/bridge/openai_gpt2_detector_run_state.json
e08081a5c17c12025400ecd2ac9a5adfdab2c2709b006be0d90b166efde3767a  v2/D0_GATE2A_PROTOCOL.md
0406be92202a2559d74343fa2ea1dc141a31dda18d33d01228734335a1fc3436  v2/configs/d0_gate2a_v1.json
207d9ecf91e93c1a1b04e1f6b3903f46d80668f87afc1dc6fbed6615f58f4a20  v2/results/d0_gate2a/trajectories.jsonl
6eb71f3efd251f24393e49a37664cc3ccf5b190748096b2b03bbceec593411a7  v2/results/d0_gate2a/result.json
8c1b61c73fff18f2bca670a39a5ae05d3a08fc09ddd58e5c7381e064ca26a575  v2/results/d0_gate2a/inspection.json
de886e672a670bd3d62c82ae37ffb55927bdb8bec25cd9f1c5958c3c7d94c50a  v2/results/d0_gate2a/run_state.json
ff27683a028fc4542661a3c4fbb20247e5a550589d0b52f34a728d9c40b50f2d  v2/D0_BRIDGE_VALIDITY_DECISION.md
```

The full frozen Gate 2A hashes are:

- `trajectories.jsonl`: `207d9ecf91e93c1a1b04e1f6b3903f46d80668f87afc1dc6fbed6615f58f4a20`
- `result.json`: `6eb71f3efd251f24393e49a37664cc3ccf5b190748096b2b03bbceec593411a7`
- `inspection.json`: `8c1b61c73fff18f2bca670a39a5ae05d3a08fc09ddd58e5c7381e064ca26a575`
- frozen protocol: `e08081a5c17c12025400ecd2ac9a5adfdab2c2709b006be0d90b166efde3767a`
