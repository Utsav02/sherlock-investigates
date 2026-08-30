# Reproducibility guide

## Scope

This guide verifies the retained methodological failure study. It does not authorize new research. Do not run Gate 2B, create D0 SFT, use a GPU, collect human data, or access the frozen Track A test split.

The base research revision is `75a751e000387da9e5238789d810451688b0beeb` on branch `track-a-ablation`. The closeout package revision is the archival commit containing this guide; see Git history for its hash. The repository has intentional unrelated working-tree changes, so use path-specific commands and never use `git add .` or `git add -A`.

## Local environments and storage

| Environment | Current local version | Purpose | Hardware | Approximate storage |
|---|---|---|---|---:|
| `venv/` | Python 3.13.2 | CPU tests, Track A analysis, D0 | CPU; no network for tests and D0 | 384 MB |
| `.bridge-venv/` | Python 3.12.5 | Pinned external-detector scoring | CPU; model cache or network required for rescoring | 685 MB |
| Repository | Not applicable | Tracked code, data, and results | Not applicable | 1.8 GB including environments and ignored data |
| `results/` | Not applicable | V1 results | Not applicable | 8.1 MB |
| `v2/results/` | Not applicable | V2 and D0 results | Not applicable | 44 MB, mainly Gate 2A trajectories |
| Ignored v2 source/canonical layers | Not applicable | Track A local reproduction inputs | CPU | Approximately 47 MB for current `v2/data/` |

The current `requirements.txt` header describes a Python 3.12/macOS resolution, while the existing root environment reports Python 3.13.2. The verified test suite, not that header, is the evidence for the current local environment. The bridge lock is explicitly resolved for CPython 3.12.5 on macOS arm64.

## Required local or ignored inputs

Track A source text and canonical data are intentionally ignored because redistribution permission is unclear. Exact local Track A reproduction needs:

- `v2/data/sources/jones_bergen_2025/data/` from OSF node `jk7bw`;
- the local manifest in `v2/data/sources/jones_bergen_2025/MANIFEST.json`;
- `v2/data/canonical/main_study_v1/`;
- `v2/data/canonical/splits/main_study_v1.json` and its frozen SHA-256 file;
- cached A2 embeddings or a local Qwen2.5-7B embedding server for A2; and
- the pinned OpenAI detector model cache, or network access, only if raw bridge scores are regenerated.

The ignored 15-minute source is not authorized for analysis. Its Gate 0 remains unresolved. The frozen Track A final test split must not be read or scored.

V1 training reruns need model downloads, a compatible historical Unsloth/PyTorch/CUDA stack, and GPU hardware. Those reruns are not authorized as part of project closeout.

## Safe commands that require no network, model download, GPU, or protected data

Run the complete CPU test suite and syntax checks:

```bash
make lint
make test
```

Validate the frozen D0 config without writing results:

```bash
venv/bin/python v2/scripts/d0_gate2a.py validate
```

Regenerate Gate 2A in an isolated temporary directory. This uses only the frozen config and standard-library CPU code. It took 36.6 seconds in the historical production run and writes about 43 MB of trajectories.

```bash
D0_VERIFY_DIR="$(mktemp -d)"
venv/bin/python v2/scripts/d0_gate2a.py run \
  --config v2/configs/d0_gate2a_v1.json \
  --output "$D0_VERIFY_DIR/trajectories.jsonl" \
  --state "$D0_VERIFY_DIR/run_state.json"
venv/bin/python v2/scripts/d0_gate2a.py analyze \
  --config v2/configs/d0_gate2a_v1.json \
  --input "$D0_VERIFY_DIR/trajectories.jsonl" \
  --output "$D0_VERIFY_DIR/result.json" \
  --inspection "$D0_VERIFY_DIR/inspection.json"
shasum -a 256 "$D0_VERIFY_DIR/trajectories.jsonl" \
  "$D0_VERIFY_DIR/result.json" "$D0_VERIFY_DIR/inspection.json"
```

Expected hashes:

```text
207d9ecf91e93c1a1b04e1f6b3903f46d80668f87afc1dc6fbed6615f58f4a20  trajectories.jsonl
6eb71f3efd251f24393e49a37664cc3ccf5b190748096b2b03bbceec593411a7  result.json
8c1b61c73fff18f2bca670a39a5ae05d3a08fc09ddd58e5c7381e064ca26a575  inspection.json
```

Do not use `make v2-d0-run` for verification against the retained directory. A no-op resume preserves `trajectories.jsonl` but changes the `run_state.json` resume timestamp. The isolated command avoids modifying frozen artifacts.

Parse every tracked JSON and JSONL artifact without loading protected source text:

```bash
git ls-files '*.json' | while IFS= read -r JSON_PATH; do
  jq empty "$JSON_PATH"
done
git ls-files '*.jsonl' | while IFS= read -r JSONL_PATH; do
  awk 'NF {print}' "$JSONL_PATH" | jq -c empty
done
```

## Supported Track A verification with local ignored inputs

Check that the canonical layer re-derives to the same digests without writing:

```bash
venv/bin/python v2/scripts/build_canonical.py --check
```

The historical analysis commands are:

```bash
venv/bin/python v2/scripts/track_a_a0.py
venv/bin/python v2/scripts/track_a_rung4.py
venv/bin/python v2/scripts/track_a_a2.py --embed-url http://127.0.0.1:51999
```

These scripts create new timestamped results. Run them only in a disposable copy if exact artifact comparison is required. A2 additionally needs the historical representation model or the existing embedding cache. The corrected JSON files in `v2/results/track_a/` are the report authority; earlier writeups retain superseded intervals.

## Supported bridge verification

The current bridge environment can be checked without rescoring:

```bash
scripts/doctor_bridge_env.sh
```

Reanalysis from retained raw scores is CPU-only but requires the ignored canonical game metadata. Write to a temporary path:

```bash
BRIDGE_VERIFY_DIR="$(mktemp -d)"
venv/bin/python v2/scripts/track_a_bridge.py analyze \
  --scores v2/results/bridge/openai_gpt2_detector_raw.jsonl \
  --output "$BRIDGE_VERIFY_DIR/openai_gpt2_detector_bridge.json" \
  --bootstrap 1000
shasum -a 256 "$BRIDGE_VERIFY_DIR/openai_gpt2_detector_bridge.json"
```

Raw rescoring is not necessary to verify the report. It requires the pinned detector weights and tokenizer from Hugging Face or a compatible local cache. The historical CPU pass took 182 seconds after MPS fallback. The retained raw score file contains hashes and scores, not transcript text.

## Historical runs that cannot be reproduced exactly from the public repository alone

| Run family | Why exact reproduction is unavailable or constrained |
|---|---|
| V1 Stage 0 and later QLoRA runs | Adapter weights are external/private or were lost; exact package and CUDA environments were not fully locked in the repository; GPU work is no longer authorized. |
| 2026-08-08 rank-32 closure curve | The original JSON and local checkpoints were lost when Kaggle storage was wiped. The tracked JSON is a labeled reconstruction from printed output. |
| 2026-08-12 pilot-repeat closure curve | The run persisted externally, but the tracked JSON is reconstructed from notebook output. |
| 2026-08-14 rank-8 closure curve | The tracked closure JSON is reconstructed; the perplexity JSON was later retrieved from the external artifact store. |
| Corrected V1 repeated-measures inference | Historical files lack prompt-level closure vectors and output hashes for every checkpoint. The covariance needed for a corrected p-value cannot be reconstructed without rerunning inference. |
| Track A from a fresh public clone | Source transcripts and canonical derivatives are ignored because redistribution permission is unclear. The manifest and transformation code are public, but the source must be acquired separately under its terms. |
| A2 from raw text | Requires a pre-transcript Qwen2.5-7B representation model and compatible embedding service, or the ignored local embedding cache. |
| External detector raw rescoring | Requires the pinned Hugging Face model revision and ignored canonical text. The retained raw score file is sufficient for analysis verification. |

## Minimal report verification sequence

1. Confirm the base research revision and dirty-tree scope:

   ```bash
   git rev-parse HEAD
   git status --short
   ```

2. Verify the frozen Gate 2A artifacts:

   ```bash
   shasum -a 256 v2/D0_GATE2A_PROTOCOL.md \
     v2/results/d0_gate2a/trajectories.jsonl \
     v2/results/d0_gate2a/result.json \
     v2/results/d0_gate2a/inspection.json
   wc -l v2/results/d0_gate2a/trajectories.jsonl
   ```

3. Verify the bridge raw-score artifact:

   ```bash
   shasum -a 256 v2/results/bridge/openai_gpt2_detector_raw.jsonl
   wc -l v2/results/bridge/openai_gpt2_detector_raw.jsonl
   ```

4. Read the two inference corrections before interpreting older result files:

   ```bash
   sed -n '1,220p' results/analysis/inference_correction_20260822.md
   sed -n '1,220p' v2/results/track_a/inference_correction_20260822.md
   ```

5. Run `make lint`, `make test`, Markdown link validation, JSON/JSONL parsing, and `git diff --check`.

6. Confirm the stop boundary:

   ```bash
   rg -n 'Owner stops|stop the active|Gate 2B.*never|test split.*untouched' \
     STATUS.md v2/DECISIONS.md v2/D0_BRIDGE_VALIDITY_DECISION.md \
     docs/FINAL_RESEARCH_REPORT.md
   ```

## Expected report-level invariants

- V1 training was real QLoRA training; D0 fine-tuning did not occur.
- Historical V1 pooled repeated-measures p-values are withdrawn.
- Track A measured passive identity signal on real transcripts, not active questioning.
- The external detector bridge depended on nested calibration learning an inverse score relationship.
- Gate 2A evaluated 16,384 frozen sampled policy trajectories and formally passed; it did not enumerate all response outcomes.
- Gate 2A selected one sequence per family and did not exercise response-conditioned adaptation.
- Gate 2B has no outcome because it was never frozen or run.
- The Track A final test split remains untouched.
- The project stopped before a trained adaptive investigator existed.
