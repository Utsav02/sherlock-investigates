# Sherlock Investigates — canonical entry points.
# All targets use the repo venv directly; no need to activate it first.
#
# Data pipeline (in order):  download → chunk → classify → augment
# Re-runs are cheap: classify and augment serve repeats from on-disk caches
# (data/processed/.cache/, data/augmented/.cache/), so a fully-cached pass
# rewrites identical outputs without calling Ollama.
#
# classify and augment need a local Ollama server (http://localhost:11434)
# only on cache misses. Training needs a GPU — see docs/runpod-runbook.md.

PY := venv/bin/python

# Ollama model used for classification/augmentation. The committed corpus and
# caches were built with qwen2.5:7b — changing this regenerates from scratch.
MODEL ?= qwen2.5:7b

# Central-chunk oversampling for augmentation. The committed train.jsonl
# (1168 examples, ~325K tokens) was built with 3, per the design doc.
OVERSAMPLE ?= 3

.PHONY: help install run pipeline download chunk classify augment \
        full-canon chunk-full-canon classify-full-canon augment-full-canon \
        train-qwen train-mistral \
        eval-qwen eval-mistral \
        label-tool score-detector \
        v2-fetch-3p v2-inspect-3p v2-itb-length \
        v2-splits v2-precision v2-paper-exclusions v2-policy \
        v2-canonical v2-track-a0 v2-track-a0-ablation v2-rung2 \
        test lint

help:
	@echo "Sherlock Investigates — targets:"
	@echo "  install        create venv and install pinned requirements"
	@echo "  run            full data pipeline: download → chunk → classify → augment"
	@echo "  download       fetch + clean Gutenberg corpus into data/raw + data/processed"
	@echo "  chunk          split training stories into data/processed/chunks.jsonl"
	@echo "  classify       label chunks via Ollama ($(MODEL)) → chunks_labeled.jsonl"
	@echo "  augment        build training set (central x$(OVERSAMPLE)) → data/augmented/train.jsonl"
	@echo "  train-qwen     QLoRA fine-tune Qwen base   (GPU only — run on Kaggle)"
	@echo "  train-mistral  QLoRA fine-tune Mistral base (GPU only — run on Kaggle)"
	@echo "  eval-qwen      Run all 3 pilot eval scripts for Qwen   (set ADAPTER=...)"
	@echo "  eval-mistral   Run all 3 pilot eval scripts for Mistral (set ADAPTER=...)"
	@echo "  label-tool     build the think-stance labelling GUI (open it in a browser)"
	@echo "  score-detector score t_think_07 against hand labels — GATE 1"
	@echo "  test           run smoke tests (pure logic, no network/Ollama needed)"
	@echo "  lint           byte-compile all scripts and tests (syntax check)"
	@echo ""
	@echo "v2 (see v2/experiment_design.md) — Stage A, no GPU, no inference:"
	@echo "  v2-fetch-3p    download OSF $(OSF_NODE) into v2/data/sources/ + MANIFEST.json"
	@echo "  v2-inspect-3p  measure both three-party studies -> v2/results/stage_a/"
	@echo "  v2-itb-length  identify Inverse Turing Bench's length unit (set ITB_CSV=...)"
	@echo "  v2-splits      freeze the participant-level Track A split (prints sha256)"
	@echo "  v2-precision   MDD / CI width per Track A contrast -> results/stage_a/"
	@echo "  v2-paper-exclusions  run the released .Rmd to reproduce the 1,023-game subset (needs R)"
	@echo "  v2-policy      print the canonical-layer column exclusions (PII policy)"
	@echo "  v2-canonical   normalize the cleared 5-min study -> v2/data/canonical/"
	@echo "  v2-track-a0    Track A arm A0 baselines, contrasts P1+P2 -> results/track_a/"
	@echo "  v2-track-a0-ablation  A0 three-way ablation (witness-only / length-free)"
	@echo "  v2-rung2       Gate 1 rung 2: SONA<->Prolific transfer, both directions"

install:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt

run: pipeline

pipeline: download chunk classify augment

download:
	$(PY) scripts/data_prep/download_gutenberg.py
	$(PY) scripts/data_prep/download_adventures.py

chunk:
	$(PY) scripts/data_prep/chunk_stories.py

classify:
	$(PY) scripts/data_prep/classify_chunks.py --model $(MODEL)

augment:
	$(PY) scripts/data_prep/augment_corpus.py --model $(MODEL) --oversample-central $(OVERSAMPLE)

# Full-canon pipeline (all 9 works → full_canon_train.jsonl)
# classify and augment are cached — re-runs only call Ollama on misses.
full-canon: chunk-full-canon classify-full-canon augment-full-canon

chunk-full-canon:
	$(PY) scripts/data_prep/chunk_full_canon.py

classify-full-canon:
	$(PY) scripts/data_prep/classify_chunks.py --model $(MODEL) \
		--input  data/processed/full_canon_chunks.jsonl \
		--output data/processed/full_canon_chunks_labeled.jsonl

augment-full-canon:
	$(PY) scripts/data_prep/augment_corpus.py --model $(MODEL) \
		--oversample-central $(OVERSAMPLE) \
		--input  data/processed/full_canon_chunks_labeled.jsonl \
		--output data/augmented/full_canon_train.jsonl

# Training runs on a GPU pod, not locally — torch/transformers/peft are
# deliberately not in requirements.txt. See docs/runpod-runbook.md.
train-qwen:
	$(PY) scripts/training/train_lora.py --config configs/pilot_qwen.yaml

train-mistral:
	$(PY) scripts/training/train_lora.py --config configs/pilot_mistral.yaml

# Eval runs — set ADAPTER to local path or HF Hub repo ID
# e.g.  make eval-qwen ADAPTER=outputs/pilot_qwen_seed42/final_adapter
#        make eval-qwen ADAPTER=utsvsngh/sherlock-qwen25-7b-pilot-seed42
ADAPTER ?= outputs/pilot_qwen_seed42/final_adapter

eval-qwen: eval-qwen-perplexity eval-qwen-mmlu eval-qwen-probe

eval-qwen-perplexity:
	$(PY) scripts/eval/perplexity.py --config configs/pilot_qwen.yaml \
		--adapter $(ADAPTER) --output results/pilot/

eval-qwen-mmlu:
	$(PY) scripts/eval/mmlu_eval.py --config configs/pilot_qwen.yaml \
		--adapter $(ADAPTER) --output results/pilot/

eval-qwen-probe:
	$(PY) scripts/eval/probe_eval.py --config configs/pilot_qwen.yaml \
		--adapter $(ADAPTER) --output results/pilot/

eval-mistral: eval-mistral-perplexity eval-mistral-mmlu eval-mistral-probe

eval-mistral-perplexity:
	$(PY) scripts/eval/perplexity.py --config configs/pilot_mistral.yaml \
		--adapter $(ADAPTER) --output results/pilot/

eval-mistral-mmlu:
	$(PY) scripts/eval/mmlu_eval.py --config configs/pilot_mistral.yaml \
		--adapter $(ADAPTER) --output results/pilot/

eval-mistral-probe:
	$(PY) scripts/eval/probe_eval.py --config configs/pilot_mistral.yaml \
		--adapter $(ADAPTER) --output results/pilot/

label-tool:
	$(PY) scripts/eval/build_think_label_tool.py

score-detector:
	$(PY) scripts/eval/score_think_detector.py

# ---------------------------------------------------------------------------
# v2 — Stage A dataset audit (v2/experiment_design.md §18). Network only for
# v2-fetch-3p; nothing here trains or runs inference, and v2/data/sources/ is
# treated as immutable once fetched.
# ---------------------------------------------------------------------------
OSF_NODE ?= jk7bw
SOURCE_NAME ?= jones_bergen_2025
ITB_CSV ?=

v2-fetch-3p:
	$(PY) v2/scripts/fetch_osf_source.py --node $(OSF_NODE) --name $(SOURCE_NAME)

v2-inspect-3p:
	$(PY) v2/scripts/inspect_three_party.py --subdir data    > /dev/null
	$(PY) v2/scripts/inspect_three_party.py --subdir 15_mins > /dev/null
	@echo "wrote v2/results/stage_a/three_party_inspection_{data,15_mins}.json"

# The benchmark file is a separate source and is deliberately not committed;
# pass its path:  make v2-itb-length ITB_CSV=/path/to/...csv
v2-itb-length:
	@test -n "$(ITB_CSV)" || (echo "set ITB_CSV=/path/to/InverseTuringBench_o50_conversations_shuffled.csv"; exit 1)
	$(PY) v2/scripts/itb_length_unit.py --itb-csv $(ITB_CSV) > /dev/null
	@echo "wrote v2/results/stage_a/itb_length_unit.json"

# The split assignment lands in v2/data/canonical/ (gitignored); the freeze is
# the sha256, pinned in tests/test_v2_splits.py.
v2-splits:
	$(PY) v2/scripts/build_splits.py

v2-precision:
	$(PY) v2/scripts/precision_track_a.py

# Runs the AUTHORS' released .Rmd via knitr::purl -- not a reimplementation of
# their exclusion criteria. Needs R with tidyverse + brms; exits non-zero if the
# reproduction stops matching the paper's 1,023 games.
v2-paper-exclusions:
	Rscript v2/scripts/reproduce_paper_exclusions.R

v2-policy:
	$(PY) v2/scripts/canonical_policy.py

# Normalises ONLY the cleared 5-minute study (registry §12 Gate 0 CONDITIONAL).
# The 15-minute study has no resolved Gate 0 and is not read.
v2-canonical:
	$(PY) v2/scripts/build_canonical.py

# Arm A0 only. Evaluates the frozen contrasts P1 and P2 by leave-one-component-out
# cross-fitting over train+dev; the test split is untouched (Gate 5, one shot).
# Arm A1 (Inverse Turing Bench) is deliberately not implemented — see the script.
v2-track-a0: v2-canonical
	$(PY) v2/scripts/track_a_a0.py

# Three-way ablation + the length-equalised control. Answers whether the A0
# signal survives removing interrogator text and every length-derived feature.
v2-track-a0-ablation: v2-canonical
	$(PY) v2/scripts/track_a_a0.py --ablation --bootstrap 1000 --tag ablation
	$(PY) v2/scripts/track_a_a0.py --conditions A0-wit-nolen-capped --variants raw \
		--bootstrap 1000 --tag capped

# Between-experiment holdout, CONFOUNDED with batch/lobby structure (recruitment
# source is perfectly nested in the components). Not a clean population holdout.
v2-rung2: v2-canonical
	$(PY) v2/scripts/track_a_rung2.py

test:
	$(PY) -m unittest discover -s tests -v

lint:
	$(PY) -m compileall -q scripts tests v2/scripts
