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
	@echo "  test           run smoke tests (pure logic, no network/Ollama needed)"
	@echo "  lint           byte-compile all scripts and tests (syntax check)"

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

test:
	$(PY) -m unittest discover -s tests -v

lint:
	$(PY) -m compileall -q scripts tests
