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

.PHONY: help install run pipeline download chunk classify augment train-qwen train-mistral test lint

help:
	@echo "Sherlock Investigates — targets:"
	@echo "  install        create venv and install pinned requirements"
	@echo "  run            full data pipeline: download → chunk → classify → augment"
	@echo "  download       fetch + clean Gutenberg corpus into data/raw + data/processed"
	@echo "  chunk          split training stories into data/processed/chunks.jsonl"
	@echo "  classify       label chunks via Ollama ($(MODEL)) → chunks_labeled.jsonl"
	@echo "  augment        build training set (central x$(OVERSAMPLE)) → data/augmented/train.jsonl"
	@echo "  train-qwen     QLoRA fine-tune Qwen base   (GPU only — run on RunPod)"
	@echo "  train-mistral  QLoRA fine-tune Mistral base (GPU only — run on RunPod)"
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

# Training runs on a GPU pod, not locally — torch/transformers/peft are
# deliberately not in requirements.txt. See docs/runpod-runbook.md.
train-qwen:
	$(PY) scripts/training/train_lora.py --config configs/pilot_qwen.yaml

train-mistral:
	$(PY) scripts/training/train_lora.py --config configs/pilot_mistral.yaml

test:
	$(PY) -m unittest discover -s tests -v

lint:
	$(PY) -m compileall -q scripts tests
