# Sherlock Investigates

A fine-tuning and adversarial conversation experiment at the intersection of LLM fine-tuning, deception detection, and chain-of-thought analysis.

## What this is

Two phases: (1) fine-tune small open-weights models on different corpora to produce variants with distinguishable reasoning priors; (2) place pairs of variants into adversarial conversations where each agent tries to identify the other as human or AI while passing as human itself. The novel measurement is the temporal gap between when an agent first becomes suspicious in its private chain of thought and when it commits to a decision in its visible utterances.

Full design rationale and all decisions: [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md).

## Repository layout

```
data/
  raw/          Gutenberg downloads, untouched (large files gitignored)
  processed/    Stripped and normalized text, train/heldout splits
  augmented/    Reformatted training data (after augmentation pipeline)
  probes/       Behavioral probe prompt sets (tracked in git)
scripts/
  data_prep/    Extraction and augmentation pipeline
  training/     LoRA fine-tuning scripts
  eval/         Perplexity and behavioral probe scripts
  conversation/ Conversation orchestration
  analysis/     Statistical analysis notebooks
configs/        YAML hyperparameter configs per run
results/
  pilot/        Pilot perplexity, probes, generation samples
  full/         Full experiment data
  analysis/     Final figures, tables, statistical outputs
```

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Data preparation

Download and clean the pilot corpus:

```bash
# A Study in Scarlet (pilot training corpus, ~43K words)
python scripts/data_prep/download_gutenberg.py
```

Saves the untouched Gutenberg file to `data/raw/` and the stripped/normalized novel text to `data/processed/`.

## Pilot at a glance

| Component | Specification |
|---|---|
| Bases | Qwen2.5-7B-Instruct, Mistral-7B-v0.3 |
| Adapter method | QLoRA, 4-bit NF4 quantization |
| Rank / alpha | 32 / 64 |
| Training corpus | ~60K words Sherlock canon, 3-5x augmentation |
| Held-out | The Adventure of the Speckled Band |
| Compute | RunPod Community RTX 4090, ~$0.34/hr |
| Budget | $5-8 USD total |

See EXPERIMENT_DESIGN.md for the full pilot table, decision flowchart, and analysis plan.

## Session log

| Session | Work done |
|---|---|
| 1 | Repository scaffold, requirements.txt, download_gutenberg.py |
| 2 | gutenberg_utils.py (shared download/strip/normalize); download_adventures.py splits ebook #1661 into stories; training stories (Scandal in Bohemia, Red-Headed League) and heldout (Speckled Band) saved to data/processed/; EXPERIMENT_DESIGN.md updated with specific story names |
| 3 (Cowork) | chunk_audit_report.md — label quality audit + token volume analysis; probe_set_v1.jsonl (30 prompts, 3 categories); augmentation_spec.md (5 framing templates + 15 worked examples); pilot_qwen.yaml + pilot_mistral.yaml (fully populated from design doc); literature_notes.md (8 cited papers structured); pilot_writeup_template.md (pre-filled 9-section template) |
| 4 | augment_corpus.py — runs 5-framing augmentation pipeline via Ollama qwen2.5:7b; executed to produce train.jsonl (576 examples, 137K tokens first pass); added --oversample-central flag; re-run at ×3 → 1168 examples, 325K tokens; filled batch size nulls in configs (per_device=2, grad_accum=8); train_lora.py — QLoRA training script with Unsloth/standard-PEFT fallback |
