# OpenAI GPT-2 output detector (RoBERTa base)

## Identity and immutable revision

- Model: `openai-community/roberta-base-openai-detector`
- Hugging Face revision: `6cba99c003b711c7fe94f8a3aa2be35a792cb6fa`
- Selected weight file: `model.safetensors`
- Weight SHA-256 (Hugging Face LFS oid):
  `3abd6d2b005f5876b945cb5b68ddde04f6e28fbd9c5d6dc5adfb06ba647e0546`
- Weight size: 500,975,390 bytes
- Config git oid: `20ca4c4d7a0c23f2a0b6974b4e243472b077729c`
- Model API queried: 2026-08-22
- Upstream model card: <https://huggingface.co/openai-community/roberta-base-openai-detector>
- Upstream detector repository:
  <https://github.com/openai/gpt-2-output-dataset/tree/master/detector>
- License: MIT

The revision and safetensors hash are executable provenance, not merely a model
name. The scorer rejects a different revision or label mapping. It uses local
inference only and never sends corpus text to an API.

## Training boundary

OpenAI trained the classifier in 2019 to distinguish WebText from output of the
1.5B GPT-2 model. The Jones & Bergen conversations evaluated here were collected
in 2025 and contain outputs from GPT-4.5, GPT-4o, Llama 3.1 405B and ELIZA. The
detector is therefore out-of-corpus and temporally prior, but its synthetic-text
training distribution is very different from the target.

## Allowed interpretation

This is an intentionally old, cheap transfer probe. A positive result would show
that some generic external signal survives the target's persona prompts. A
negative result would not show that modern detectors fail, because the model was
trained specifically on GPT-2/WebText and on much longer examples (the upstream
evaluation used 510-token samples). It must not be represented as a contemporary
state-of-the-art detector.

The upstream card warns against using its predictions for grave allegations of
misconduct and says automated detection is not adequate as a standalone method.
This repository uses it only for aggregate methodological research. No
individual-level judgment is reported.

## Target preprocessing

- Input is witness turns only, joined in chronological order with newlines.
- No template normalization or target-corpus fitting precedes raw scoring.
- Empty-witness games are excluded; this includes released sides whose messages
  exist as rows but whose content was erased entirely by redaction.
- Tokenization is the pinned model tokenizer with truncation at 512 model tokens.
- Per-dialogue raw probabilities are durable, resumable, and text-free; input
  text is represented only by SHA-256.
- The frozen Track A test split is excluded before scoring.

## Gate 0 / redistribution

The model is MIT-licensed. The Jones & Bergen source remains governed by its own
registry record. Raw conversations are neither copied into results nor uploaded
to a third party.
