"""
MMLU spot-check evaluation — pilot gate H3.

H3 (confirmatory): accuracy on MMLU drops by <3 percentage points vs the base
model (fine-tuning on Holmes canon does not destroy general reasoning).

Evaluates on a fixed subset of MMLU subjects (default 5 subjects × up to 100
questions each = up to 500 questions). Uses the log-probability of the answer
token (A/B/C/D) rather than generation — fast and reliable for multiple choice.

Usage (from repo root):
    python scripts/eval/mmlu_eval.py \\
        --config  configs/pilot_qwen.yaml \\
        --adapter outputs/pilot_qwen_seed42/final_adapter \\
        --output  results/pilot/

    python scripts/eval/mmlu_eval.py \\
        --config  configs/pilot_qwen.yaml \\
        --adapter utsvsngh/sherlock-qwen25-7b-pilot-seed42 \\
        --n-questions 200

Exit code 0 = H3 passes. Exit code 1 = H3 fails.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Gate threshold
H3_MAX_DROP_PP = 3.0  # <3 percentage point accuracy drop

# Subjects chosen to cover different reasoning types without domain overlap
# with Holmes (no history-of-literature, no Victorian-era topics)
SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "college_mathematics",
    "formal_logic",
    "high_school_chemistry",
]

CHOICES = ["A", "B", "C", "D"]

PROMPT_TEMPLATE = """\
The following is a multiple-choice question. Choose the single best answer.

{question}
A. {A}
B. {B}
C. {C}
D. {D}

Answer:"""


# ---------------------------------------------------------------------------
# Model loading (same 4-bit NF4 config as training)
# ---------------------------------------------------------------------------

def _bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model(model_name: str, adapter_path: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# MMLU scoring
# ---------------------------------------------------------------------------

def get_choice_token_ids(tokenizer) -> list[int]:
    """Return the single token id for each of A B C D."""
    ids = []
    for ch in CHOICES:
        toks = tokenizer(ch, add_special_tokens=False).input_ids
        # Take the last token in case the tokenizer prepends a space
        ids.append(toks[-1])
    return ids


def predict_answer(model, tokenizer, prompt: str, choice_ids: list[int]) -> int:
    """Return the index (0-3) of the highest-probability answer token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1, :]  # (vocab,)
    choice_logits = logits[choice_ids]
    return int(choice_logits.argmax().item())


def evaluate_subject(model, tokenizer, subject: str,
                     choice_ids: list[int], n_max: int) -> dict:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
    ds = ds.select(range(min(n_max, len(ds))))

    correct = 0
    for row in ds:
        prompt = PROMPT_TEMPLATE.format(
            question=row["question"],
            A=row["choices"][0],
            B=row["choices"][1],
            C=row["choices"][2],
            D=row["choices"][3],
        )
        pred = predict_answer(model, tokenizer, prompt, choice_ids)
        if pred == row["answer"]:
            correct += 1

    return {"subject": subject, "n": len(ds), "correct": correct,
            "accuracy": correct / len(ds)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MMLU evaluation (H3)")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--adapter",     required=True)
    parser.add_argument("--output",      default="results/pilot")
    parser.add_argument("--n-questions", type=int, default=100,
                        help="max questions per subject (default 100)")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    base_model = cfg["base_model"]
    out_dir    = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MMLU evaluation")
    print(f"  Config:   {args.config}")
    print(f"  Base:     {base_model}")
    print(f"  Adapter:  {args.adapter}")
    print(f"  Subjects: {SUBJECTS}")
    print(f"  Max q/subject: {args.n_questions}")
    print("=" * 60)

    output = {
        "run_name":  cfg.get("run_name", "unknown"),
        "base_model": base_model,
        "adapter":   args.adapter,
        "subjects":  SUBJECTS,
        "n_max_per_subject": args.n_questions,
        "base":      {},
        "finetuned": {},
        "gates":     {},
    }

    for label, adapter in [("base", None), ("finetuned", args.adapter)]:
        print(f"\n[{label}] Loading model ...")
        model, tokenizer = load_model(
            base_model,
            adapter_path=adapter,
        )
        choice_ids = get_choice_token_ids(tokenizer)

        subject_results = []
        total_correct = 0
        total_n = 0

        for subj in SUBJECTS:
            print(f"  {subj} ...", end=" ", flush=True)
            res = evaluate_subject(model, tokenizer, subj, choice_ids, args.n_questions)
            subject_results.append(res)
            total_correct += res["correct"]
            total_n       += res["n"]
            print(f"{res['correct']}/{res['n']} = {res['accuracy']*100:.1f}%")

        overall_acc = total_correct / total_n
        output[label] = {
            "subjects":    subject_results,
            "total_n":     total_n,
            "total_correct": total_correct,
            "accuracy":    overall_acc,
        }
        print(f"  Overall: {total_correct}/{total_n} = {overall_acc*100:.1f}%")

        del model  # free VRAM before loading next

    # ---- Gate ----
    base_acc = output["base"]["accuracy"]
    ft_acc   = output["finetuned"]["accuracy"]
    drop_pp  = (base_acc - ft_acc) * 100
    h3_pass  = drop_pp < H3_MAX_DROP_PP

    print("\n" + "=" * 60)
    print("GATE RESULTS")
    print("=" * 60)
    print(f"  Base accuracy:  {base_acc*100:.1f}%")
    print(f"  FT accuracy:    {ft_acc*100:.1f}%")
    print(f"  Drop: {drop_pp:.1f}pp   (gate: <3pp)   {'✓ PASS' if h3_pass else '✗ FAIL'}")
    print("=" * 60)

    output["gates"] = {
        "H3_base_accuracy_pct":     round(base_acc * 100, 2),
        "H3_finetuned_accuracy_pct": round(ft_acc * 100, 2),
        "H3_drop_pp":               round(drop_pp, 2),
        "H3_pass":                  h3_pass,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"mmlu_{cfg.get('run_name', 'run')}_{ts}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults → {out_path.relative_to(ROOT)}")

    if not h3_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
