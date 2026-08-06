"""
Behavioral probe evaluation — pilot gate H4.

H4 (confirmatory): the fine-tuned model produces measurably more deduction-
structured responses on DEDUCTION_INVITING prompts than the base model, with
no significant difference on NEUTRAL prompts (separation is domain-specific,
not a blanket style shift).

The probe set has three categories:
  DEDUCTION_INVITING  — prompts that invite Holmesian inference chains (n=10)
  REASONING_REQUIRED  — logic puzzles where both models should perform equally (n=10)
  NEUTRAL             — small-talk where no difference is expected (n=10)

Scoring uses a lexical deduction-density metric:
  score = (deduction_markers - hedging_markers) / max(1, word_count) × 1000

Deduction markers: words/phrases that signal committed inference
  (therefore, thus, hence, clearly, evidently, must be, can only, it follows,
   I deduce, this indicates, this suggests, I conclude, this means, it is clear)

Hedging markers: words that signal avoidance of commitment
  (maybe, perhaps, might, could be, possibly, I'm not sure, I think,
   it's hard to say, uncertain, unclear, difficult to determine)

Gate: mean deduction score on DEDUCTION_INVITING is higher for fine-tuned
model than base model (one-sided, no minimum effect size required at pilot).

Usage (from repo root):
    python scripts/eval/probe_eval.py \\
        --config  configs/pilot_qwen.yaml \\
        --adapter outputs/pilot_qwen_seed42/final_adapter \\
        --output  results/pilot/

    python scripts/eval/probe_eval.py \\
        --config  configs/pilot_qwen.yaml \\
        --adapter utsvsngh/sherlock-qwen25-7b-pilot-seed42 \\
        --max-new-tokens 300

Exit code 0 = H4 passes. Exit code 1 = H4 fails.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

# Reuse the orchestrator's extractor rather than re-implementing the split.
# A divergence between the two would make the gate measure something the
# experiment never sees.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversation"))
from agent import _resolve_think_block  # noqa: E402
import yaml

ROOT = Path(__file__).resolve().parents[2]

PROBE_PATH = ROOT / "data" / "probes" / "probe_set_v1.jsonl"

# Gate: finetuned deduction score must exceed base on DEDUCTION_INVITING
DEDUCTION_CATEGORY = "DEDUCTION_INVITING"

DEDUCTION_MARKERS = [
    r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bclearly\b",
    r"\bevidently\b", r"\bmust be\b", r"\bcan only\b", r"\bit follows\b",
    r"\bI deduce\b", r"\bthis indicates\b", r"\bthis suggests\b",
    r"\bI conclude\b", r"\bthis means\b", r"\bit is clear\b",
    r"\bobviously\b", r"\bwithout doubt\b", r"\bwithout question\b",
    r"\bwe can see\b", r"\bone can see\b",
]

HEDGING_MARKERS = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bmight\b", r"\bcould be\b",
    r"\bpossibly\b", r"\bI'm not sure\b", r"\bI am not sure\b",
    r"\bI think\b", r"\bI guess\b", r"\buncertain\b", r"\bunclear\b",
    r"\bhard to say\b", r"\bdifficult to (say|determine|know)\b",
    r"\bcan't be sure\b", r"\bcannot be sure\b",
]


# ---------------------------------------------------------------------------
# Model loading
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
# Generation and scoring
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, prompt: str,
                     max_new_tokens: int) -> tuple[str, str, bool]:
    """Returns (answer_text, think_text, truncated).

    THREE FIXES, 2026-07-28, all of which invalidated the probe gate on
    R1-distill models:

    1. Uses apply_chat_template. The previous raw `tokenizer(prompt)` call did
       not match how the orchestrator prompts the model, and the chat template
       is what triggers the thinking format at all.

    2. SEPARATES the think block from the answer before scoring. Previously the
       whole decoded string was scored, and on a model whose chat template
       pre-opens <think> that string is mostly REASONING. The consequence is
       not subtle: "I think" is a HEDGING_MARKER and R1 think blocks routinely
       open "Okay, so I think...", while "must be" / "therefore" / "this
       suggests" are DEDUCTION_MARKERS that saturate any R1 reasoning trace.
       Both scores measured reasoning register rather than the answer, so every
       R1-distill variant would score alike and the gate could show no
       separation regardless of the fine-tune.

    3. Decodes with skip_special_tokens=False so </think> survives to be found.
       With it True the tag was stripped and the split was impossible.

    Returns the think block too — scored separately below, because "did the
    fine-tune change the reasoning register inside the think block, the answer,
    or both?" is a more informative question than the original single score.
    """
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy — deterministic for reproducibility
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    n_gen = out.shape[-1] - ids.shape[-1]
    raw = tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=False)

    think, answer = _resolve_think_block(raw, {})
    answer = re.sub(r"<\|[^|]*\|>", "", answer).strip()   # drop chat sentinels
    return answer, (think or ""), n_gen >= max_new_tokens


def deduction_score(text: str) -> dict:
    """Lexical deduction-density score (per 1000 words)."""
    text_lower = text.lower()
    words = len(text_lower.split())

    n_deduction = sum(
        len(re.findall(p, text_lower, re.IGNORECASE))
        for p in DEDUCTION_MARKERS
    )
    n_hedging = sum(
        len(re.findall(p, text_lower, re.IGNORECASE))
        for p in HEDGING_MARKERS
    )

    raw = n_deduction - n_hedging
    density = raw / max(1, words) * 1000

    return {
        "n_deduction_markers": n_deduction,
        "n_hedging_markers":   n_hedging,
        "word_count":          words,
        "deduction_density":   round(density, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Behavioral probe evaluation (H4)")
    parser.add_argument("--config",         required=True)
    parser.add_argument("--adapter",        required=True)
    parser.add_argument("--output",         default="results/pilot")
    parser.add_argument("--max-new-tokens", type=int, default=900,
                        help="900, not 300. Think blocks measured at ~420 "
                             "tokens on the real task (2026-07-28), so 300 "
                             "truncated mid-reasoning and left NO answer to "
                             "score at all.")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    base_model = cfg["base_model"]
    out_dir    = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load probe set
    probes = [json.loads(l) for l in PROBE_PATH.read_text().splitlines() if l.strip()]
    by_cat = {}
    for p in probes:
        by_cat.setdefault(p["category"], []).append(p)

    print("=" * 60)
    print("Behavioral probe evaluation")
    print(f"  Config:         {args.config}")
    print(f"  Base:           {base_model}")
    print(f"  Adapter:        {args.adapter}")
    print(f"  Probes:         {len(probes)} total — "
          + ", ".join(f"{k}:{len(v)}" for k, v in sorted(by_cat.items())))
    print(f"  Max new tokens: {args.max_new_tokens}")
    print("=" * 60)

    output = {
        "run_name":       cfg.get("run_name", "unknown"),
        "base_model":     base_model,
        "adapter":        args.adapter,
        "max_new_tokens": args.max_new_tokens,
        "base":           [],
        "finetuned":      [],
        "gates":          {},
    }

    for label, adapter in [("base", None), ("finetuned", args.adapter)]:
        print(f"\n[{label}] Loading model ...")
        model, tokenizer = load_model(base_model, adapter_path=adapter)

        results = []
        for probe in probes:
            print(f"  [{probe['category']}] probe {probe['id']} ...", end=" ", flush=True)
            answer, think, truncated = generate_response(
                model, tokenizer, probe["prompt"], args.max_new_tokens)
            # The GATE scores the ANSWER — the visible utterance. The think
            # block is scored separately and kept for analysis, never mixed in:
            # mixing them is what made this measure reasoning register instead
            # of behaviour (see generate_response).
            scores       = deduction_score(answer)
            think_scores = deduction_score(think) if think else {}
            record   = {
                "id":                  probe["id"],
                "category":            probe["category"],
                "prompt":              probe["prompt"],
                "response":            answer,
                "think_block":         think,
                "think_scores":        think_scores,
                "truncated":           truncated,
                "n_answer_chars":      len(answer),
                "n_think_chars":       len(think),
                "expected_direction":  probe["expected_direction"],
                **scores,
            }
            results.append(record)
            flag = " TRUNCATED" if truncated else ""
            if not answer:
                flag += " NO-ANSWER"
            print(f"density={scores['deduction_density']:+.2f} "
                  f"(ded={scores['n_deduction_markers']}, hed={scores['n_hedging_markers']}) "
                  f"think={len(think)}c answer={len(answer)}c{flag}")

        n_trunc = sum(1 for r in results if r["truncated"])
        n_empty = sum(1 for r in results if not r["response"])
        if n_trunc or n_empty:
            print(f"\n  WARNING [{label}]: {n_trunc}/{len(results)} truncated, "
                  f"{n_empty}/{len(results)} produced NO answer after the think "
                  f"block.")
            print("  Gate H4 is computed on the ANSWER, so empty answers make it "
                  "meaningless. Raise --max-new-tokens and re-run before reading "
                  "any separation result.")

        output[label] = results
        del model  # free VRAM

    # ---- Gate H4 ----
    def mean_density(records, category):
        vals = [r["deduction_density"] for r in records if r["category"] == category]
        return sum(vals) / len(vals) if vals else 0.0

    base_deduction_density = mean_density(output["base"],      DEDUCTION_CATEGORY)
    ft_deduction_density   = mean_density(output["finetuned"], DEDUCTION_CATEGORY)
    base_neutral_density   = mean_density(output["base"],      "NEUTRAL")
    ft_neutral_density     = mean_density(output["finetuned"], "NEUTRAL")

    h4_pass = ft_deduction_density > base_deduction_density

    print("\n" + "=" * 60)
    print("GATE RESULTS")
    print("=" * 60)
    print(f"  DEDUCTION_INVITING — base:      {base_deduction_density:+.4f}")
    print(f"  DEDUCTION_INVITING — finetuned: {ft_deduction_density:+.4f}")
    delta = ft_deduction_density - base_deduction_density
    print(f"  Delta: {delta:+.4f}   (gate: finetuned > base)   {'✓ PASS' if h4_pass else '✗ FAIL'}")
    print()
    print(f"  NEUTRAL — base:      {base_neutral_density:+.4f}  (reference)")
    print(f"  NEUTRAL — finetuned: {ft_neutral_density:+.4f}  (should be similar)")
    print("=" * 60)

    output["gates"] = {
        "H4_deduction_base_mean_density":      round(base_deduction_density, 4),
        "H4_deduction_finetuned_mean_density": round(ft_deduction_density, 4),
        "H4_deduction_delta":                  round(delta, 4),
        "H4_neutral_base_mean_density":        round(base_neutral_density, 4),
        "H4_neutral_finetuned_mean_density":   round(ft_neutral_density, 4),
        "H4_pass":                             h4_pass,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"probe_{cfg.get('run_name', 'run')}_{ts}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults → {out_path.relative_to(ROOT)}")

    if not h4_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
