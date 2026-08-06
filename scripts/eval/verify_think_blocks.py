#!/usr/bin/env python3
"""Does a fine-tuned adapter still emit <think> blocks?  THE stage-0 gate.

The three-level commitment gap depends entirely on <think> content surviving
fine-tuning. If QLoRA on raw prose degrades the reasoning format, the project's
novel measurement disappears and no amount of GPU budget recovers it. This is
the cheapest place to find that out.

Extraction uses agent.py's own _resolve_think_block rather than a local reimpl,
so this script cannot pass while the real orchestrator fails.

    python scripts/eval/verify_think_blocks.py \
        --adapter outputs/kaggle_t4_validation_seed42/final_adapter \
        --base unsloth/DeepSeek-R1-Distill-Qwen-7B

Exit code 0 = every generation produced a think block (PASS).
Exit code 1 = at least one did not. Do not spend GPU budget on a FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "conversation"))

# Deduction-inviting prompts. Holmes-flavoured because a fine-tune that took
# should reason in that register, but the FORMAT check works regardless of
# content — we are testing that <think> appears at all, not that it is clever.
PROMPTS = [
    "You meet a stranger whose left cuff is frayed and whose right shoe is "
    "newly resoled. What do you conclude, and why?",
    "A man claims he walked here from the station, but his umbrella is dry on "
    "a rainy day. What follows?",
    "Someone in conversation uses a phrase that sounds slightly too formal. "
    "How would you decide whether that means anything?",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True, help="path to final_adapter/")
    ap.add_argument("--base", default="unsloth/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--max-new-tokens", type=int, default=1200,
                    help="1200, not 400. Pilot think blocks ran 1.4-1.8K chars, "
                         "and because the chat template pre-opens <think>, the "
                         "only detectable boundary is the CLOSING tag. A 400-"
                         "token budget truncates before it can appear and the "
                         "run reads as a false FAIL — observed 2026-07-28.")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="Matches agent.py, so this exercises the real "
                         "sampling configuration rather than a greedy one.")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from agent import _resolve_think_block  # the production extractor

    # Accept a local path OR a HuggingFace repo id. Kaggle deletes
    # /kaggle/working at session end, so a local-only path makes this script
    # unrunnable exactly when you most want to re-check a result.
    adapter = Path(args.adapter)
    is_local = adapter.exists()
    if not is_local and "/" not in args.adapter:
        sys.exit(f"ERROR: {args.adapter} is neither a local path nor a repo id")
    adapter_ref = str(adapter) if is_local else args.adapter
    print(f"  adapter source: {'local path' if is_local else 'HuggingFace repo'}")

    # bf16 needs NATIVE Ampere tensor cores (SM 8.0+). Do not use
    # torch.cuda.is_bf16_supported(): it counts Turing's software emulation
    # and returns True on a T4. Must match train_lora.native_bf16(), or the
    # adapter is evaluated under a different dtype than it was trained in.
    bf16_ok = (torch.cuda.is_available()
               and torch.cuda.get_device_capability()[0] >= 8)
    print(f"  compute dtype: {'bfloat16' if bf16_ok else 'float16'}")

    # Same quantization as training. Loading onto a differently-quantized base
    # would test a configuration that will never actually be run.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16_ok else torch.float16,
    )

    print(f"  loading base : {args.base}")
    # Tokenizer from the BASE model, not the adapter: a LoRA adapter directory
    # has adapter_config.json and no config.json, so AutoTokenizer's AutoConfig
    # path fails there (and on a private HF repo the miss surfaces as a
    # confusing 401 rather than a 404).
    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb,
        device_map={"": 0},   # pin to GPU 0 — "auto" shards across both T4s
        trust_remote_code=True,
    )
    print(f"  loading adapter: {adapter_ref}")
    model = PeftModel.from_pretrained(base, adapter_ref)
    model.eval()

    print("\n" + "=" * 70)
    print("  THINK-BLOCK SURVIVAL TEST")
    print("=" * 70)

    n_ok = 0
    for i, prompt in enumerate(PROMPTS, 1):
        # The chat template is what triggers R1-distill's thinking format.
        # Applying it wrongly is the most common cause of think blocks silently
        # vanishing — precisely the failure this script exists to catch.
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=False)

        # {} because HF generation has no message_extra. _extract_think_block
        # handles the pre-opened shape (closing tag only), which is what this
        # chat template produces.
        think, _ = _resolve_think_block(text, {})
        ok = bool(think and think.strip())
        n_ok += ok

        print(f"\n  [{i}/{len(PROMPTS)}] think block: {'YES' if ok else 'NO'}"
              f"  ({len(think) if think else 0} chars)")
        if ok:
            print(f"      {think[:200].strip()}...")
        else:
            print(f"      raw: {text[:200].strip()}...")

    print("\n" + "=" * 70)
    print(f"  RESULT: {n_ok}/{len(PROMPTS)} generations contained a think block")
    if n_ok == len(PROMPTS):
        print("  PASS — the reasoning format survived fine-tuning.")
        print("  Stage 0 complete. Proceed to the 14B run.")
    elif n_ok:
        print("  PARTIAL — intermittent. Check the chat template, and whether")
        print("  stray '<think>' strings leaked into the training corpus.")
    else:
        print("  FAIL — fine-tuning destroyed the reasoning format.")
        print("  STOP. Do not spend GPU budget; no compute recovers this.")
        print("  BUT FIRST rule out the instrument, in this order:")
        print("    1. Is the raw text above reasoning-shaped? If it reads like")
        print("       thinking, the model is fine and the EXTRACTOR is wrong.")
        print("    2. Raise --max-new-tokens: the chat template pre-opens")
        print("       <think>, so only the CLOSING tag is detectable, and a")
        print("       short budget truncates before it appears.")
        print("    3. Only then suspect the chat template or modules_to_save.")
    print("=" * 70)
    sys.exit(0 if n_ok == len(PROMPTS) else 1)


if __name__ == "__main__":
    main()
