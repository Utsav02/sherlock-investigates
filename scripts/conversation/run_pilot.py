"""CLI: run a small pilot batch of adversarial conversations and print a summary."""
import argparse
import asyncio
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Ensure sibling modules are importable when run as a script from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts
from orchestrator import run_conversation
from schema import AgentConfig, ConversationConfig, ConversationResult

ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a pilot batch of adversarial conversations."
    )
    p.add_argument("--model-a",         default="qwen2.5:7b")
    p.add_argument("--model-b",         default="qwen2.5:7b")
    p.add_argument("--endpoint",        default="http://localhost:11434/v1")
    p.add_argument("--api-key",         default="ollama")
    p.add_argument("--adapter-a",       default=None)
    p.add_argument("--adapter-b",       default=None)
    p.add_argument("--n-conversations", type=int, default=5)
    p.add_argument("--max-turns",       type=int, default=12)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--output-dir",      default="results/pilot/conversations/")
    p.add_argument("--thinking-mode",   action="store_true",
                   help="R1-distill-style models: use thinking-compatible prompts "
                        "and extract <think> blocks (sets AgentConfig.thinking_mode "
                        "on both agents)")
    p.add_argument("--no-personas",     action="store_true",
                   help="Disable persona symmetry-breaking (reproduces the "
                        "pre-2026-07-26 degenerate-loop conditions)")
    p.add_argument("--temperature",      type=float, default=0.7)
    p.add_argument("--frequency-penalty", type=float, default=0.3)
    p.add_argument("--presence-penalty",  type=float, default=0.3)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    return p.parse_args()


def _print_summary(results: list[ConversationResult]) -> None:
    n          = len(results)
    accused    = [r for r in results if r.record.termination_reason == "accusation"]
    gaps       = [r.record.commitment_gap for r in results if r.record.commitment_gap is not None]
    trap_types: list[str] = []

    parse_modes: Counter = Counter()
    think_present = 0
    think_lens: list[int] = []
    n_turns_total = 0
    scores_by_pos: dict[tuple[str, int], list[float]] = defaultdict(list)
    for result in results:
        for t in result.turns:
            n_turns_total += 1
            parse_modes[getattr(t, "parse_mode", "json")] += 1
            if t.think_block:
                think_present += 1
                think_lens.append(len(t.think_block))
            scores_by_pos[(t.speaker_id, t.turn_idx)].append(t.suspicion_score)
            ts_type = (t.trap_strategy or {}).get("type", "none")
            if ts_type and ts_type != "none":
                trap_types.append(ts_type)

    print(f"\n{'='*60}")
    print(f"  Pilot summary — {n} conversations")
    print(f"{'='*60}")
    degenerate = [r for r in results if r.record.is_degenerate]
    ratios     = [r.record.unique_reply_ratio for r in results]

    print(f"  Accusations:      {len(accused)} / {n}")
    print(f"  Max-turns:        {n - len(accused)} / {n}")
    print(f"  Parse modes:      {dict(parse_modes)}  ({n_turns_total} turns)")

    # Gate: conversations that collapsed into a repeated utterance carry no
    # accumulating evidence, so every gap metric below is meaningless on them.
    flag = "  <-- GATE FAILED (>20%)" if n and len(degenerate) / n > 0.2 else ""
    print(f"  Degenerate:       {len(degenerate)} / {n}{flag}")
    if ratios:
        print(f"  Unique-reply ratio: mean {sum(ratios)/len(ratios):.2f}"
              f"  (min {min(ratios):.2f})")

    # Delta between the legacy topic-mention measure and the directed measure.
    # If these are equal, the redefinition changed nothing and needs revisiting.
    topic_fired    = sum(1 for r in results if r.record.t_think_topic is not None)
    directed_fired = sum(1 for r in results if r.record.t_think_07 is not None)
    if topic_fired or directed_fired:
        print(f"  t_think fired:    topic {topic_fired}/{n}"
              f"  |  directed {directed_fired}/{n}")
    if think_lens:
        print(f"  Think blocks:     {think_present}/{n_turns_total} turns"
              f"  (mean {sum(think_lens)//len(think_lens)} chars)")
    else:
        print(f"  Think blocks:     0/{n_turns_total} turns"
              "  (expected 0 unless --thinking-mode with an R1-style model)")

    if gaps:
        print(f"  Commitment gaps:  {gaps}  (mean {sum(gaps)/len(gaps):.1f} turns)")
    else:
        print("  Commitment gaps:  none (suspicion_score never reached 0.7 and held)")

    # Mean suspicion per speaker per conversation turn position
    all_idxs = sorted({idx for _, idx in scores_by_pos})
    if all_idxs:
        print(f"\n  Suspicion trajectory (mean suspicion_score per turn):")
        print(f"  {'turn':>5}  {'A':>8}  {'B':>8}")
        for idx in all_idxs:
            a = scores_by_pos.get(("A", idx), [])
            b = scores_by_pos.get(("B", idx), [])
            a_str = f"{sum(a)/len(a):.3f}" if a else "   —  "
            b_str = f"{sum(b)/len(b):.3f}" if b else "   —  "
            print(f"  {idx:>5}  {a_str:>8}  {b_str:>8}")

    if trap_types:
        print(f"\n  Trap strategy types used: {dict(Counter(trap_types))}")
    else:
        print(
            "\n  No non-none trap strategies observed."
            "  If guided_json was unsupported, the fallback parser ran — check raw JSONL."
        )
    print()


async def _run_all(args: argparse.Namespace) -> list[ConversationResult]:
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir    = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    turns_path = out_dir / f"turns_{timestamp}.jsonl"
    conv_path  = out_dir / f"conversations_{timestamp}.jsonl"

    print(f"Output → {out_dir}/")
    print(f"  turns:         turns_{timestamp}.jsonl")
    print(f"  conversations: conversations_{timestamp}.jsonl\n")

    sampling = {
        "temperature":        args.temperature,
        "frequency_penalty":  args.frequency_penalty,
        "presence_penalty":   args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
    }

    results: list[ConversationResult] = []
    for i in range(args.n_conversations):
        seed = args.seed + i
        persona_a, persona_b = ("", "") if args.no_personas else prompts.persona_pair(seed)
        cfg  = ConversationConfig(
            agent_A=AgentConfig(
                model_id=args.model_a,
                endpoint=args.endpoint,
                api_key=args.api_key,
                adapter_id=args.adapter_a,
                thinking_mode=args.thinking_mode,
                persona=persona_a,
                **sampling,
            ),
            agent_B=AgentConfig(
                model_id=args.model_b,
                endpoint=args.endpoint,
                api_key=args.api_key,
                adapter_id=args.adapter_b,
                thinking_mode=args.thinking_mode,
                persona=persona_b,
                **sampling,
            ),
            max_turns=args.max_turns,
            seed=seed,
        )

        result = await run_conversation(cfg, turns_path=turns_path, conv_path=conv_path)
        rec    = result.record
        results.append(result)

        gap_str = f"  gap={rec.commitment_gap}" if rec.commitment_gap is not None else ""
        print(
            f"  [{i+1}/{args.n_conversations}] {rec.conv_id}"
            f"  seed={seed}"
            f"  turns={rec.n_turns}"
            f"  {rec.termination_reason}"
            f"  winner={rec.winner or '—'}"
            f"{gap_str}"
        )

    return results


def main() -> None:
    args = _parse_args()
    results = asyncio.run(_run_all(args))
    _print_summary(results)


if __name__ == "__main__":
    main()
