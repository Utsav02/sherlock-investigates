"""CLI: run a small pilot batch of adversarial conversations and print a summary."""
import argparse
import asyncio
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Ensure sibling modules are importable when run as a script from any CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    return p.parse_args()


def _print_summary(results: list[ConversationResult]) -> None:
    n          = len(results)
    accused    = [r for r in results if r.record.termination_reason == "accusation"]
    gaps       = [r.record.commitment_gap for r in results if r.record.commitment_gap is not None]
    trap_types: list[str] = []

    scores_by_pos: dict[tuple[str, int], list[float]] = defaultdict(list)
    for result in results:
        for t in result.turns:
            scores_by_pos[(t.speaker_id, t.turn_idx)].append(t.suspicion_score)
            ts_type = (t.trap_strategy or {}).get("type", "none")
            if ts_type and ts_type != "none":
                trap_types.append(ts_type)

    print(f"\n{'='*60}")
    print(f"  Pilot summary — {n} conversations")
    print(f"{'='*60}")
    print(f"  Accusations:      {len(accused)} / {n}")
    print(f"  Max-turns:        {n - len(accused)} / {n}")

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

    results: list[ConversationResult] = []
    for i in range(args.n_conversations):
        seed = args.seed + i
        cfg  = ConversationConfig(
            agent_A=AgentConfig(
                model_id=args.model_a,
                endpoint=args.endpoint,
                api_key=args.api_key,
                adapter_id=args.adapter_a,
            ),
            agent_B=AgentConfig(
                model_id=args.model_b,
                endpoint=args.endpoint,
                api_key=args.api_key,
                adapter_id=args.adapter_b,
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
