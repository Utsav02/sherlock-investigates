"""Async conversation orchestrator: runs two agents in adversarial alternating turns."""
import uuid

from openai import AsyncOpenAI

import agent as _agent
import conv_logging
from schema import (
    UNUSABLE_PARSE_MODES,
    AgentConfig,
    ConversationConfig,
    ConversationRecord,
    ConversationResult,
    TurnRecord,
)

# Spacing between conversations in seed space. Must exceed the largest possible
# within-conversation offset (2 * max_turns + 1) by a wide margin so that no two
# conversations ever share a generation seed. Prime, to avoid resonance with any
# stride the caller might use when incrementing base seeds.
_SEED_STRIDE = 100_003
_INT32_MAX = 2**31 - 1


def derive_seed(base_seed: int, turn_idx: int, speaker_id: str) -> int:
    """Unique generation seed per (conversation, turn, speaker).

    The previous scheme (`base_seed + turn_idx`) made conversation i and
    conversation i+1 share 11 of 12 seeds when callers incremented the base by
    one per conversation — so "replicates" were not independent and any variance
    estimate over them was understated. Both agents also shared a seed sequence.
    """
    offset = turn_idx * 2 + (0 if speaker_id == "A" else 1)
    return (base_seed * _SEED_STRIDE + offset) % _INT32_MAX


def _make_client(cfg: AgentConfig) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=cfg.endpoint, api_key=cfg.api_key)


def _cfg_to_dict(cfg: AgentConfig) -> dict:
    return {
        "model_id":            cfg.model_id,
        "adapter_id":          cfg.adapter_id,
        "role":                cfg.role,
        "ground_truth_is_llm": cfg.ground_truth_is_llm,
        "thinking_mode":       cfg.thinking_mode,
        "temperature":         cfg.temperature,
        "frequency_penalty":   cfg.frequency_penalty,
        "presence_penalty":    cfg.presence_penalty,
        "repetition_penalty":  cfg.repetition_penalty,
        "persona":             cfg.persona,
    }


async def run_conversation(
    cfg:        ConversationConfig,
    turns_path=None,
    conv_path=None,
) -> ConversationResult:
    """Run one adversarial conversation and return all records.

    turn_idx increments for every individual agent message (A=even, B=odd).
    max_turns is the number of times each agent speaks; total messages = max_turns * 2.
    """
    conv_id      = cfg.conv_id or uuid.uuid4().hex[:8]
    cfg.agent_A.role = "initiator"
    cfg.agent_B.role = "responder"

    client_A = _make_client(cfg.agent_A)
    client_B = _make_client(cfg.agent_B)

    # Each agent sees its own prior replies as "assistant" and the opponent's as "user"
    history_A: list[dict] = []
    history_B: list[dict] = []

    turns: list[TurnRecord] = []
    termination_reason = "max_turns"
    winner: str | None = None

    for turn_idx in range(cfg.max_turns * 2):
        is_A        = turn_idx % 2 == 0
        speaker_id  = "A"         if is_A else "B"
        speaker_cfg = cfg.agent_A if is_A else cfg.agent_B
        history     = history_A   if is_A else history_B
        client      = client_A    if is_A else client_B

        output, prompt_toks, gen_toks, latency, think_block, messages_input = (
            await _agent.generate_turn(
                history, speaker_cfg, client, derive_seed(cfg.seed, turn_idx, speaker_id)
            )
        )

        record = TurnRecord(
            conv_id=conv_id,
            turn_idx=turn_idx,
            speaker_id=speaker_id,
            speaker_role=speaker_cfg.role,
            model_id=speaker_cfg.model_id,
            adapter_id=speaker_cfg.adapter_id,
            prompt_tokens=prompt_toks,
            gen_tokens=gen_toks,
            latency_ms=latency,
            ground_truth_is_llm=speaker_cfg.ground_truth_is_llm,
            reply=output.reply,
            suspicion_score=output.suspicion_score,
            reasoning_trace=output.reasoning_trace,
            cues=output.cues,
            trap_strategy={
                "plan": output.trap_strategy.plan,
                "type": output.trap_strategy.type,
            },
            public_accusation=output.public_accusation,
            think_block=think_block,
            messages_input=messages_input,
            parse_mode=output.parse_mode,
        )
        turns.append(record)

        if turns_path:
            conv_logging.log_turn(record, turns_path)

        # Only the reply crosses the channel; all other fields stay private.
        # Unusable turns (api_error / parse_failed) contribute NOTHING to either
        # history: an empty or prompt-derived reply is not model output, and
        # feeding it to the opponent corrupts the rest of the conversation.
        if output.parse_mode in UNUSABLE_PARSE_MODES:
            continue

        if is_A:
            history_A.append({"role": "assistant", "content": output.reply})
            history_B.append({"role": "user",      "content": output.reply})
        else:
            history_B.append({"role": "assistant", "content": output.reply})
            history_A.append({"role": "user",      "content": output.reply})

        if output.public_accusation:
            termination_reason = "accusation"
            winner = speaker_id
            break

        # Abort a collapsed conversation rather than burning turns on it.
        degeneracy = conv_logging.conversation_degeneracy(turns)
        if degeneracy["is_degenerate"]:
            termination_reason = "degenerate"
            break

    metrics = conv_logging.compute_conversation_metrics(turns)
    ref_m   = metrics.get(winner or "A", {})
    winner_think_07 = ref_m.get("t_think_07")
    winner_private_07 = ref_m.get("t_private_07")
    think_commitment_gap = (
        (winner_private_07 - winner_think_07)
        if (winner_think_07 is not None and winner_private_07 is not None)
        else None
    )

    A_accused = any(t.public_accusation for t in turns if t.speaker_id == "A")
    B_accused = any(t.public_accusation for t in turns if t.speaker_id == "B")
    # "correct" means the accusing agent correctly flagged its opponent as an LLM
    A_correct: bool | None = cfg.agent_B.ground_truth_is_llm if A_accused else None
    B_correct: bool | None = cfg.agent_A.ground_truth_is_llm if B_accused else None

    degeneracy = conv_logging.conversation_degeneracy(turns)

    conv_record = ConversationRecord(
        conv_id=conv_id,
        agent_A_cfg=_cfg_to_dict(cfg.agent_A),
        agent_B_cfg=_cfg_to_dict(cfg.agent_B),
        n_turns=len(turns),
        winner=winner,
        termination_reason=termination_reason,
        A_correct=A_correct,
        B_correct=B_correct,
        t_private_07=ref_m.get("t_private_07"),
        t_public=ref_m.get("t_public"),
        commitment_gap=ref_m.get("commitment_gap"),
        seed=cfg.seed,
        t_think_07=winner_think_07,
        think_commitment_gap=think_commitment_gap,
        t_think_topic=ref_m.get("t_think_topic"),
        unique_reply_ratio=degeneracy["unique_reply_ratio"],
        max_consecutive_repeats=degeneracy["max_consecutive_repeats"],
        is_degenerate=degeneracy["is_degenerate"],
    )

    if conv_path:
        conv_logging.log_conversation(conv_record, conv_path)

    return ConversationResult(record=conv_record, turns=turns)
