"""Async conversation orchestrator: runs two agents in adversarial alternating turns."""
import uuid

from openai import AsyncOpenAI

import agent as _agent
import conv_logging
from schema import (
    AgentConfig,
    ConversationConfig,
    ConversationRecord,
    ConversationResult,
    TurnRecord,
)


def _make_client(cfg: AgentConfig) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=cfg.endpoint, api_key=cfg.api_key)


def _cfg_to_dict(cfg: AgentConfig) -> dict:
    return {
        "model_id":            cfg.model_id,
        "adapter_id":          cfg.adapter_id,
        "role":                cfg.role,
        "ground_truth_is_llm": cfg.ground_truth_is_llm,
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

        output, prompt_toks, gen_toks, latency = await _agent.generate_turn(
            history, speaker_cfg, client, cfg.seed + turn_idx
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
        )
        turns.append(record)

        if turns_path:
            conv_logging.log_turn(record, turns_path)

        # Only the reply crosses the channel; all other fields stay private
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

    metrics = conv_logging.compute_conversation_metrics(turns)
    ref_m   = metrics.get(winner or "A", {})

    A_accused = any(t.public_accusation for t in turns if t.speaker_id == "A")
    B_accused = any(t.public_accusation for t in turns if t.speaker_id == "B")
    # "correct" means the accusing agent correctly flagged its opponent as an LLM
    A_correct: bool | None = cfg.agent_B.ground_truth_is_llm if A_accused else None
    B_correct: bool | None = cfg.agent_A.ground_truth_is_llm if B_accused else None

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
    )

    if conv_path:
        conv_logging.log_conversation(conv_record, conv_path)

    return ConversationResult(record=conv_record, turns=turns)
