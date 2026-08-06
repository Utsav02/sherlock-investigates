"""Per-turn generation against an OpenAI-compatible endpoint."""
import json
import logging
import re
import time

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

import prompts
import schema
from schema import AgentConfig, TrapStrategy, TurnOutput, TURN_SCHEMA


# The anti-echo rule is repeated per-turn, not left in the system prompt alone.
# Same reasoning as the JSON reminder (Decision Log 2026-06-17): the system
# prompt gets buried as context grows and the model reverts after ~3-4 turns.
# The 2026-07-26 anti-echo run showed exactly that decay — clean openings,
# mirroring returning around turn 4 (seed 1003).
_NO_ECHO = "Do not repeat or rephrase what I just said; say something new."

_JSON_REMINDER = (
    f"\n[{_NO_ECHO} Respond with a JSON object only — no other text.]"
)
_JSON_REMINDER_THINKING = (
    f"\n[{_NO_ECHO} After your thinking, respond with a JSON object only — "
    "no other text outside your thinking block.]"
)


def _build_messages(history: list[dict], agent_cfg: AgentConfig) -> list[dict]:
    if agent_cfg.thinking_mode:
        system = (
            prompts.INITIATOR_SYSTEM_THINKING
            if agent_cfg.role == "initiator"
            else prompts.RESPONDER_SYSTEM_THINKING
        )
        reminder = _JSON_REMINDER_THINKING
    else:
        system = (
            prompts.INITIATOR_SYSTEM
            if agent_cfg.role == "initiator"
            else prompts.RESPONDER_SYSTEM
        )
        reminder = _JSON_REMINDER

    # A light persona differentiates two otherwise-identical policies. Without it,
    # same-model self-play has a fixed point at "repeat the last utterance".
    if agent_cfg.persona:
        system = f"{system}\n\n{agent_cfg.persona}"

    messages: list[dict] = [{"role": "system", "content": system}]
    for msg in history:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"] + reminder})
        else:
            messages.append(msg)
    return messages


def _extract_think_block(text: str) -> tuple[str | None, str]:
    """Strip <think>…</think> from raw R1-distill output.

    Handles TWO shapes, because DeepSeek-R1-Distill's chat template ends with
    ``<｜Assistant｜><think>\\n`` — the opening tag lives in the PROMPT, so a
    completion often contains only the closing tag:

        1. balanced   "<think>reasoning</think>answer"   (tag inside the output)
        2. pre-opened "reasoning</think>answer"          (tag consumed by the
                      chat template; verified against unsloth/DeepSeek-R1-
                      Distill-Qwen-7B on 2026-07-28)

    Shape 2 previously returned (None, text) — a SILENT null. On a raw vLLM
    deployment with no ``--reasoning-parser`` that would have made every
    t_think_07 None across an entire run, with nothing in the logs to say why.
    Ollama and vLLM-with-parser are unaffected: they return the reasoning in a
    separate field that _resolve_think_block reads.

    Returns (think_block_content, remainder). (None, text) when neither shape
    is present.
    """
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if m:
        think_block = m.group(1).strip()
        remainder = (text[: m.start()] + text[m.end() :]).strip()
        return think_block, remainder

    # Pre-opened: a closing tag with no opener means everything before it is
    # the think block. Only the FIRST closing tag counts — a later one would be
    # the model writing about tags rather than closing its own.
    close = re.search(r"</think>", text, re.IGNORECASE)
    if close:
        think_block = text[: close.start()].strip()
        remainder = text[close.end():].strip()
        return (think_block or None), remainder

    return None, text


def _resolve_think_block(raw: str, message_extra: dict | None) -> tuple[str | None, str]:
    """Think content arrives via one of two transports, depending on the server:

    - inline ``<think>…</think>`` tags left in ``content`` (raw vLLM with no
      reasoning parser configured);
    - a separate ``reasoning`` / ``reasoning_content`` field with the tags
      stripped from ``content`` (Ollama >= 0.9 OpenAI-compat endpoint,
      vLLM with ``--reasoning-parser``). Verified against Ollama 0.30.8 on
      2026-07-18 — inline-only extraction silently records think_block=None.

    Inline tags win when both are present. Returns (think_block | None,
    remaining_text_for_json_parse).
    """
    think, remainder = _extract_think_block(raw)
    if think is None and message_extra:
        for key in ("reasoning", "reasoning_content"):
            val = message_extra.get(key)
            if isinstance(val, str) and val.strip():
                think = val.strip()
                break
    return think, remainder


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    # Strip markdown code fences that some models emit
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: find the outermost {...} blob
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# Fragments of the schema's own field descriptions (prompts._JSON_BLOCK). When the
# model echoes the template instead of filling it in, the regex fallback happily
# extracts the DESCRIPTION as the reply and the orchestrator then feeds it to the
# opponent as conversational input. Observed at turn 1 of both 2026-07-18 pilot
# runs, which is the likely seed of the degenerate loop in those conversations.
_PLACEHOLDER_MARKERS = (
    "the words you literally speak",
    "your private detective notes",
    "exact quote or behaviour",
    "what you intend to probe",
    "knowledge_cutoff|sensory",
    "float 0.0",
    "true only when you are ready",
)


def _looks_like_placeholder(text: str) -> bool:
    """True if `text` is schema template text rather than model output."""
    if not text:
        return False
    stripped = text.strip()
    # Unfilled angle-bracket slot, e.g. "<the words you literally speak…>"
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    low = stripped.lower()
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


def _fallback_parse(text: str) -> TurnOutput:
    """Extract whatever fields are present; fill the rest with safe defaults.

    If the extracted reply is schema template text, the whole turn is marked
    "parse_failed": the reply is what crosses the channel, so a prompt-derived
    reply is worse than no reply at all.
    """
    reply_m  = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"',          text, re.DOTALL)
    score_m  = re.search(r'"suspicion_score"\s*:\s*(\d+(?:\.\d+)?)',     text)
    trace_m  = re.search(r'"reasoning_trace"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    accuse_m = re.search(r'"public_accusation"\s*:\s*(true|false)',       text, re.IGNORECASE)
    plan_m   = re.search(r'"plan"\s*:\s*"((?:[^"\\]|\\.)*)"',            text, re.DOTALL)
    type_m   = re.search(
        r'"type"\s*:\s*"(knowledge_cutoff|sensory|numeric|self_reference|none)"', text
    )
    reply = reply_m.group(1) if reply_m else text[:300].strip()

    if _looks_like_placeholder(reply):
        log.warning(
            "fallback parse recovered schema template text as the reply; "
            "marking turn parse_failed (reply=%r)", reply[:80],
        )
        return TurnOutput(
            reply="",
            suspicion_score=0.5,
            reasoning_trace="[parse_failed] model echoed the JSON template",
            cues=[],
            trap_strategy=TrapStrategy(plan="", type="none"),
            public_accusation=False,
            parse_mode="parse_failed",
        )

    return TurnOutput(
        reply=reply,
        suspicion_score=(
            max(0.0, min(1.0, float(score_m.group(1)))) if score_m else 0.5
        ),
        reasoning_trace=trace_m.group(1) if trace_m else "",
        cues=[],
        trap_strategy=TrapStrategy(
            plan=plan_m.group(1) if plan_m else "",
            type=type_m.group(1) if type_m else "none",
        ),
        public_accusation=accuse_m.group(1).lower() == "true" if accuse_m else False,
        parse_mode="fallback",
    )


def _dict_to_turn_output(d: dict) -> TurnOutput:
    ts = d.get("trap_strategy") or {}
    return TurnOutput(
        reply=str(d.get("reply", "")),
        suspicion_score=float(max(0.0, min(1.0, d.get("suspicion_score", 0.5)))),
        reasoning_trace=str(d.get("reasoning_trace", "")),
        cues=list(d.get("cues") or []),
        trap_strategy=TrapStrategy(
            plan=str(ts.get("plan", "")),
            type=str(ts.get("type", "none")),
        ),
        public_accusation=bool(d.get("public_accusation", False)),
    )


async def generate_turn(
    history:   list[dict],
    agent_cfg: AgentConfig,
    client:    AsyncOpenAI,
    seed:      int,
) -> tuple[TurnOutput, int, int, float, str | None, list[dict]]:
    """Returns (TurnOutput, prompt_tokens, gen_tokens, latency_ms, think_block, messages_input).

    think_block: raw <think>…</think> content extracted from R1-distill output; None for
    standard models. messages_input is the exact messages list sent to the API — persisted in
    TurnRecord to enable TransformerLens replay for post-hoc activation analysis.

    Passes guided_json to vLLM; Ollama ignores unknown body fields and relies on the
    system-prompt JSON instructions instead. Regex fallback runs if the model's text isn't
    valid JSON.
    """
    messages = _build_messages(history, agent_cfg)
    t0 = time.monotonic()

    try:
        resp = await client.chat.completions.create(
            model=agent_cfg.model_id,
            messages=messages,
            temperature=agent_cfg.temperature,
            seed=seed,
            # Repetition penalties counter the symmetric self-play attractor that
            # collapsed the 2026-07-18 pilot conversations into one repeated
            # sentence. frequency/presence are the OpenAI-compatible spelling;
            # repetition_penalty is vLLM's. Values live on AgentConfig because
            # they are part of the measurement instrument.
            frequency_penalty=agent_cfg.frequency_penalty,
            presence_penalty=agent_cfg.presence_penalty,
            response_format={"type": "json_object"},   # Ollama/OpenAI JSON mode
            extra_body={
                "guided_json": TURN_SCHEMA,             # vLLM schema enforcement
                "repetition_penalty": agent_cfg.repetition_penalty,
            },
        )
        latency_ms    = (time.monotonic() - t0) * 1000
        message       = resp.choices[0].message
        raw           = message.content or ""
        message_extra = getattr(message, "model_extra", None) or {}
        prompt_tokens = resp.usage.prompt_tokens     if resp.usage else 0
        gen_tokens    = resp.usage.completion_tokens if resp.usage else 0
    except Exception as exc:
        # Never let an API failure masquerade as a real turn: log it loudly and
        # mark the output so logging/metrics can distinguish it (parse_mode="api_error").
        latency_ms = (time.monotonic() - t0) * 1000
        log.error(
            "generate_turn API call failed (model=%s, endpoint=%s): %s: %s",
            agent_cfg.model_id, agent_cfg.endpoint, type(exc).__name__, exc,
        )
        return (
            TurnOutput(
                reply="",
                suspicion_score=0.5,
                reasoning_trace=f"[api_error] {type(exc).__name__}: {exc}",
                cues=[],
                trap_strategy=TrapStrategy(plan="", type="none"),
                public_accusation=False,
                parse_mode="api_error",
            ),
            0, 0, latency_ms, None, messages,
        )

    think_block, json_text = _resolve_think_block(raw, message_extra)
    parsed = _parse_json(json_text)
    output = _dict_to_turn_output(parsed) if parsed is not None else _fallback_parse(json_text)
    return output, prompt_tokens, gen_tokens, latency_ms, think_block, messages
