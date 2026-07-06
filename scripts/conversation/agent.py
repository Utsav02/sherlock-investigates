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


_JSON_REMINDER = "\n[Respond with a JSON object only — no other text.]"
_JSON_REMINDER_THINKING = "\n[After your thinking, respond with a JSON object only — no other text outside your thinking block.]"


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

    messages: list[dict] = [{"role": "system", "content": system}]
    for msg in history:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["content"] + reminder})
        else:
            messages.append(msg)
    return messages


def _extract_think_block(text: str) -> tuple[str | None, str]:
    """Strip <think>…</think> from raw R1-distill output.

    Returns (think_block_content, remainder). If no think block, returns (None, text).
    """
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if m:
        think_block = m.group(1).strip()
        remainder = (text[: m.start()] + text[m.end() :]).strip()
        return think_block, remainder
    return None, text


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


def _fallback_parse(text: str) -> TurnOutput:
    """Extract whatever fields are present; fill the rest with safe defaults."""
    reply_m  = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"',          text, re.DOTALL)
    score_m  = re.search(r'"suspicion_score"\s*:\s*(\d+(?:\.\d+)?)',     text)
    trace_m  = re.search(r'"reasoning_trace"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    accuse_m = re.search(r'"public_accusation"\s*:\s*(true|false)',       text, re.IGNORECASE)
    plan_m   = re.search(r'"plan"\s*:\s*"((?:[^"\\]|\\.)*)"',            text, re.DOTALL)
    type_m   = re.search(
        r'"type"\s*:\s*"(knowledge_cutoff|sensory|numeric|self_reference|none)"', text
    )
    return TurnOutput(
        reply=reply_m.group(1) if reply_m else text[:300].strip(),
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
            temperature=0.7,
            seed=seed,
            response_format={"type": "json_object"},   # Ollama/OpenAI JSON mode
            extra_body={"guided_json": TURN_SCHEMA},    # vLLM schema enforcement
        )
        latency_ms    = (time.monotonic() - t0) * 1000
        raw           = resp.choices[0].message.content or ""
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

    think_block, json_text = _extract_think_block(raw)
    parsed = _parse_json(json_text)
    output = _dict_to_turn_output(parsed) if parsed is not None else _fallback_parse(json_text)
    return output, prompt_tokens, gen_tokens, latency_ms, think_block, messages
