"""JSON schema and dataclasses for per-turn structured output and conversation records."""
from dataclasses import dataclass

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "reply":           {"type": "string"},
        "suspicion_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning_trace": {"type": "string"},
        "cues":            {"type": "array", "items": {"type": "string"}},
        "trap_strategy": {
            "type": "object",
            "properties": {
                "plan": {"type": "string"},
                "type": {
                    "type": "string",
                    "enum": ["knowledge_cutoff", "sensory", "numeric", "self_reference", "none"],
                },
            },
            "required": ["plan", "type"],
        },
        "public_accusation": {"type": "boolean"},
    },
    "required": [
        "reply",
        "suspicion_score",
        "reasoning_trace",
        "cues",
        "trap_strategy",
        "public_accusation",
    ],
}


@dataclass
class TrapStrategy:
    plan: str
    type: str


@dataclass
class TurnOutput:
    reply:             str
    suspicion_score:   float
    reasoning_trace:   str
    cues:              list[str]
    trap_strategy:     TrapStrategy
    public_accusation: bool
    # provenance: "json" (schema-valid parse), "fallback" (regex extraction from
    # malformed output), "api_error" (call failed — all fields are neutral filler),
    # "parse_failed" (regex extraction returned prompt text, not model output —
    # all fields are unusable; see agent._looks_like_placeholder)
    parse_mode:        str = "json"


# parse_modes whose field values are not model behaviour. These turns must be
# excluded from every metric AND must never enter the opponent's history —
# a fabricated or prompt-derived reply poisons the other agent's context.
UNUSABLE_PARSE_MODES = frozenset({"api_error", "parse_failed"})


@dataclass
class AgentConfig:
    model_id:            str
    endpoint:            str
    api_key:             str        = "ollama"
    adapter_id:          str | None = None
    role:                str        = "initiator"   # "initiator" | "responder"
    ground_truth_is_llm: bool       = True
    thinking_mode:       bool       = False         # True for DeepSeek-R1-distill models
    # Sampling params are part of the measurement instrument — changing them
    # changes results; log any change in the CLAUDE.md Decision Log.
    temperature:         float      = 0.7
    frequency_penalty:   float      = 0.3   # OpenAI/Ollama-compatible
    presence_penalty:    float      = 0.3
    repetition_penalty:  float      = 1.1   # vLLM (sent via extra_body)
    # Light persona text appended to the system prompt. Breaks the symmetric
    # self-play attractor that collapses same-model conversations into a single
    # repeated utterance (see Decision Log 2026-07-26).
    persona:             str        = ""


@dataclass
class ConversationConfig:
    agent_A:      AgentConfig
    agent_B:      AgentConfig
    max_turns:    int = 12
    seed:         int = 42
    ablation_type: str = "none"
    conv_id:      str = ""


@dataclass
class TurnRecord:
    # metadata
    conv_id:             str
    turn_idx:            int
    speaker_id:          str        # "A" | "B"
    speaker_role:        str        # "initiator" | "responder"
    model_id:            str
    adapter_id:          str | None
    prompt_tokens:       int
    gen_tokens:          int
    latency_ms:          float
    ground_truth_is_llm: bool
    # per-turn output
    reply:               str
    suspicion_score:     float
    reasoning_trace:     str
    cues:                list[str]
    trap_strategy:       dict       # {"plan": str, "type": str}
    public_accusation:   bool
    # interpretability fields (None for non-thinking models)
    think_block:         str | None = None   # raw <think>…</think> content from R1-distill
    messages_input:      list | None = None  # exact messages list sent to API — enables TransformerLens replay
    # provenance of the structured fields — see TurnOutput.parse_mode.
    # "api_error" turns carry fabricated-neutral values and are excluded from metrics.
    parse_mode:          str = "json"


@dataclass
class ConversationRecord:
    conv_id:             str
    agent_A_cfg:         dict
    agent_B_cfg:         dict
    n_turns:             int
    winner:              str | None   # "A" | "B" | None
    termination_reason:  str          # "accusation" | "max_turns" | "degenerate"
    A_correct:           bool | None
    B_correct:           bool | None
    t_private_07:        int | None   # winner's first turn where suspicion_score stayed >= 0.7
    t_public:            int | None   # winner's turn where public_accusation fired
    commitment_gap:      int | None   # t_public - t_private_07  (behavioral gap)
    seed:                int
    # three-level commitment gap fields (populated only for thinking models)
    t_think_07:          int | None = None  # first turn of DIRECTED suspicion in think_block
    think_commitment_gap: int | None = None  # t_private_07 - t_think_07  (think→score gap)
    # Superseded topic-mention measure, retained so analysis can quantify how much
    # the 2026-07-26 directed-suspicion redefinition changed the metric. NOT the
    # headline number — see conv_logging._think_block_mentions_ai.
    t_think_topic:       int | None = None
    # Degeneracy diagnostics. Same-model self-play collapses into a repeated
    # utterance; such conversations carry no accumulating evidence and every
    # downstream gap metric is meaningless on them.
    unique_reply_ratio:      float = 1.0   # distinct replies / usable turns
    max_consecutive_repeats: int   = 1     # longest run of identical consecutive replies
    is_degenerate:           bool  = False


@dataclass
class ConversationResult:
    record: ConversationRecord
    turns:  list[TurnRecord]
