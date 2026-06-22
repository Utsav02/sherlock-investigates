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


@dataclass
class AgentConfig:
    model_id:            str
    endpoint:            str
    api_key:             str        = "ollama"
    adapter_id:          str | None = None
    role:                str        = "initiator"   # "initiator" | "responder"
    ground_truth_is_llm: bool       = True


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


@dataclass
class ConversationRecord:
    conv_id:           str
    agent_A_cfg:       dict
    agent_B_cfg:       dict
    n_turns:           int
    winner:            str | None   # "A" | "B" | None
    termination_reason: str         # "accusation" | "max_turns"
    A_correct:         bool | None
    B_correct:         bool | None
    t_private_07:      int | None   # winner's first turn where suspicion_score stayed >= 0.7
    t_public:          int | None   # winner's turn where public_accusation fired
    commitment_gap:    int | None   # t_public - t_private_07
    seed:              int


@dataclass
class ConversationResult:
    record: ConversationRecord
    turns:  list[TurnRecord]
