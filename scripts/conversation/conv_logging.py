"""Append-only JSONL logging for turns and conversations, and conversation metrics."""
import dataclasses
import json
from pathlib import Path

from schema import ConversationRecord, TurnRecord


def log_turn(record: TurnRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def log_conversation(record: ConversationRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def compute_conversation_metrics(turns: list[TurnRecord]) -> dict:
    """Returns {"A": {...}, "B": {...}} with t_private_07, t_public, commitment_gap per speaker.

    t_private_07: lowest turn_idx where the speaker's suspicion_score first reached >= 0.7
    and every subsequent turn for that speaker also stayed >= 0.7.  None if never sustained.
    """
    result: dict[str, dict] = {}

    for speaker_id in ("A", "B"):
        speaker_turns = [(t.turn_idx, t) for t in turns if t.speaker_id == speaker_id]

        t_private: int | None = None
        for i, (idx, t) in enumerate(speaker_turns):
            if t.suspicion_score >= 0.7:
                if all(st.suspicion_score >= 0.7 for _, st in speaker_turns[i:]):
                    t_private = idx
                    break

        t_public: int | None = next(
            (idx for idx, t in speaker_turns if t.public_accusation), None
        )

        gap = (t_public - t_private) if (t_public is not None and t_private is not None) else None

        result[speaker_id] = {
            "t_private_07":   t_private,
            "t_public":       t_public,
            "commitment_gap": gap,
        }

    return result
