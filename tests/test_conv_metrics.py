"""
Tests for the conversation-metrics measurement instrument.

Pure logic — no network, no model. Covers the two measurement-validity fixes:
1. Think-block suspicion keywords must be word-bounded ("ai" must not fire
   inside "wait"/"said"/"again"; "bot" must not fire inside "both").
2. Turns with parse_mode == "api_error" carry fabricated-neutral values and
   must be excluded from all gap metrics.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "conversation"))

import conv_logging
from schema import TurnRecord


def _turn(turn_idx, speaker_id, suspicion=0.0, accuse=False, think=None, parse_mode="json"):
    return TurnRecord(
        conv_id="test",
        turn_idx=turn_idx,
        speaker_id=speaker_id,
        speaker_role="initiator" if speaker_id == "A" else "responder",
        model_id="test-model",
        adapter_id=None,
        prompt_tokens=0,
        gen_tokens=0,
        latency_ms=0.0,
        ground_truth_is_llm=True,
        reply="hello",
        suspicion_score=suspicion,
        reasoning_trace="",
        cues=[],
        trap_strategy={"plan": "", "type": "none"},
        public_accusation=accuse,
        think_block=think,
        messages_input=None,
        parse_mode=parse_mode,
    )


class TestThinkBlockSuspicion(unittest.TestCase):
    def test_common_words_do_not_fire(self):
        # Every one of these contains a suspicion keyword as a substring only.
        for text in [
            "Wait, I said we should maintain the plan again.",
            "They are both certain to explain the rain.",
            "A fair share of maintenance is automatic-ish.",
        ]:
            self.assertFalse(conv_logging._think_block_suspicious(text), text)

    def test_word_bounded_keywords_fire(self):
        for text in [
            "Could this be an AI?",
            "Their phrasing feels like a language model.",
            "I think I'm talking to a bot.",
            "Sounds like GPT to me.",
            "This is suspicious.",
            "They might be not a human at all.",
        ]:
            self.assertTrue(conv_logging._think_block_suspicious(text), text)

    def test_none_and_empty_blocks(self):
        self.assertFalse(conv_logging._think_block_suspicious(None))
        self.assertFalse(conv_logging._think_block_suspicious(""))


class TestMetricsApiErrorExclusion(unittest.TestCase):
    def test_api_error_turn_does_not_break_sustained_suspicion(self):
        # A: 0.8, [api_error 0.5], 0.9 — sustained >= 0.7 should hold from turn 0.
        turns = [
            _turn(0, "A", suspicion=0.8),
            _turn(1, "B", suspicion=0.1),
            _turn(2, "A", suspicion=0.5, parse_mode="api_error"),
            _turn(3, "B", suspicion=0.1),
            _turn(4, "A", suspicion=0.9, accuse=True),
        ]
        metrics = conv_logging.compute_conversation_metrics(turns)
        self.assertEqual(metrics["A"]["t_private_07"], 0)
        self.assertEqual(metrics["A"]["t_public"], 4)
        self.assertEqual(metrics["A"]["commitment_gap"], 4)

    def test_api_error_think_block_does_not_set_t_think(self):
        turns = [
            _turn(0, "A", suspicion=0.5, think="probably an AI", parse_mode="api_error"),
            _turn(2, "A", suspicion=0.8, think="just chatting about the weather"),
        ]
        metrics = conv_logging.compute_conversation_metrics(turns)
        self.assertIsNone(metrics["A"]["t_think_07"])

    def test_sustained_check_still_rejects_dips(self):
        # 0.75 then a genuine dip to 0.6 — first 0.75 must NOT count as sustained.
        turns = [
            _turn(0, "A", suspicion=0.75),
            _turn(2, "A", suspicion=0.6),
            _turn(4, "A", suspicion=0.8),
        ]
        metrics = conv_logging.compute_conversation_metrics(turns)
        self.assertEqual(metrics["A"]["t_private_07"], 4)


class TestParseModeDefaults(unittest.TestCase):
    def test_default_parse_mode_is_json(self):
        self.assertEqual(_turn(0, "A").parse_mode, "json")


class TestThinkTransport(unittest.TestCase):
    """Think content arrives inline (<think> tags) or via a separate
    reasoning field depending on the serving stack — both must resolve."""

    def _agent(self):
        import agent
        return agent

    def test_inline_tags_extracted_and_stripped(self):
        agent = self._agent()
        think, rest = agent._resolve_think_block(
            "<think>they seem robotic</think>{\"reply\": \"hi\"}", {})
        self.assertEqual(think, "they seem robotic")
        self.assertEqual(rest, '{"reply": "hi"}')

    def test_reasoning_extra_field_fallback(self):
        agent = self._agent()
        think, rest = agent._resolve_think_block(
            '{"reply": "hi"}', {"reasoning": " hmm, suspicious phrasing "})
        self.assertEqual(think, "hmm, suspicious phrasing")
        self.assertEqual(rest, '{"reply": "hi"}')

    def test_reasoning_content_key_supported(self):
        agent = self._agent()
        think, _ = agent._resolve_think_block(
            '{"reply": "hi"}', {"reasoning_content": "vllm parser output"})
        self.assertEqual(think, "vllm parser output")

    def test_inline_wins_over_extra(self):
        agent = self._agent()
        think, _ = agent._resolve_think_block(
            "<think>inline</think>{}", {"reasoning": "extra"})
        self.assertEqual(think, "inline")

    def test_absent_everywhere_is_none(self):
        agent = self._agent()
        think, rest = agent._resolve_think_block('{"reply": "hi"}', {})
        self.assertIsNone(think)
        self.assertEqual(rest, '{"reply": "hi"}')
        think, _ = agent._resolve_think_block('{}', {"reasoning": "   "})
        self.assertIsNone(think)


if __name__ == "__main__":
    unittest.main()
