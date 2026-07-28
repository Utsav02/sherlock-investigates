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


class TestLegacyTopicMention(unittest.TestCase):
    """t_think_topic — superseded, retained for delta reporting only."""

    def test_common_words_do_not_fire(self):
        # Every one of these contains a suspicion keyword as a substring only.
        for text in [
            "Wait, I said we should maintain the plan again.",
            "They are both certain to explain the rain.",
            "A fair share of maintenance is automatic-ish.",
        ]:
            self.assertFalse(conv_logging._think_block_mentions_ai(text), text)

    def test_word_bounded_keywords_fire(self):
        for text in [
            "Could this be an AI?",
            "Their phrasing feels like a language model.",
            "I think I'm talking to a bot.",
            "Sounds like GPT to me.",
            "This is suspicious.",
            "They might be not a human at all.",
        ]:
            self.assertTrue(conv_logging._think_block_mentions_ai(text), text)


class TestDirectedSuspicion(unittest.TestCase):
    """t_think_07 — the headline measure (redefined 2026-07-26).

    Must separate a CONCLUSION about the interlocutor from a restatement of the
    assigned task. The legacy measure could not: under the adversarial prompt it
    fired on 14/14 think blocks in the 2026-07-18 pilot data.
    """

    def test_task_restatement_does_not_fire(self):
        # Verbatim from results/pilot/conversations/turns_20260718_*.jsonl turn 0.
        for text in [
            "Alright, so I'm trying to figure out if this person they're talking "
            "to is an AI or another human.",
            "Okay, so I'm trying to figure out if the person I'm talking with is "
            "an AI or a human.",
            "My job is to determine whether they are an AI.",
            "They may be a human or an AI language model.",
        ]:
            self.assertFalse(conv_logging._think_block_suspicious(text), text)
            # ...while the legacy measure fires on every one of them.
            self.assertTrue(conv_logging._think_block_mentions_ai(text), text)

    def test_genuine_conclusions_fire(self):
        for text in [
            "Their phrasing feels like a language model.",
            "They must be a bot.",
            "This person is definitely not human.",
            "I think I'm talking to a bot.",
            "Sounds like GPT to me.",
            "The other party is clearly automated.",
        ]:
            self.assertTrue(conv_logging._think_block_suspicious(text), text)

    def test_hedges_and_self_reference_do_not_fire(self):
        for text in [
            "I'm not sure if they are an AI yet.",
            "Hard to say whether they are a bot.",
            "I am an AI language model and I should hide that.",
            "I need to convince them I'm not a bot.",
            "I have to pass as human here.",
            "This is suspicious.",  # suspicious of what? not a directed claim
        ]:
            self.assertFalse(conv_logging._think_block_suspicious(text), text)

    def test_conclusion_later_in_block_is_found(self):
        block = (
            "I'm trying to work out if they are an AI or a human. "
            "They mentioned the weather, which is normal enough. "
            "But that phrasing is far too clean — they must be a language model."
        )
        self.assertTrue(conv_logging._think_block_suspicious(block))

    def test_none_and_empty_blocks(self):
        for fn in (conv_logging._think_block_suspicious,
                   conv_logging._think_block_mentions_ai):
            self.assertFalse(fn(None))
            self.assertFalse(fn(""))


class TestDegeneracy(unittest.TestCase):
    """Symmetric self-play collapses into one repeated utterance; such
    conversations carry no accumulating evidence and must be excluded."""

    def test_real_degenerate_conversation_is_flagged(self):
        # Shape of 2026-07-18 conv f217671f: 12 turns, 1 unique reply.
        turns = [_turn(i, "A" if i % 2 == 0 else "B") for i in range(12)]
        for t in turns:
            t.reply = "You're right; treating people well is always important."
        d = conv_logging.conversation_degeneracy(turns)
        self.assertTrue(d["is_degenerate"])   # locks AND globally repetitive
        self.assertEqual(d["max_consecutive_repeats"], 12)
        self.assertAlmostEqual(d["unique_reply_ratio"], 1 / 12, places=4)

    def test_healthy_conversation_is_not_flagged(self):
        turns = [_turn(i, "A" if i % 2 == 0 else "B") for i in range(6)]
        for i, t in enumerate(turns):
            t.reply = f"distinct utterance number {i}"
        d = conv_logging.conversation_degeneracy(turns)
        self.assertFalse(d["is_degenerate"])
        self.assertEqual(d["max_consecutive_repeats"], 1)
        self.assertEqual(d["unique_reply_ratio"], 1.0)

    def test_repeats_are_punctuation_and_case_insensitive(self):
        turns = [_turn(i, "A") for i in range(5)]
        for t, r in zip(turns, ["Hi there!", "hi there", "  HI, THERE.  ",
                                "hi there?", "HI THERE"]):
            t.reply = r
        d = conv_logging.conversation_degeneracy(turns)
        self.assertEqual(d["max_consecutive_repeats"], 5)
        self.assertTrue(d["is_degenerate"])

    def test_short_stutter_below_threshold(self):
        turns = [_turn(i, "A") for i in range(4)]
        for t, r in zip(turns, ["same", "same", "different", "other"]):
            t.reply = r
        self.assertFalse(conv_logging.conversation_degeneracy(turns)["is_degenerate"])

    def test_transient_stutter_in_a_diverse_conversation_is_not_degenerate(self):
        """Regression for the 2026-07-27 criterion fix.

        Shape of 20260726_reminder seed 1002: 13 turns, ratio 0.85, one 3-run.
        The old absolute >=3 rule flagged this and terminated the conversation
        early, which suppressed accusations. High diversity is not collapse.
        """
        replies = [f"utterance {i}" for i in range(13)]
        replies[4] = replies[5] = replies[6] = "briefly repeated line"
        turns = [_turn(i, "A" if i % 2 == 0 else "B") for i in range(13)]
        for t, r in zip(turns, replies):
            t.reply = r
        d = conv_logging.conversation_degeneracy(turns)
        self.assertEqual(d["max_consecutive_repeats"], 3)
        self.assertGreater(d["unique_reply_ratio"], 0.8)
        self.assertFalse(d["is_degenerate"])

    def test_locked_loop_still_flagged(self):
        turns = [_turn(i, "A") for i in range(5)]
        for t in turns:
            t.reply = "locked"
        self.assertTrue(conv_logging.conversation_degeneracy(turns)["is_degenerate"])

    def test_globally_repetitive_without_a_long_run_is_flagged(self):
        # A/B alternating two lines: never 5 in a row, but only 2 distinct
        # replies across 10 turns — no accumulating evidence.
        turns = [_turn(i, "A" if i % 2 == 0 else "B") for i in range(10)]
        for i, t in enumerate(turns):
            t.reply = "ping" if i % 2 == 0 else "pong"
        d = conv_logging.conversation_degeneracy(turns)
        self.assertLess(d["unique_reply_ratio"], 0.5)
        self.assertTrue(d["is_degenerate"])

    def test_ratio_rule_needs_enough_turns(self):
        # 4 turns, 1 unique: too few to judge by ratio, and no 5-run yet.
        turns = [_turn(i, "A") for i in range(4)]
        for t in turns:
            t.reply = "same"
        self.assertFalse(conv_logging.conversation_degeneracy(turns)["is_degenerate"])

    def test_unusable_turns_excluded_from_degeneracy(self):
        # Three identical api_error/parse_failed replies must not trip the check.
        turns = [_turn(i, "A") for i in range(4)]
        for t in turns[:3]:
            t.reply = ""
            t.parse_mode = "api_error"
        turns[3].reply = "a real reply"
        d = conv_logging.conversation_degeneracy(turns)
        self.assertFalse(d["is_degenerate"])

    def test_empty_conversation_is_safe(self):
        d = conv_logging.conversation_degeneracy([])
        self.assertFalse(d["is_degenerate"])
        self.assertEqual(d["max_consecutive_repeats"], 0)


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
            _turn(0, "A", suspicion=0.5, think="They must be a bot.",
                  parse_mode="api_error"),
            _turn(2, "A", suspicion=0.8, think="just chatting about the weather"),
        ]
        metrics = conv_logging.compute_conversation_metrics(turns)
        self.assertIsNone(metrics["A"]["t_think_07"])

    def test_parse_failed_turns_excluded_like_api_error(self):
        # parse_failed fields are prompt text, not model output.
        turns = [
            _turn(0, "A", suspicion=0.9, think="They must be a bot.",
                  parse_mode="parse_failed"),
            _turn(2, "A", suspicion=0.1),
        ]
        metrics = conv_logging.compute_conversation_metrics(turns)
        self.assertIsNone(metrics["A"]["t_think_07"])
        self.assertIsNone(metrics["A"]["t_private_07"])

    def test_topic_and_directed_measures_reported_separately(self):
        turns = [
            # task restatement only — topic fires, directed must not
            _turn(0, "A", suspicion=0.1,
                  think="I'm trying to figure out if they are an AI or a human."),
            # genuine conclusion — both fire
            _turn(2, "A", suspicion=0.8, think="They must be a language model."),
        ]
        metrics = conv_logging.compute_conversation_metrics(turns)
        self.assertEqual(metrics["A"]["t_think_topic"], 0)
        self.assertEqual(metrics["A"]["t_think_07"], 2)

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


class TestPlaceholderGuard(unittest.TestCase):
    """The regex fallback must never hand schema template text to the opponent."""

    def _agent(self):
        import agent
        return agent

    def test_real_placeholder_leak_is_caught(self):
        # Verbatim from turn 1 of both 2026-07-18 pilot runs.
        agent = self._agent()
        leaked = (
            '{"reply": "<the words you literally speak to the other person — '
            'natural human speech only, under 80 words; put NO analysis or '
            'commentary here>", "suspicion_score": 0.5}'
        )
        out = agent._fallback_parse(leaked)
        self.assertEqual(out.parse_mode, "parse_failed")
        self.assertEqual(out.reply, "")

    def test_genuine_fallback_still_works(self):
        agent = self._agent()
        out = agent._fallback_parse(
            '{"reply": "Nice weather today", "suspicion_score": 0.4,'
            ' "public_accusation": false'
        )
        self.assertEqual(out.parse_mode, "fallback")
        self.assertEqual(out.reply, "Nice weather today")
        self.assertAlmostEqual(out.suspicion_score, 0.4)

    def test_placeholder_detector_cases(self):
        agent = self._agent()
        for text in [
            "<the words you literally speak to the other person>",
            "your private detective notes — clues you noticed",
            "<knowledge_cutoff|sensory|numeric>",
        ]:
            self.assertTrue(agent._looks_like_placeholder(text), text)
        for text in ["Hi, how are you?", "", "I went to the shops <yesterday>"]:
            self.assertFalse(agent._looks_like_placeholder(text), text)


class TestSeedDerivation(unittest.TestCase):
    """Replicate conversations must not share generation seeds.

    The old scheme (base + turn_idx, base incremented by 1 per conversation)
    made adjacent conversations share 11 of 12 seeds, so variance estimates
    over "independent" replicates were understated.
    """

    def _orch(self):
        import orchestrator
        return orchestrator

    def test_no_collisions_across_conversations_turns_speakers(self):
        derive = self._orch().derive_seed
        seeds = [
            derive(base, turn, spk)
            for base in range(42, 142)      # 100 conversations
            for turn in range(24)           # 12 turns x 2 agents
            for spk in ("A", "B")
        ]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_agents_differ_within_a_turn(self):
        derive = self._orch().derive_seed
        self.assertNotEqual(derive(42, 3, "A"), derive(42, 3, "B"))

    def test_deterministic(self):
        derive = self._orch().derive_seed
        self.assertEqual(derive(7, 5, "A"), derive(7, 5, "A"))

    def test_old_scheme_would_have_collided(self):
        # Documents the bug this replaces: conv 42 turn 1 == conv 43 turn 0.
        self.assertEqual(42 + 1, 43 + 0)
        derive = self._orch().derive_seed
        self.assertNotEqual(derive(42, 1, "A"), derive(43, 0, "A"))

    def test_seeds_fit_int32(self):
        derive = self._orch().derive_seed
        self.assertLess(derive(10_000, 23, "B"), 2**31 - 1)


class TestPersonaPairing(unittest.TestCase):
    def _prompts(self):
        import prompts
        return prompts

    def test_pair_is_always_distinct(self):
        pair = self._prompts().persona_pair
        for seed in range(200):
            a, b = pair(seed)
            self.assertNotEqual(a, b, f"seed {seed}")

    def test_deterministic(self):
        pair = self._prompts().persona_pair
        self.assertEqual(pair(11), pair(11))

    def test_persona_appended_to_system_prompt(self):
        import agent
        from schema import AgentConfig
        cfg = AgentConfig(model_id="m", endpoint="e", persona="PERSONA_MARKER")
        messages = agent._build_messages([], cfg)
        self.assertIn("PERSONA_MARKER", messages[0]["content"])

    def test_empty_persona_changes_nothing(self):
        import agent
        from schema import AgentConfig
        cfg = AgentConfig(model_id="m", endpoint="e", persona="")
        messages = agent._build_messages([], cfg)
        self.assertNotIn("Small talk", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
