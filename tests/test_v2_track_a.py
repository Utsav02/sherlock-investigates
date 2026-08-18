#!/usr/bin/env python3
"""
Unit tests for the Track A arm-A0 machinery and the canonical loader's pure
helpers. No network, no source data, no model inference — every test below runs
on constructed inputs so `make test` stays offline.

The statistical helpers are checked against hand-computable values rather than
against their own output, because the failure this repo keeps re-learning is an
instrument that is plausible and wrong (`t_think_07`, precision 0.185).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "v2" / "scripts"))

import build_canonical  # noqa: E402
import canonical_policy  # noqa: E402
import track_a_a0 as a0  # noqa: E402


def dialogue(label: str, is_human: bool, messages, changed=None):
    """Minimal canonical-shaped dialogue row."""
    changed = changed or [False] * len(messages)
    return {
        "example_id": f"x-{label}-{'h' if is_human else 'a'}",
        "conversation_label": label,
        "is_human": is_human,
        "turns": [
            {"role": "W", "content": m, "is_changed": c, "timestamp": ""}
            for m, c in zip(messages, changed)
        ],
    }


# ---------------------------------------------------------------------------
# policy enforcement
# ---------------------------------------------------------------------------

class TestPolicyEnforcement(unittest.TestCase):
    def test_loader_never_keeps_tt_profile_other(self):
        """The excluded column must be absent from the loader's kept list."""
        self.assertNotIn("other", build_canonical.KEPT_COLUMNS["tt_profile"])

    def test_policy_check_rejects_it_if_someone_adds_it_back(self):
        with self.assertRaises(canonical_policy.ExcludedColumnError):
            canonical_policy.check_columns("tt_profile", ["user_id", "other"])

    def test_every_kept_column_list_passes_the_policy(self):
        for table, columns in build_canonical.KEPT_COLUMNS.items():
            canonical_policy.check_columns(table, columns)   # must not raise


# ---------------------------------------------------------------------------
# text variants
# ---------------------------------------------------------------------------

class TestTextVariants(unittest.TestCase):
    def test_strip_placeholders_removes_both_bracket_styles(self):
        self.assertEqual(
            build_canonical.strip_placeholders("hi <NAME> from [LOCATION] ok"),
            "hi from ok",
        )

    def test_strip_placeholders_leaves_ordinary_text(self):
        self.assertEqual(build_canonical.strip_placeholders("a < b and B > c"),
                         "a < b and B > c")

    def test_collapse_ws(self):
        self.assertEqual(build_canonical.collapse_ws(" a \n\n b\tc "), "a b c")

    def test_variants_strip_progressively(self):
        d = dialogue("A", True, ["hi <NAME>", "rewritten bit"], changed=[False, True])
        self.assertEqual(a0.dialogue_messages(d, "raw"), ["hi <NAME>", "rewritten bit"])
        self.assertEqual(a0.dialogue_messages(d, "nostub"), ["hi", "rewritten bit"])
        self.assertEqual(a0.dialogue_messages(d, "nostub_nochanged"), ["hi"])

    def test_only_witness_turns_are_used(self):
        d = dialogue("A", True, ["witness text"])
        d["turns"].append({"role": "I", "content": "interrogator text",
                           "is_changed": False, "timestamp": ""})
        self.assertEqual(a0.dialogue_messages(d, "raw"), ["witness text"])


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

class TestFeatures(unittest.TestCase):
    def test_length_features_on_known_input(self):
        f = a0.length_features(["abcd", "ab"])
        self.assertEqual(f["n_messages"], 2.0)
        self.assertEqual(f["total_chars"], 6.0)
        self.assertEqual(f["mean_chars"], 3.0)
        self.assertEqual(f["max_chars"], 4.0)
        self.assertEqual(f["empty"], 0.0)

    def test_empty_dialogue_is_flagged_not_crashing(self):
        for fn in (a0.length_features, a0.punctuation_features,
                   a0.function_word_features):
            f = fn([])
            self.assertEqual(f["empty"], 1.0, fn.__name__)

    def test_punctuation_rate_is_per_1000_chars(self):
        f = a0.punctuation_features(["a.b.c.d."])          # 8 chars, 4 periods
        self.assertAlmostEqual(f["punct_period"], 500.0)

    def test_function_word_rate(self):
        f = a0.function_word_features(["the cat the dog"])  # 4 tokens, 'the' x2
        self.assertAlmostEqual(f["fw_the"], 500.0)

    def test_tfidf_terms_include_bigrams(self):
        self.assertEqual(a0.tfidf_terms(["a b"]), ["a", "b", "a_b"])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):
    def test_auroc_perfect_and_inverted(self):
        self.assertAlmostEqual(a0.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(a0.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_auroc_all_ties_is_one_half(self):
        self.assertAlmostEqual(a0.auroc([0.5] * 4, [0, 1, 0, 1]), 0.5)

    def test_auroc_hand_computed(self):
        # one positive above one negative, one tie -> (1 + 0.5) / 2 = 0.75
        self.assertAlmostEqual(a0.auroc([0.2, 0.6, 0.6], [0, 1, 0]), 0.75)

    def test_brier_and_log_loss(self):
        self.assertAlmostEqual(a0.brier([1.0, 0.0], [1, 0]), 0.0)
        self.assertAlmostEqual(a0.brier([0.5, 0.5], [1, 0]), 0.25)
        self.assertGreater(a0.log_loss([0.5, 0.5], [1, 0]), 0.69)

    def test_ece_is_zero_for_a_perfectly_calibrated_set(self):
        probs = [0.25] * 4 + [0.75] * 4
        labels = [1, 0, 0, 0, 1, 1, 1, 0]
        self.assertAlmostEqual(a0.ece(probs, labels), 0.0, places=9)

    def test_reliability_bins_sum_to_n(self):
        rows = a0.reliability([0.05, 0.15, 0.95], [0, 1, 1])
        self.assertEqual(sum(r["n"] for r in rows), 3)

    def test_mcnemar_matches_textbook_values(self):
        self.assertAlmostEqual(a0.mcnemar_exact(0, 0), 1.0)
        # b=0, c=5 -> 2 * (1/2)^5 = 0.0625
        self.assertAlmostEqual(a0.mcnemar_exact(0, 5), 0.0625)
        self.assertAlmostEqual(a0.mcnemar_exact(5, 5), 1.0)


# ---------------------------------------------------------------------------
# logistic regression
# ---------------------------------------------------------------------------

class TestLogistic(unittest.TestCase):
    def test_sigmoid_is_stable_at_extremes(self):
        self.assertAlmostEqual(a0.sigmoid(0.0), 0.5)
        self.assertAlmostEqual(a0.sigmoid(800.0), 1.0)
        self.assertAlmostEqual(a0.sigmoid(-800.0), 0.0)

    def test_solve_recovers_a_known_solution(self):
        x = a0.solve([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
        self.assertAlmostEqual(x[0], 1.0, places=9)
        self.assertAlmostEqual(x[1], 3.0, places=9)

    def test_dense_logistic_separates_a_separable_set(self):
        X = [[-2.0], [-1.0], [1.0], [2.0]]
        y = [0, 0, 1, 1]
        beta = a0.fit_dense_logistic(X, y, l2=1e-3)
        self.assertLess(a0.predict_dense(beta, [-2.0]), 0.2)
        self.assertGreater(a0.predict_dense(beta, [2.0]), 0.8)

    def test_dense_logistic_ridge_shrinks_weights(self):
        X = [[-2.0], [-1.0], [1.0], [2.0]]
        y = [0, 0, 1, 1]
        weak = a0.fit_dense_logistic(X, y, l2=1e-3)[1]
        strong = a0.fit_dense_logistic(X, y, l2=100.0)[1]
        self.assertLess(abs(strong), abs(weak))

    def test_sparse_logistic_separates_and_reports_convergence(self):
        rows = [[(0, 1.0)], [(0, 1.0)], [(1, 1.0)], [(1, 1.0)]]
        y = [0, 0, 1, 1]
        w, bias, diag = a0.fit_sparse_logistic(rows, y, dim=2, l2=1e-3)
        self.assertLess(a0.sigmoid(bias + w[0]), 0.4)
        self.assertGreater(a0.sigmoid(bias + w[1]), 0.6)
        self.assertIn("iterations", diag)
        self.assertGreater(diag["iterations"], 0)


# ---------------------------------------------------------------------------
# scoring and baselines
# ---------------------------------------------------------------------------

class TestScoring(unittest.TestCase):
    def setUp(self):
        self.games = [
            {"game_id": "1", "human_conversation_label": "A", "component": 0,
             "interrogator_user_id": "u1", "human_witness_user_id": "u2",
             "witness_system": "s", "empty_side": False, "split": "train"},
            {"game_id": "2", "human_conversation_label": "B", "component": 0,
             "interrogator_user_id": "u1", "human_witness_user_id": "u3",
             "witness_system": "s", "empty_side": False, "split": "train"},
        ]
        self.by_game = {
            "1": {"A": dialogue("A", True, ["h"]), "B": dialogue("B", False, ["a"])},
            "2": {"A": dialogue("A", False, ["a"]), "B": dialogue("B", True, ["h"])},
        }
        for gid, sides in self.by_game.items():
            for lbl, d in sides.items():
                d["example_id"] = f"{gid}-{lbl}"

    def test_perfect_predictions_score_one(self):
        probs = {"1-A": 0.0, "1-B": 1.0, "2-A": 1.0, "2-B": 0.0}
        s = a0.score_games(probs, self.games, self.by_game)
        self.assertEqual(s["game_correct"], [1.0, 1.0])
        self.assertEqual(s["ties"], 0)

    def test_constant_predictions_tie_at_half_credit(self):
        probs = {k: 0.5 for k in ("1-A", "1-B", "2-A", "2-B")}
        s = a0.score_games(probs, self.games, self.by_game)
        self.assertEqual(s["game_correct"], [0.5, 0.5])
        self.assertEqual(s["ties"], 2)

    def test_tie_break_applies_the_position_rule(self):
        probs = {k: 0.5 for k in ("1-A", "1-B", "2-A", "2-B")}
        s = a0.score_games(probs, self.games, self.by_game, tie_break="A")
        # game 1's human sits in A (hit), game 2's sits in B (miss)
        self.assertEqual(s["game_correct"], [1.0, 0.0])

    def test_dialogue_probs_stay_aligned_with_game_order(self):
        probs = {"1-A": 0.1, "1-B": 0.9, "2-A": 0.8, "2-B": 0.2}
        s = a0.score_games(probs, self.games, self.by_game)
        self.assertEqual(s["dialogue_probs"], [0.1, 0.9, 0.8, 0.2])
        self.assertEqual(s["dialogue_labels"], [0, 1, 1, 0])
        # metrics_from must slice dialogues 2i, 2i+1 for game i
        m = a0.metrics_from(s, [0])
        self.assertAlmostEqual(m["dialogue_brier"], (0.1 ** 2 + 0.1 ** 2) / 2)

    def test_majority_baseline_emits_a_constant_and_a_tie_break(self):
        train = [dialogue("A", True, ["x"]), dialogue("B", False, ["y"]),
                 dialogue("A", True, ["z"]), dialogue("B", False, ["w"])]
        det = a0.MajorityDetector()
        det.fit(train, "raw")
        self.assertEqual(det.predict(train, "raw"), [0.5] * 4)
        self.assertEqual(det.tie_break, "A")

    def test_random_baseline_has_no_tie_break(self):
        det = a0.ConstantDetector()
        det.fit([], "raw")
        self.assertIsNone(det.tie_break)


# ---------------------------------------------------------------------------
# clustering units
# ---------------------------------------------------------------------------

class TestClusterUnits(unittest.TestCase):
    def test_units_group_as_expected(self):
        games = [
            {"game_id": "1", "interrogator_user_id": "u1",
             "human_witness_user_id": "u2", "component": 0},
            {"game_id": "2", "interrogator_user_id": "u1",
             "human_witness_user_id": "u3", "component": 0},
            {"game_id": "3", "interrogator_user_id": "u9",
             "human_witness_user_id": "u8", "component": 1},
        ]
        units = a0.cluster_units(games)
        self.assertEqual(len(units["game"]), 3)
        self.assertEqual(len(units["interrogator"]), 2)
        self.assertEqual(units["interrogator"]["u1"], [0, 1])
        self.assertEqual(len(units["component"]), 2)

    def test_widen_picks_the_wider_interval(self):
        narrow = {"lo": 0.4, "hi": 0.5}
        wide = {"lo": 0.3, "hi": 0.6}
        self.assertIs(a0.widen(narrow, wide), wide)
        self.assertIs(a0.widen(wide, narrow), wide)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# ablation conditions (added 2026-08-18)
# ---------------------------------------------------------------------------

class TestAblationConditions(unittest.TestCase):
    def _dialogue_both_sides(self):
        d = dialogue("A", True, ["witness one", "witness two"])
        d["turns"].insert(0, {"role": "I", "content": "interrogator asks",
                              "is_changed": False, "timestamp": ""})
        return d

    def test_sides_witness_excludes_interrogator(self):
        d = self._dialogue_both_sides()
        self.assertEqual(a0.dialogue_messages(d, "raw", "witness"),
                         ["witness one", "witness two"])

    def test_sides_both_includes_interrogator(self):
        d = self._dialogue_both_sides()
        self.assertEqual(a0.dialogue_messages(d, "raw", "both"),
                         ["interrogator asks", "witness one", "witness two"])

    def test_default_sides_is_witness_only(self):
        """The as-run behaviour must be the default, or the original artifact
        stops being reproducible from this script."""
        d = self._dialogue_both_sides()
        self.assertEqual(a0.dialogue_messages(d, "raw"),
                         a0.dialogue_messages(d, "raw", "witness"))

    def test_length_family_is_entirely_length_derived(self):
        """Every length feature must be in LENGTH_DERIVED, so the nolen cell
        leaves that family with nothing rather than with a stray survivor."""
        keys = set(a0.length_features(["some text"]))
        self.assertEqual(keys - a0.LENGTH_DERIVED, set())

    def test_nolen_drops_exactly_the_registered_features(self):
        det = a0.DenseDetector("fw", a0.function_word_features, l2=5.0)
        det.drop_length = True
        det.fit([dialogue("A", True, ["the cat sat"]),
                 dialogue("B", False, ["a dog ran"])], "raw")
        self.assertNotIn("fw_type_token", det.keys)
        self.assertNotIn("empty", det.keys)
        self.assertIn("fw_the", det.keys)

    def test_punctuation_keeps_rates_under_nolen(self):
        det = a0.DenseDetector("punct", a0.punctuation_features)
        det.drop_length = True
        det.fit([dialogue("A", True, ["Hi. Ok?"]),
                 dialogue("B", False, ["yes! no."])], "raw")
        self.assertIn("punct_period", det.keys)
        self.assertIn("upper_rate", det.keys)
        self.assertNotIn("empty", det.keys)

    def test_length_detector_degenerates_gracefully_under_nolen(self):
        """With no features left it must fall back to the class prior, not crash."""
        det = a0.DenseDetector("length", a0.length_features)
        det.drop_length = True
        train = [dialogue("A", True, ["aa"]), dialogue("B", False, ["bbbb"])]
        det.fit(train, "raw")
        self.assertEqual(det.keys, [])
        self.assertTrue(det.diagnostics()["degenerate_no_features"])
        probs = det.predict(train, "raw")
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs[0], probs[1])   # constant => ties => 0.5 credit

    def test_make_detectors_propagates_the_condition(self):
        cond = a0.Condition("t", "both", True)
        for det in a0.make_detectors(cond):
            self.assertEqual(det.sides, "both")
            self.assertTrue(det.drop_length)

    def test_condition_registry_covers_the_requested_cells(self):
        names = [c.name for c in a0.CONDITIONS]
        for required in ("A0-full", "A0-witness", "A0-wit-nolen"):
            self.assertIn(required, names)

    def test_full_and_witness_conditions_are_identical_by_construction(self):
        full = next(c for c in a0.CONDITIONS if c.name == "A0-full")
        wit = next(c for c in a0.CONDITIONS if c.name == "A0-witness")
        self.assertEqual((full.sides, full.drop_length),
                         (wit.sides, wit.drop_length))

    def test_tfidf_vectors_are_l2_normalised(self):
        """Length must not leak back through the vector norm."""
        det = a0.TfidfDetector(min_df=1)
        train = [dialogue("A", True, ["alpha beta gamma delta epsilon"]),
                 dialogue("B", False, ["zeta eta theta"])]
        det.fit(train, "raw")
        for d in train:
            vec = det._vector(a0.tfidf_terms(a0.dialogue_messages(d, "raw")))
            norm = sum(v * v for _, v in vec) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=9)

    def test_token_cap_truncates_to_a_fixed_budget(self):
        d = {"turns": [{"role": "W", "content": "a b c d e f", "is_changed": False},
                       {"role": "W", "content": "g h i j", "is_changed": False}]}
        self.assertEqual(a0.dialogue_messages(d, "raw", "witness", 4), ["a b c d"])
        self.assertEqual(
            sum(len(m.split()) for m in a0.dialogue_messages(d, "raw", "witness", 8)), 8)

    def test_token_cap_leaves_short_dialogues_alone(self):
        d = {"turns": [{"role": "W", "content": "a b", "is_changed": False}]}
        self.assertEqual(a0.dialogue_messages(d, "raw", "witness", 20), ["a b"])

    def test_token_cap_default_is_none_so_as_run_is_unchanged(self):
        d = {"turns": [{"role": "W", "content": " ".join(str(i) for i in range(50)),
                        "is_changed": False}]}
        self.assertEqual(a0.dialogue_messages(d, "raw"),
                         a0.dialogue_messages(d, "raw", "witness", None))


# ---------------------------------------------------------------------------
# evaluation-set policy and balanced accuracy (added 2026-08-18, review round 2)
# ---------------------------------------------------------------------------

class TestEvalSetPolicy(unittest.TestCase):
    def _dialogues(self):
        rows = []
        for gid, (h_msgs, a_msgs) in {
            "1": (["hello there"], ["hi friend"]),      # both speak
            "2": ([], ["hi friend"]),                   # human silent
            "3": (["hello there"], []),                 # AI silent
            "4": ([], []),                              # both silent
        }.items():
            for label, msgs, human in (("A", h_msgs, True), ("B", a_msgs, False)):
                d = dialogue(label, human, msgs)
                d["game_id"] = gid
                d["example_id"] = f"{gid}-{label}"
                d["n_witness_messages"] = len(msgs)
                rows.append(d)
        return rows

    def test_empty_witness_games_finds_every_affected_game(self):
        self.assertEqual(a0.empty_witness_games(self._dialogues()), {"2", "3", "4"})

    def test_empty_witness_games_is_condition_independent(self):
        """The drop set comes from canonical message counts, not from featurised
        text, so every ablation cell drops the SAME games."""
        rows = self._dialogues()
        first = a0.empty_witness_games(rows)
        for cond in a0.CONDITIONS:
            self.assertEqual(a0.empty_witness_games(rows), first, cond.name)

    def test_balanced_accuracy_hand_computed(self):
        # 2 positives (one caught), 2 negatives (both caught)
        # sens = 0.5, spec = 1.0 -> 0.75
        self.assertAlmostEqual(
            a0.balanced_accuracy([0.9, 0.1, 0.2, 0.3], [1, 1, 0, 0]), 0.75)

    def test_balanced_accuracy_equals_accuracy_when_classes_balanced(self):
        probs = [0.9, 0.8, 0.2, 0.1]
        labels = [1, 1, 0, 0]
        self.assertAlmostEqual(a0.balanced_accuracy(probs, labels), 1.0)

    def test_balanced_accuracy_is_in_metric_keys(self):
        self.assertIn("dialogue_balanced_accuracy", a0.METRIC_KEYS)

    def test_evalset_composition_reports_exact_balance(self):
        games = [{"game_id": "1", "human_conversation_label": "A",
                  "interrogator_recruitment_source": "1"}]
        comp = a0.evalset_composition(games, self._dialogues())
        self.assertEqual(comp["all"]["games"], 1)
        self.assertEqual(comp["all"]["human_dialogues"], 1)
        self.assertEqual(comp["all"]["ai_dialogues"], 1)
        self.assertTrue(comp["all"]["class_balance_exact"])
        self.assertEqual(comp["prolific"]["games"], 1)
        self.assertEqual(comp["sona_ucsd"]["games"], 0)
