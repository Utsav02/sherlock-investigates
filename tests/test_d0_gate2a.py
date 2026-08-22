import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2" / "scripts"))
import d0_gate2a as d0  # noqa: E402


class D0Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = d0.load_config(ROOT / "v2/configs/d0_gate2a_v1.json")
        cls.family = cls.config["families"][0]


class ConfigTests(D0Fixture):
    def test_config_has_strict_normalized_likelihoods(self):
        self.assertEqual(len(self.config["questions"]), 12)
        self.assertEqual(len(self.config["families"]), 16)
        for family in self.config["families"]:
            for question in self.config["questions"]:
                probs = d0.likelihoods(self.config, family, question["id"])
                for typ in ("human", "ai"):
                    self.assertAlmostEqual(sum(probs[typ].values()), 1.0, places=14)
                    self.assertTrue(all(0 < p < 1 for p in probs[typ].values()))

    def test_development_and_heldout_surfaces_are_disjoint(self):
        development = {q["development"] for q in self.config["questions"]}
        heldout = {q["heldout"] for q in self.config["questions"]}
        self.assertFalse(development & heldout)
        for category in d0.CATEGORIES:
            self.assertFalse(set(self.config["response_renderings"]["development"][category]) &
                             set(self.config["response_renderings"]["heldout"][category]))

    def test_invalid_surface_overlap_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["questions"][0]["heldout"] = config["questions"][0]["development"]
        with self.assertRaisesRegex(ValueError, "surfaces overlap"):
            d0.validate_config(config)


class MechanicsTests(D0Fixture):
    def test_balanced_labels_are_deterministic(self):
        first = d0.episode_labels(self.config, self.family)
        self.assertEqual(first, d0.episode_labels(self.config, self.family))
        self.assertEqual(first.count("human"), 128)
        self.assertEqual(first.count("ai"), 128)

    def test_posterior_matches_hand_calculation(self):
        self.assertAlmostEqual(d0.posterior(.5, .8, .2), .8)
        self.assertAlmostEqual(d0.posterior(.8, .2, .8), .5)

    def test_stronger_absolute_signal_has_more_exact_eig(self):
        strong = d0.exact_eig(self.config, self.family, "surroundings", .5)
        weak = d0.exact_eig(self.config, self.family, "logic", .5)
        self.assertGreater(strong, weak)
        self.assertGreaterEqual(weak, -1e-15)

    def test_bed_selects_largest_eig_and_uot_is_deterministic(self):
        unused = [q["id"] for q in self.config["questions"]]
        selected = d0.select_question("bed_eig", self.config, self.family, 0, 1, unused, .5)
        expected = min(unused, key=lambda q: (-d0.exact_eig(self.config, self.family, q, .5), q))
        self.assertEqual(selected, expected)
        self.assertEqual(
            d0.select_question("uot_sample", self.config, self.family, 3, 2, unused, .41),
            d0.select_question("uot_sample", self.config, self.family, 3, 2, unused, .41))

    def test_common_response_schedule_and_valid_ledgers(self):
        rows = [d0.simulate_trajectory(self.config, self.family, 7, policy)
                for policy in d0.POLICIES]
        self.assertEqual(len({r["schedule_sha256"] for r in rows}), 1)
        validation = d0.validate_rows(self.config, rows, require_complete=False, reproduce=True)
        self.assertTrue(validation["valid"], validation["failures"])
        for row in rows:
            qids = [turn["question_id"] for turn in row["ledger"]]
            self.assertEqual(len(qids), 4)
            self.assertEqual(len(set(qids)), 4)
            self.assertTrue(math.isfinite(row["final_log_loss"]))


class DurabilityTests(D0Fixture):
    def tiny_config(self):
        config = copy.deepcopy(self.config)
        config["episodes_per_family"] = 2
        return config

    def test_interrupted_run_resumes_exactly_without_duplicates(self):
        config = self.tiny_config()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "rows.jsonl"
            state = Path(temp) / "state.json"
            d0.run_benchmark(config, output, state, resume=False, limit=5)
            self.assertEqual(len(d0.read_rows(output)), 5)
            self.assertNotIn("simulate", json.loads(state.read_text())["completed"])
            d0.run_benchmark(config, output, state, resume=True)
            rows = d0.read_rows(output)
            self.assertEqual(len(rows), len(d0.expected_ids(config)))
            self.assertEqual(len({r["trajectory_id"] for r in rows}), len(rows))
            self.assertTrue(d0.validate_rows(config, rows)["valid"])

    def test_resume_rejects_duplicate_output(self):
        config = self.tiny_config()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "rows.jsonl"
            state = Path(temp) / "state.json"
            row = d0.simulate_trajectory(config, config["families"][0], 0, "random")
            output.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(SystemExit):
                d0.run_benchmark(config, output, state, resume=True)


class GateTests(D0Fixture):
    @staticmethod
    def comparisons(mean=.1, lower=.02, positives=8, dev_mean=.01):
        result = {}
        for split, split_mean in (("development", dev_mean), ("heldout", mean)):
            result[split] = {}
            for comparator in ("random", "fixed"):
                result[split][comparator] = {
                    "episode_delta": {"mean": split_mean},
                    "cluster_bootstrap_95": [lower, .2],
                    "positive_family_count": positives,
                }
        return result

    def test_gate_distinguishes_pass_inconclusive_and_fail(self):
        integrity = {"valid": True}
        self.assertEqual(d0.evaluate_gate(self.config, self.comparisons(), integrity)["decision"], "PASS")
        self.assertEqual(d0.evaluate_gate(self.config, self.comparisons(mean=.03), integrity)["decision"],
                         "INCONCLUSIVE")
        self.assertEqual(d0.evaluate_gate(self.config, self.comparisons(mean=-.01), integrity)["decision"],
                         "FAIL")
        self.assertEqual(d0.evaluate_gate(self.config, self.comparisons(), {"valid": False})["decision"],
                         "FAIL")


if __name__ == "__main__":
    unittest.main()
