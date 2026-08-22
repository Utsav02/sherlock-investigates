import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2" / "scripts"))
import track_a_bridge as bridge  # noqa: E402


class BridgeSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = bridge.scoring_records()

    def test_selection_is_train_dev_only_and_paired(self):
        self.assertEqual(len(self.rows), 1700)
        self.assertEqual({row["split"] for row in self.rows}, {"train", "dev"})
        by_game = {}
        for row in self.rows:
            by_game.setdefault(row["game_id"], set()).add(row["conversation_label"])
        self.assertEqual(len(by_game), 850)
        self.assertTrue(all(sides == {"A", "B"} for sides in by_game.values()))

    def test_scoring_manifest_has_text_hash_but_results_need_not_store_text(self):
        row = self.rows[0]
        self.assertEqual(len(row["text_sha256"]), 64)
        self.assertTrue(row["text"])
        self.assertEqual(
            bridge.sha256_bytes(row["text"].encode("utf-8")), row["text_sha256"]
        )

    def test_limit_is_deterministic_prefix(self):
        self.assertEqual(bridge.scoring_records(2), self.rows[:2])


class DurableScoreTests(unittest.TestCase):
    def test_duplicate_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scores.jsonl"
            row = {"example_id": "x"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(RuntimeError, "duplicate score"):
                bridge.read_score_rows(path)

    def test_resume_rejects_changed_text_or_revision(self):
        expected = {"x": {"text_sha256": "a" * 64}}
        with self.assertRaisesRegex(RuntimeError, "text hash changed"):
            bridge.validate_existing({"x": {
                "text_sha256": "b" * 64,
                "model_revision": bridge.MODEL_REVISION,
                "p_ai": 0.5,
            }}, expected)
        with self.assertRaisesRegex(RuntimeError, "model revision changed"):
            bridge.validate_existing({"x": {
                "text_sha256": "a" * 64,
                "model_revision": "moving-main",
                "p_ai": 0.5,
            }}, expected)

    def test_atomic_json_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            bridge.atomic_json(path, {"status": "running", "processed": 1})
            bridge.atomic_json(path, {"status": "complete", "processed": 2})
            self.assertEqual(json.loads(path.read_text()),
                             {"status": "complete", "processed": 2})
class NestedCalibrationTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for component in range(4):
            rows.extend([
                {"component": component, "p_ai": 0.1 + component * 0.01, "label": 0},
                {"component": component, "p_ai": 0.9 - component * 0.01, "label": 1},
            ])
        return rows

    def test_platt_is_finite_and_orders_separable_scores(self):
        model = bridge.fit_platt(self.rows(), 1.0)
        low = bridge.predict_platt(model, 0.1)
        high = bridge.predict_platt(model, 0.9)
        self.assertTrue(math.isfinite(low) and math.isfinite(high))
        self.assertLess(low, high)

    def test_platt_can_record_an_inverted_transfer_relationship(self):
        inverted = [dict(row, label=1 - row["label"]) for row in self.rows()]
        model = bridge.fit_platt(inverted, 1.0)
        self.assertLess(model["beta"][1], 0.0)

    def test_inner_component_selection_is_deterministic(self):
        first = bridge.select_l2_nested(self.rows())
        second = bridge.select_l2_nested(list(reversed(self.rows())))
        self.assertEqual(first, second)
        self.assertIn(first[0], bridge.L2_GRID)
        self.assertEqual(set(first[1]), {str(value) for value in bridge.L2_GRID})


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_falls_back_when_mps_is_advertised_but_unusable(self):
        class Tensor:
            def to(self, _device):
                raise RuntimeError("unsupported OS")

        class MPS:
            @staticmethod
            def is_available():
                return True

        class CUDA:
            @staticmethod
            def is_available():
                return False

        class Torch:
            class backends:
                mps = MPS()
            cuda = CUDA()

            @staticmethod
            def ones(_n):
                return Tensor()

        self.assertEqual(bridge.choose_device(Torch, "auto"), "cpu")


class GateDecisionTests(unittest.TestCase):
    @staticmethod
    def direction(acc, brier, participant_lo, component_lo):
        return {
            "nested_calibrated": {
                "point_estimates": {"game_accuracy": acc, "dialogue_brier": brier},
                "intervals": {
                    "participant": {"detectors": {"external": {
                        "game_accuracy": {"lo": participant_lo}}}},
                    "component": {"detectors": {"external": {
                        "game_accuracy": {"lo": component_lo}}}},
                },
            }
        }

    def test_gate_distinguishes_pass_inconclusive_and_fail(self):
        passed = {k: self.direction(0.7, 0.2, 0.55, 0.52) for k in ("a", "b")}
        self.assertEqual(bridge.gate_decision(passed)["verdict"], "PASS")
        inconclusive = dict(passed)
        inconclusive["b"] = self.direction(0.7, 0.2, 0.55, 0.49)
        self.assertEqual(bridge.gate_decision(inconclusive)["verdict"], "INCONCLUSIVE")
        failed = dict(passed)
        failed["b"] = self.direction(0.49, 0.2, 0.4, 0.4)
        self.assertEqual(bridge.gate_decision(failed)["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
