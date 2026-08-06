from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs" / "mechanism_feasibility_v1.json"
V1_RECORD = ROOT / "protocols" / "protocol_v1_no_go.json"
V2_CONFIG = ROOT / "configs" / "mechanism_feasibility_v2.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MethodContractV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.v1 = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
        self.record = json.loads(V1_RECORD.read_text(encoding="utf-8"))
        self.v2 = json.loads(V2_CONFIG.read_text(encoding="utf-8"))

    def test_v1_no_go_is_preserved_without_method_overclaim(self) -> None:
        self.assertEqual(self.record["protocol_decision"], "NO_GO")
        self.assertEqual(self.record["gate_decision"], "GATE_FAIL")
        self.assertEqual(self.record["method_conclusion"], "not_determined")
        self.assertEqual(self.record["observed_condition_number"], 19.87784943452387)
        for relative, expected in self.record["source_sha256"].items():
            self.assertEqual(_sha256(ROOT / relative), expected)
        self.assertEqual(self.v2["v1_preservation"]["v1_config_sha256"], _sha256(V1_CONFIG))

    def test_v2_channels_are_explicit_finite_full_rank_and_admissible(self) -> None:
        channel = self.v2["synthetic_protocol"]["channel"]
        self.assertEqual(channel["construction"], "R_theta1_times_diag_s1_s2_times_R_theta2")
        self.assertTrue(channel["no_random_matrix_draws"])
        self.assertFalse(channel["resampling_permitted"])
        cases = channel["case_parameters"]
        self.assertEqual([case["case_index"] for case in cases], list(range(8)))
        for case in cases:
            values = [case["theta_1"], case["theta_2"], case["s_1"], case["s_2"], *case["bias"]]
            self.assertTrue(all(math.isfinite(value) for value in values))
            self.assertGreater(case["s_1"], 0.0)
            self.assertGreater(case["s_2"], 0.0)
            self.assertGreater(case["s_1"] * case["s_2"], 0.0)
            condition = max(case["s_1"], case["s_2"]) / min(case["s_1"], case["s_2"])
            self.assertLessEqual(condition, channel["maximum_condition_number"])
            self.assertLessEqual(condition, self.v2["gate_1a"]["maximum_condition_number"])

    def test_v2_gate_1a_is_input_only_and_complete(self) -> None:
        gate = self.v2["gate_1a"]
        self.assertEqual(gate["purpose"], "input_admissibility_only_no_method_metrics")
        self.assertEqual(gate["case_count"], 8)
        self.assertEqual(gate["null_case_count"], 8)
        self.assertIn("AISB_acquisition", gate["forbidden_calls"])
        self.assertIn("owner_scoring", gate["forbidden_calls"])
        cases = self.v2["synthetic_protocol"]["case_protocols"]
        edit_counts = Counter(case["edit"] for case in cases)
        self.assertEqual(len(cases), gate["case_count"])
        self.assertTrue(all(count == 2 for count in edit_counts.values()))
        self.assertEqual(self.v2["synthetic_protocol"]["null_protocol"]["case_count"], gate["null_case_count"])

    def test_v2_gate_1b_thresholds_are_not_lowered_from_v1(self) -> None:
        gate = self.v2["gate_1b"]
        v1_gate = self.v1["synthetic_gate"]
        self.assertEqual(gate["minimum_truth_coverage"], v1_gate["minimum_truth_coverage"])
        self.assertEqual(gate["maximum_false_acquisition_rate"], v1_gate["maximum_false_acquisition_rate"])
        self.assertEqual(gate["minimum_owner_positive_fraction"], v1_gate["minimum_owner_positive_fraction"])
        self.assertEqual(gate["minimum_score_margin"], v1_gate["minimum_score_margin"])
        self.assertEqual(gate["maximum_public_held_out_mse"], self.v1["saved_video_gate"]["maximum_public_held_out_mse"])
        self.assertEqual(gate["maximum_channel_condition_number"], self.v1["saved_video_gate"]["maximum_channel_condition_number"])
        self.assertTrue(gate["requires_gate_1a_pass"])
        self.assertTrue(gate["requires_identical_config_and_channel_digests"])

    def test_v2_retains_every_v1_noise_and_null_seed(self) -> None:
        seeds = self.v2["synthetic_protocol"]["seed_derivation"]
        v1_seeds = self.v1["synthetic_protocol"]["seed_derivation"]
        self.assertNotIn("synthetic.channel", seeds["domains"])
        self.assertEqual(seeds["master"], v1_seeds["master"])
        self.assertEqual(seeds["version"], v1_seeds["version"])
        self.assertEqual(seeds["input_formula"], v1_seeds["input_formula"])
        self.assertEqual(
            seeds["domains"],
            ["synthetic.noise", "synthetic.null.observation"],
        )
        for domain, expected in seeds["golden_case_0"].items():
            self.assertEqual(expected, v1_seeds["golden_case_0"][domain])
            payload = "{}|{}|0".format(seeds["master"], domain).encode("utf-8")
            actual = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
            self.assertEqual(actual, expected)
        for case_index in range(8):
            for domain in seeds["domains"]:
                v1_payload = "{}|{}|{}".format(v1_seeds["master"], domain, case_index).encode("utf-8")
                v2_payload = "{}|{}|{}".format(seeds["master"], domain, case_index).encode("utf-8")
                v1_derived = int.from_bytes(hashlib.sha256(v1_payload).digest()[:8], "big", signed=False)
                v2_derived = int.from_bytes(hashlib.sha256(v2_payload).digest()[:8], "big", signed=False)
                self.assertEqual(v2_derived, v1_derived)
        rng = self.v2["synthetic_protocol"]["rng"]
        self.assertEqual(rng["matrix_draw_order"], "none_constructed_from_explicit_case_parameters")


if __name__ == "__main__":
    unittest.main()
