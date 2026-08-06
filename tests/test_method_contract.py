from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "mechanism_feasibility_v1.json"


class MethodContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_scope_dimensions_carrier_and_proxy_exclusions(self) -> None:
        self.assertEqual(self.config["claim_scope"], "mechanism_feasibility_only")
        protocol = self.config["synthetic_protocol"]
        saved = self.config["saved_video_gate"]
        self.assertEqual(protocol["observation_dimension"], 2)
        self.assertEqual(protocol["channel"]["matrix_shape"], [2, 2])
        self.assertEqual(protocol["null_protocol"]["observation_dimension"], 2)
        self.assertEqual(saved["observation_dimension"], 2)
        carrier = self.config["carrier"]
        self.assertEqual(carrier["primary"], "direct_dit_final_model_output_residual")
        self.assertEqual(carrier["expected_injected_step_count"], self.config["generation"]["inference_steps"])
        self.assertEqual(carrier["minimum_injected_step_fraction"], 1.0)
        forbidden = set(self.config["forbidden_final_evidence"])
        self.assertTrue({"scheduler_control_only", "internal_activation_only", "clean_watermarked_paired_difference", "ground_truth_source_indices", "prompt_only_geometric_control"} <= forbidden)

    def test_public_split_keys_prompt_and_model_are_frozen(self) -> None:
        schedule = self.config["state_schedule"]
        calibration = set(schedule["public_calibration_point_ids"])
        held_out = set(schedule["public_held_out_point_ids"])
        self.assertEqual(calibration | held_out, set(schedule["public_point_ids"]))
        self.assertFalse(calibration & held_out)
        keys = self.config["candidate_keys"]
        self.assertEqual(len(keys["wrong_key_ids"]), len(set(keys["wrong_key_ids"])))
        self.assertEqual(len(keys["wrong_key_ids"]), 31)
        self.assertNotIn(keys["owner_key_id"], keys["wrong_key_ids"])
        generation = self.config["generation"]
        self.assertEqual(generation["seeds"], list(range(1275, 1283)))
        self.assertTrue(generation["prompt"] and generation["negative_prompt"])
        revision = self.config["model"]["revision"]
        self.assertEqual(len(revision), 40)
        int(revision, 16)

    def test_template_scan_and_ambiguity_serialization_are_frozen(self) -> None:
        protocol = self.config["synthetic_protocol"]
        points = protocol["template"]["points"]
        self.assertEqual(len(points), 12)
        self.assertEqual([p["logical_state_id"] for p in points[:6]], self.config["state_schedule"]["public_point_ids"])
        for primary, copy1, copy2 in ((0, 6, 9), (1, 7, 10), (2, 8, 11)):
            self.assertEqual(points[primary]["xy"], points[copy1]["xy"])
            self.assertEqual(points[primary]["xy"], points[copy2]["xy"])
        acquisition = protocol["acquisition"]
        expected = {"12": 1, "11": math.comb(12, 1), "10": math.comb(12, 2)}
        self.assertEqual(acquisition["expected_candidate_count_per_start_by_observed_length"], expected)
        self.assertGreaterEqual(acquisition["top_k_per_start_per_observed_length"], max(expected.values()))
        self.assertEqual(acquisition["candidate_truncation"], "none")
        artifact = protocol["ambiguity_artifact"]
        self.assertEqual(artifact["membership"], "single_deterministic_best_non_overlapping_sequence_after_threshold")
        self.assertTrue(artifact["not_all_optimal_sequences"])
        self.assertEqual(artifact["calibration_input"], "readback_frozen_ambiguity_artifact")
        self.assertEqual(artifact["candidate_key_access"], "forbidden")

    def test_sequence_edits_occurrences_null_channel_and_scoring(self) -> None:
        protocol = self.config["synthetic_protocol"]
        sequence = protocol["sequence"]
        self.assertEqual(sum(s["length"] for s in sequence["segments"]), sequence["total_window_count_before_edit"])
        cases = protocol["case_protocols"]
        self.assertEqual([c["case_index"] for c in cases], list(range(8)))
        counts = Counter(c["edit"] for c in cases)
        self.assertEqual(set(counts), set(self.config["synthetic_gate"]["required_edit_cases"]))
        self.assertTrue(all(count == 2 for count in counts.values()))
        edit = protocol["edit_application"]
        self.assertEqual(edit["order"][-1], "apply_declared_edit_to_observation_sequence")
        self.assertIn("without_new_noise", edit["duplication"])
        occurrence = protocol["public_occurrence_aggregation"]
        self.assertIn("all_retained_occurrences", occurrence["calibration"])
        self.assertIn("all_retained_occurrences", occurrence["held_out"])
        null = protocol["null_protocol"]
        self.assertEqual(null["case_count"], self.config["synthetic_gate"]["case_count"])
        self.assertEqual(null["window_count"], sequence["total_window_count_before_edit"])
        scoring = protocol["scoring"]
        self.assertEqual(scoring["key_search"], "exhaustive_owner_and_all_configured_wrong_keys")
        self.assertEqual(scoring["search_truncation"], "none")

    def test_rng_draw_order_and_golden_seeds_are_exact(self) -> None:
        protocol = self.config["synthetic_protocol"]
        rng = protocol["rng"]
        self.assertEqual(rng["rng_version"], "python_random_mt19937_seed_version2_gauss_box_muller_v1")
        self.assertEqual(rng["stream_reset"], "new_random_random_instance_for_each_domain_and_case")
        seeds = protocol["seed_derivation"]
        for domain, expected in seeds["golden_case_0"].items():
            payload = f"{seeds['master']}|{domain}|0".encode("utf-8")
            actual = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
            self.assertEqual(actual, expected)
        self.assertEqual(rng["channel_draw_order"][0], "matrix_row_major_2x2_uniform_minus1_plus1")
        self.assertEqual(rng["noise_draw_order"], "window_major_then_dimension_major_gauss_zero_noise_std")

    def test_metric_denominators_aggregations_and_ties_are_frozen(self) -> None:
        metrics = self.config["metric_protocol"]
        self.assertEqual(metrics["truth_coverage"]["zero_denominator"], "fail")
        self.assertEqual(metrics["false_acquisition_rate"]["zero_denominator"], "fail")
        self.assertEqual(metrics["owner_positive_fraction"]["tie"], "owner_failure")
        self.assertEqual(metrics["score_margin"]["aggregation"], "minimum_across_cases")
        self.assertIn("all_retained_occurrences", metrics["public_held_out_mse"]["per_case"])
        self.assertEqual(metrics["second_singular_value_over_repeatability"]["zero_denominator_floor"], 1e-12)


if __name__ == "__main__":
    unittest.main()
