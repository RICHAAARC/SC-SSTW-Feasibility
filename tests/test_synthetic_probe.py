from __future__ import annotations

import sys
import unittest
from pathlib import Path
import hashlib
import json
import math
import random
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from sc_sstw_feasibility.aisb import best_non_overlapping_sequence, make_default_templates, make_redundant_templates, make_double_redundant_templates, scan_burst_candidates, template_observation_pairs
from sc_sstw_feasibility.acquisition import acquire_pilots_by_periodic_beam
from sc_sstw_feasibility.calibration import calibrate_channel, calibrate_from_pilot_pairs, equalize_observations
from sc_sstw_feasibility.channel import generate_observations, make_random_channel
from sc_sstw_feasibility.scoring import score_key, score_key_with_calibration
from sc_sstw_feasibility.sync import (
    dynamic_time_sync,
    dynamic_time_sync_score,
    dynamic_time_sync_score_bounded,
    dynamic_time_sync_score_flat,
    dynamic_time_sync_score_bounded_flat,
    flatten_state_pairs,
)
from sc_sstw_feasibility.sync_fast import (
    as_c_flat_sequence,
    dynamic_time_sync_score_bounded_flat_c,
    dynamic_time_sync_score_bounded_prepared_c,
    dynamic_time_sync_score_bounded_prepared_workspace_c,
    make_c_dtw_workspace,
)
from sc_sstw_feasibility.state import PilotPattern, generate_state_sequence
from run_aisb_channel_mismatch_probe import (
    _apply_quadratic_mismatch,
    _mismatch_case,
    _random_non_burst_case as _mismatch_random_non_burst_case,
)
from run_aisb_ambiguity_probe import (
    _ambiguity_case,
    _candidate_ambiguity_sequences,
    _random_non_burst_case as _ambiguity_random_non_burst_case,
    _targeted_shifted_window_case,
)
from run_aisb_capacity_probe import _capacity_case
from run_aisb_capacity_mismatch_probe import _capacity_mismatch_case
from run_aisb_stress_mismatch_payload_probe import _stress_mismatch_payload_case
from run_aisb_stress_grid_probe import _ambiguity_scoring_grid, _stress_grid, _summarize_grid_cell
from run_aisb_long_sequence_probe import _apply_long_edits, _build_long_sequence, _long_sequence_case
from run_aisb_long_ambiguity_probe import _long_targeted_ambiguity_case
from run_aisb_multi_ambiguity_probe import _multi_ambiguity_case
from run_aisb_payload_probe import _payload_case
from run_aisb_probe import _redundant_two_deletion_case, _redundant_mixed_sequence_two_deletion_case
from run_aisb_exact_search_scale_probe import _exact_scale_case
from run_aisb_threshold_margin_probe import _high_mismatch_case, _random_non_burst_case as _threshold_random_non_burst_case
from run_aisb_threshold_candidate_exact_probe import _summarize as _threshold_candidate_exact_summarize
from run_aisb_sequence_consistency_probe import (
    _high_mismatch_sequence_case,
    _random_non_burst_sequence_case,
    _summarize_high as _sequence_summarize_high,
    _summarize_random as _sequence_summarize_random,
)
from run_aisb_sequence_supported_exact_probe import _sequence_supported_exact_case
from run_aisb_sequence_ambiguity_exact_probe import (
    _append_jsonl,
    _sequence_ambiguity_exact_case,
    _select_diagnostic_pruned_candidates,
    _supported_ambiguity_sequences,
)
from run_aisb_payload_probe import _candidate_key
from run_aisb_stress_probe import _random_non_burst_case, _stress_case
from run_temporal_robustness_probe import _case as _temporal_robustness_case


class SyntheticProbeTests(unittest.TestCase):
    def test_score_only_dtw_matches_full_path_scorer(self) -> None:
        observed = [
            (0.0, 0.0),
            (0.25, 0.05),
            (0.5, 0.25),
            (0.5, 0.25),
            (0.75, 0.6),
            (1.0, 1.0),
        ]
        candidate = [
            (0.0, 0.0),
            (0.22, 0.08),
            (0.48, 0.24),
            (0.68, 0.42),
            (0.75, 0.6),
            (0.88, 0.82),
            (1.0, 1.0),
        ]

        full = dynamic_time_sync(observed, candidate, skip_penalty=0.19, repeat_penalty=0.11)
        score_only = dynamic_time_sync_score(observed, candidate, skip_penalty=0.19, repeat_penalty=0.11)

        self.assertAlmostEqual(score_only, full.score, places=15)

    def test_score_only_dtw_matches_repeat_heavy_path_scorer(self) -> None:
        observed = [
            (0.0, 0.0),
            (0.0, 0.0),
            (0.2, 0.4),
            (0.2, 0.4),
            (0.55, 0.45),
            (0.9, 0.1),
        ]
        candidate = [
            (0.0, 0.0),
            (0.2, 0.4),
            (0.55, 0.45),
            (0.9, 0.1),
        ]

        full = dynamic_time_sync(observed, candidate, skip_penalty=0.23, repeat_penalty=0.07)
        score_only = dynamic_time_sync_score(observed, candidate, skip_penalty=0.23, repeat_penalty=0.07)

        self.assertAlmostEqual(score_only, full.score, places=15)

    def test_bounded_score_matches_unbounded_or_safely_abandons(self) -> None:
        observed = [
            (0.0, 0.0),
            (0.2, 0.4),
            (0.55, 0.45),
            (0.9, 0.1),
        ]
        good_candidate = [
            (0.0, 0.0),
            (0.2, 0.4),
            (0.55, 0.45),
            (0.9, 0.1),
        ]
        bad_candidate = [
            (10.0, 10.0),
            (10.2, 10.4),
            (10.55, 10.45),
            (10.9, 10.1),
        ]

        exact_good = dynamic_time_sync_score(observed, good_candidate)
        bounded_good, good_abandoned = dynamic_time_sync_score_bounded(
            observed,
            good_candidate,
            min_score_to_beat=-1.0,
        )
        bounded_bad, bad_abandoned = dynamic_time_sync_score_bounded(
            observed,
            bad_candidate,
            min_score_to_beat=exact_good,
        )

        self.assertFalse(good_abandoned)
        self.assertAlmostEqual(bounded_good, exact_good, places=15)
        self.assertTrue(bad_abandoned)
        self.assertLessEqual(bounded_bad, exact_good)

    def test_flat_score_matches_tuple_score(self) -> None:
        observed = [
            (0.0, 0.0),
            (0.2, 0.4),
            (0.55, 0.45),
            (0.55, 0.45),
            (0.9, 0.1),
        ]
        candidate = [
            (0.0, 0.0),
            (0.15, 0.35),
            (0.55, 0.45),
            (0.75, 0.25),
            (0.9, 0.1),
        ]

        tuple_score = dynamic_time_sync_score(observed, candidate)
        flat_score = dynamic_time_sync_score_flat(
            flatten_state_pairs(observed),
            flatten_state_pairs(candidate),
        )
        tuple_bounded, tuple_abandoned = dynamic_time_sync_score_bounded(
            observed,
            candidate,
            min_score_to_beat=tuple_score - 0.1,
        )
        flat_bounded, flat_abandoned = dynamic_time_sync_score_bounded_flat(
            flatten_state_pairs(observed),
            flatten_state_pairs(candidate),
            min_score_to_beat=tuple_score - 0.1,
        )

        self.assertAlmostEqual(flat_score, tuple_score, places=15)
        self.assertEqual(flat_abandoned, tuple_abandoned)
        self.assertAlmostEqual(flat_bounded, tuple_bounded, places=15)

    def test_native_flat_score_matches_python_flat_score(self) -> None:
        observed = flatten_state_pairs([
            (0.0, 0.0),
            (0.2, 0.4),
            (0.55, 0.45),
            (0.55, 0.45),
            (0.9, 0.1),
        ])
        candidate = flatten_state_pairs([
            (0.0, 0.0),
            (0.15, 0.35),
            (0.55, 0.45),
            (0.75, 0.25),
            (0.9, 0.1),
        ])

        python_score, python_abandoned = dynamic_time_sync_score_bounded_flat(
            observed,
            candidate,
            min_score_to_beat=-1.0,
        )
        native_score, native_abandoned = dynamic_time_sync_score_bounded_flat_c(
            observed,
            candidate,
            min_score_to_beat=-1.0,
        )
        prepared_score, prepared_abandoned = dynamic_time_sync_score_bounded_prepared_c(
            as_c_flat_sequence(observed),
            as_c_flat_sequence(candidate),
            min_score_to_beat=-1.0,
        )
        prepared_observed = as_c_flat_sequence(observed)
        prepared_candidate = as_c_flat_sequence(candidate)
        workspace_score, workspace_abandoned = dynamic_time_sync_score_bounded_prepared_workspace_c(
            prepared_observed,
            prepared_candidate,
            make_c_dtw_workspace(prepared_candidate.pair_count),
            min_score_to_beat=-1.0,
        )

        self.assertEqual(native_abandoned, python_abandoned)
        self.assertAlmostEqual(native_score, python_score, places=15)
        self.assertEqual(prepared_abandoned, python_abandoned)
        self.assertAlmostEqual(prepared_score, python_score, places=15)
        self.assertEqual(workspace_abandoned, python_abandoned)
        self.assertAlmostEqual(workspace_score, python_score, places=15)

    def test_pilot_calibration_recovers_low_error_state(self) -> None:
        pattern = PilotPattern(period=4)
        states = generate_state_sequence("owner", 64, pilot_pattern=pattern)
        channel = make_random_channel(11, relation_count=12, noise_std=0.01)
        observations = generate_observations(states, channel, seed=19)

        calibration = calibrate_channel(observations, pilot_pattern=pattern)
        equalized = equalize_observations(observations, calibration)

        pilot_errors = []
        for index in range(0, 64, 4):
            expected = pattern.pilot_at(index)
            assert expected is not None
            observed = equalized[index]
            pilot_errors.append((observed[0] - expected[0]) ** 2 + (observed[1] - expected[1]) ** 2)
        self.assertLess(sum(pilot_errors) / len(pilot_errors), 0.005)
        self.assertLess(calibration.condition_number, 10.0)

    def test_owner_key_separates_from_wrong_keys_after_deletion_and_crop(self) -> None:
        pattern = PilotPattern(period=4)
        owner_key = "owner_key"
        states = generate_state_sequence(owner_key, 72, pilot_pattern=pattern)
        channel = make_random_channel(23, relation_count=16, noise_std=0.025)
        observations = generate_observations(states, channel, seed=29)
        edited = []
        source_indices = []
        for source_index, value in enumerate(observations[5:66], start=5):
            local_index = source_index - 5
            if local_index in {8, 9, 22, 40}:
                continue
            edited.append(value)
            source_indices.append(source_index)

        owner_sync, condition_number, pilot_mse = score_key(
            edited,
            owner_key,
            pilot_pattern=pattern,
            candidate_length=72,
            source_indices=source_indices,
        )
        wrong_scores = [
            score_key(
                edited,
                f"wrong_key_{index}",
                pilot_pattern=pattern,
                candidate_length=72,
                source_indices=source_indices,
            )[0].score
            for index in range(10)
        ]

        self.assertGreater(owner_sync.score, max(wrong_scores))
        self.assertGreater(owner_sync.score - max(wrong_scores), 0.05)
        self.assertLess(condition_number, 10.0)
        self.assertLess(pilot_mse, 0.01)

    def test_observation_only_pilot_acquisition_is_key_independent(self) -> None:
        pattern = PilotPattern(period=4)
        owner_key = "owner_key_example"
        states = generate_state_sequence(owner_key, 64, pilot_pattern=pattern)
        channel = make_random_channel(20260730, relation_count=16, noise_std=0.03)
        observations = generate_observations(states, channel, seed=17)
        edited = []
        for source_index, value in enumerate(observations[4:60], start=4):
            local_index = source_index - 4
            if local_index in {7, 8, 21, 35}:
                continue
            edited.append(value)

        acquired = acquire_pilots_by_periodic_beam(
            edited,
            pilot_pattern=pattern,
            beam_width=220,
        )
        owner_sync = score_key_with_calibration(
            edited,
            owner_key,
            calibration=acquired.calibration,
            pilot_pattern=pattern,
            candidate_length=64,
        )
        wrong_scores = [
            score_key_with_calibration(
                edited,
                f"wrong_key_{index}",
                calibration=acquired.calibration,
                pilot_pattern=pattern,
                candidate_length=64,
            ).score
            for index in range(12)
        ]

        self.assertGreaterEqual(len(acquired.observed_indices), 8)
        self.assertLess(acquired.calibration.condition_number, 10.0)
        self.assertLess(acquired.calibration.pilot_reconstruction_mse, 0.01)
        self.assertGreater(owner_sync.score, max(wrong_scores))

    def test_aisb_affine_invariant_acquisition_finds_bursts_without_channel_fit(self) -> None:
        templates = make_default_templates()
        template = templates[0]
        states = [(0.2, -0.1), (0.1, 0.3), *template.points, (-0.4, 0.2)]
        channel = make_random_channel(71, relation_count=10, noise_std=0.005)
        observations = generate_observations(states, channel, seed=73)

        candidates = scan_burst_candidates(observations, templates)
        accepted = best_non_overlapping_sequence(
            candidates,
            burst_length=template.length,
            residual_threshold=0.006,
        )

        self.assertEqual([(candidate.start_index, candidate.template_id) for candidate in accepted], [(2, "burst_alpha")])

    def test_aisb_recovers_burst_with_single_checksum_deletion(self) -> None:
        templates = make_default_templates()
        template = templates[1]
        states = [(0.2, -0.1), *template.points[:4], *template.points[5:], (-0.4, 0.2)]
        channel = make_random_channel(91, relation_count=10, noise_std=0.005)
        observations = generate_observations(states, channel, seed=93)

        candidates = scan_burst_candidates(observations, templates, allow_single_deletion=True)
        accepted = best_non_overlapping_sequence(
            candidates,
            burst_length=template.length,
            residual_threshold=0.006,
        )

        self.assertEqual(
            [(candidate.start_index, candidate.template_id, candidate.missing_template_index) for candidate in accepted],
            [(1, "burst_beta", 4)],
        )

    def test_aisb_rejects_random_non_burst_observations_with_deletion_scan(self) -> None:
        rng = random.Random(81)
        states = [
            (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))
            for _ in range(40)
        ]
        channel = make_random_channel(83, relation_count=10, noise_std=0.005)
        observations = generate_observations(states, channel, seed=87)

        candidates = scan_burst_candidates(observations, make_default_templates(), allow_single_deletion=True)
        accepted = best_non_overlapping_sequence(
            candidates,
            burst_length=6,
            residual_threshold=0.006,
        )

        self.assertEqual(accepted, [])

    def test_aisb_alignment_supports_owner_wrong_key_scoring_after_checksum_deletion(self) -> None:
        templates = {template.template_id: template for template in make_default_templates()}
        burst_plan = [(6, "burst_alpha"), (22, "burst_beta"), (39, "burst_gamma")]
        owner_key = "owner_mixed_unit"

        def secret_state(key: str, index: int) -> tuple[float, float]:
            digest = hashlib.sha256(f"{key}:{index}".encode("utf-8")).digest()
            angle = 2.0 * math.pi * (int.from_bytes(digest[:8], "big") / float(1 << 64))
            return (math.cos(angle), math.sin(angle))

        def mixed_states(key: str) -> list[tuple[float, float]]:
            states = [secret_state(key, index) for index in range(58)]
            for start, template_id in burst_plan:
                for offset, point in enumerate(templates[template_id].points):
                    states[start + offset] = point
            return states

        owner_states = mixed_states(owner_key)
        deleted_sources = {start + 3 + (ordinal % 3) for ordinal, (start, _) in enumerate(burst_plan)}
        channel = make_random_channel(101, relation_count=12, noise_std=0.01)
        observations = generate_observations(owner_states, channel, seed=103)
        edited = [observation for index, observation in enumerate(observations) if index not in deleted_sources]
        source_indices = [index for index in range(len(owner_states)) if index not in deleted_sources]

        candidates = scan_burst_candidates(edited, make_default_templates(), allow_single_deletion=True)
        accepted = best_non_overlapping_sequence(candidates, burst_length=6, residual_threshold=0.006)
        pilot_pairs = []
        for candidate in accepted:
            pilot_pairs.extend(template_observation_pairs(candidate, edited, templates[candidate.template_id]))
        calibration = calibrate_from_pilot_pairs(pilot_pairs)
        equalized = equalize_observations(edited, calibration)
        owner_sync = dynamic_time_sync(equalized, mixed_states(owner_key))
        wrong_scores = [
            dynamic_time_sync(equalized, mixed_states(f"wrong_mixed_unit_{index}")).score
            for index in range(6)
        ]

        self.assertEqual(len(accepted), len(burst_plan))
        self.assertGreater(owner_sync.score, max(wrong_scores))
        self.assertGreater(owner_sync.score - max(wrong_scores), 0.05)
        self.assertLess(calibration.pilot_reconstruction_mse, 0.01)

    def test_redundant_aisb_recovers_each_single_missing_position(self) -> None:
        templates = make_redundant_templates()
        template = templates[0]
        channel = make_random_channel(121, relation_count=10, noise_std=0.004)
        for missing_index in range(template.length):
            states = [(0.3, -0.2), *[point for index, point in enumerate(template.points) if index != missing_index], (-0.2, 0.4)]
            observations = generate_observations(states, channel, seed=123 + missing_index)
            candidates = scan_burst_candidates(observations, templates, allow_single_deletion=True)
            accepted = best_non_overlapping_sequence(
                candidates,
                burst_length=template.length,
                residual_threshold=0.006,
            )

            self.assertEqual(
                [(candidate.start_index, candidate.template_id, candidate.missing_template_index) for candidate in accepted],
                [(1, "redundant_alpha", missing_index)],
            )

    def test_redundant_aisb_random_non_burst_threshold_margin(self) -> None:
        best_residuals = []
        accepted_count = 0
        templates = make_redundant_templates()
        for case_index in range(24):
            rng = random.Random(130 + case_index)
            states = [(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(48)]
            channel = make_random_channel(170 + case_index, relation_count=10, noise_std=0.004)
            observations = generate_observations(states, channel, seed=210 + case_index)
            candidates = scan_burst_candidates(observations, templates, allow_single_deletion=True)
            best_residuals.append(min(candidate.residual for candidate in candidates))
            accepted_count += len(best_non_overlapping_sequence(
                candidates,
                burst_length=templates[0].length,
                residual_threshold=0.006,
            ))

        self.assertEqual(accepted_count, 0)
        self.assertGreater(min(best_residuals), 0.006)

    def test_redundant_aisb_alignment_supports_owner_wrong_key_scoring_after_any_deletion(self) -> None:
        templates = {template.template_id: template for template in make_redundant_templates()}
        burst_plan = [(7, "redundant_alpha"), (26, "redundant_beta"), (47, "redundant_gamma")]
        owner_key = "owner_redundant_unit"
        missing_index = 1

        def secret_state(key: str, index: int) -> tuple[float, float]:
            digest = hashlib.sha256(f"{key}:{index}".encode("utf-8")).digest()
            angle = 2.0 * math.pi * (int.from_bytes(digest[:8], "big") / float(1 << 64))
            return (math.cos(angle), math.sin(angle))

        def mixed_states(key: str) -> list[tuple[float, float]]:
            states = [secret_state(key, index) for index in range(64)]
            for start, template_id in burst_plan:
                for offset, point in enumerate(templates[template_id].points):
                    states[start + offset] = point
            return states

        owner_states = mixed_states(owner_key)
        deleted_sources = {start + missing_index for start, _ in burst_plan}
        channel = make_random_channel(301, relation_count=12, noise_std=0.008)
        observations = generate_observations(owner_states, channel, seed=303)
        edited = [observation for index, observation in enumerate(observations) if index not in deleted_sources]

        candidates = scan_burst_candidates(edited, make_redundant_templates(), allow_single_deletion=True)
        accepted = best_non_overlapping_sequence(candidates, burst_length=9, residual_threshold=0.006)
        pilot_pairs = []
        for candidate in accepted:
            pilot_pairs.extend(template_observation_pairs(candidate, edited, templates[candidate.template_id]))
        calibration = calibrate_from_pilot_pairs(pilot_pairs)
        equalized = equalize_observations(edited, calibration)
        owner_sync = dynamic_time_sync(equalized, mixed_states(owner_key))
        wrong_scores = [
            dynamic_time_sync(equalized, mixed_states(f"wrong_redundant_unit_{index}")).score
            for index in range(6)
        ]

        self.assertEqual(
            [(candidate.start_index, candidate.template_id, candidate.missing_template_index) for candidate in accepted],
            [(7, "redundant_alpha", 1), (25, "redundant_beta", 1), (45, "redundant_gamma", 1)],
        )
        self.assertGreater(owner_sync.score, max(wrong_scores))
        self.assertGreater(owner_sync.score - max(wrong_scores), 0.05)
        self.assertLess(calibration.pilot_reconstruction_mse, 0.01)

    def test_double_redundant_template_keeps_anchor_class_after_two_deletions(self) -> None:
        template = make_double_redundant_templates()[0]
        anchor_groups = [{0, 6, 9}, {1, 7, 10}, {2, 8, 11}]

        for first in range(template.length):
            for second in range(first + 1, template.length):
                missing = {first, second}
                self.assertTrue(
                    all(group - missing for group in anchor_groups),
                    (first, second),
                )

    def test_redundant_aisb_recovers_representative_double_missing_positions(self) -> None:
        for case_index in range(9):
            result = _redundant_two_deletion_case(case_index)

            self.assertTrue(result["pass"], result)
            self.assertEqual(result["alignment_accuracy"], 1.0)
            self.assertEqual(result["false_positive"], 0)
            self.assertEqual(result["false_negative"], 0)
            self.assertLess(result["state_reconstruction_mse"], 0.02)

    def test_redundant_aisb_double_missing_supports_owner_wrong_key_scoring(self) -> None:
        result = _redundant_mixed_sequence_two_deletion_case(4)

        self.assertTrue(result["pass"], result)
        self.assertEqual(result["alignment_accuracy"], 1.0)
        self.assertEqual(result["false_positive"], 0)
        self.assertEqual(result["false_negative"], 0)
        self.assertGreater(result["score_margin"], 0.02)
        self.assertLess(result["calibration_pilot_mse"], 0.02)

    def test_temporal_robustness_probe_smoke_cases_pass(self) -> None:
        clock = _temporal_robustness_case(3, mode="clock_distortion", residual_threshold=0.015)
        combined = _temporal_robustness_case(4, mode="combined", residual_threshold=0.015)

        self.assertTrue(clock["pass"], clock)
        self.assertTrue(combined["pass"], combined)
        self.assertTrue(clock["truth_covered_by_candidates"])
        self.assertTrue(combined["truth_covered_by_candidates"])
        self.assertGreater(clock["score_margin"], 0.02)
        self.assertGreater(combined["score_margin"], 0.02)

    def test_redundant_aisb_stress_case_handles_crop_deletion_and_repeats(self) -> None:
        result = _stress_case(3, noise_std=0.016)

        self.assertTrue(result["pass"])
        self.assertEqual(result["alignment_accuracy"], 1.0)
        self.assertEqual(result["false_positive"], 0)
        self.assertEqual(result["false_negative"], 0)
        self.assertGreater(result["score_margin"], 0.02)
        self.assertLess(result["state_reconstruction_mse"], 0.02)

    def test_redundant_aisb_stress_random_non_burst_rejects(self) -> None:
        result = _random_non_burst_case(7, noise_std=0.016)

        self.assertTrue(result["pass"])
        self.assertEqual(result["accepted_count"], 0)
        self.assertGreater(result["best_residual"], 0.006)

    def test_aisb_payload_scoring_recovers_owner_message(self) -> None:
        result = _payload_case(4, noise_std=0.016)

        self.assertTrue(result["pass"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertEqual(result["alignment_accuracy"], 1.0)
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_payload_scoring_uses_fair_wrong_key_message_search(self) -> None:
        result = _payload_case(9, noise_std=0.016)

        self.assertTrue(result["pass"])
        self.assertEqual(result["message_space_size"], 8)
        self.assertEqual(result["wrong_key_count"], 12)
        self.assertGreater(result["owner_best_score"], result["best_wrong_score"])

    def test_aisb_channel_mismatch_probe_preserves_payload_scoring(self) -> None:
        result = _mismatch_case(4, gamma=0.5)

        self.assertTrue(result["pass"])
        self.assertEqual(result["alignment_accuracy"], 1.0)
        self.assertTrue(result["owner_message_recovered"])
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_channel_mismatch_random_non_burst_rejects(self) -> None:
        result = _mismatch_random_non_burst_case(5, gamma=0.5)

        self.assertTrue(result["pass"])
        self.assertEqual(result["accepted_count"], 0)
        self.assertGreater(result["best_residual"], 0.006)

    def test_aisb_overlap_tie_break_rejects_shifted_window_ambiguity(self) -> None:
        result = _mismatch_case(9, gamma=0.5)

        self.assertTrue(result["pass"])
        self.assertEqual(result["alignment_accuracy"], 1.0)
        self.assertEqual(result["false_positive"], 0)
        self.assertEqual(result["false_negative"], 0)

    def test_aisb_ambiguity_set_payload_scoring_uses_fair_search(self) -> None:
        result = _ambiguity_case(4, noise_std=0.016)

        self.assertTrue(result["pass"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertEqual(result["message_space_size"], 8)
        self.assertEqual(result["wrong_key_count"], 12)
        self.assertGreater(result["owner_best_score"], result["best_wrong_score"])

    def test_aisb_ambiguity_set_rejects_empty_candidate_set(self) -> None:
        self.assertEqual(
            _candidate_ambiguity_sequences(
                [],
                residual_threshold=0.006,
                near_tie_ratio=1.25,
                per_cluster_limit=3,
                max_sequences=512,
            ),
            [],
        )
        result = _ambiguity_random_non_burst_case(5, noise_std=0.016)
        self.assertTrue(result["pass"])
        self.assertEqual(result["ambiguity_sequence_count"], 0)

    def test_aisb_ambiguity_set_scores_targeted_shifted_window_case(self) -> None:
        result = _targeted_shifted_window_case(0, noise_std=0.012)

        self.assertTrue(result["pass"])
        self.assertGreater(result["ambiguity_sequence_count"], 1)
        self.assertTrue(result["ambiguity_contains_truth"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_capacity_probe_separates_owner_with_larger_message_space(self) -> None:
        result = _capacity_case(
            3,
            message_space_size=16,
            wrong_key_count=24,
            noise_std=0.016,
        )

        self.assertTrue(result["pass"])
        self.assertTrue(result["alignment_exact"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_capacity_mismatch_probe_preserves_scoring_at_gamma_half(self) -> None:
        result = _capacity_mismatch_case(
            3,
            message_space_size=32,
            wrong_key_count=24,
            gamma=0.5,
            noise_std=0.012,
        )

        self.assertTrue(result["pass"])
        self.assertTrue(result["alignment_exact"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_stress_mismatch_payload_probe_survives_harder_edits(self) -> None:
        result = _stress_mismatch_payload_case(
            2,
            message_space_size=16,
            wrong_key_count=24,
            gamma=0.5,
            noise_std=0.012,
        )

        self.assertTrue(result["pass"])
        self.assertTrue(result["alignment_exact"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_long_sequence_probe_preserves_owner_margin(self) -> None:
        result = _long_sequence_case(
            1,
            burst_count=10,
            message_space_size=32,
            wrong_key_count=16,
            gamma=0.5,
            noise_std=0.012,
        )

        self.assertTrue(result["pass"])
        self.assertTrue(result["alignment_exact"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_long_ambiguity_probe_scores_shared_alignment_set(self) -> None:
        result = _long_targeted_ambiguity_case(
            1,
            burst_count=10,
            message_space_size=16,
            wrong_key_count=12,
            gamma=0.5,
            noise_std=0.012,
        )

        self.assertTrue(result["pass"])
        self.assertGreater(result["ambiguity_sequence_count"], 1)
        self.assertTrue(result["ambiguity_contains_truth"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_multi_ambiguity_probe_scores_nonunique_alignment_set(self) -> None:
        result = _multi_ambiguity_case(
            1,
            burst_count=10,
            message_space_size=8,
            wrong_key_count=8,
            gamma=0.5,
            noise_std=0.012,
        )

        self.assertTrue(result["pass"])
        self.assertGreater(result["ambiguity_sequence_count"], 1)
        self.assertTrue(result["ambiguity_contains_truth"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_exact_score_only_scale_probe_preserves_exhaustive_scoring(self) -> None:
        result = _exact_scale_case(
            0,
            burst_count=10,
            message_space_size=32,
            wrong_key_count=24,
            gamma=0.5,
            noise_std=0.012,
        )

        self.assertTrue(result["pass"])
        self.assertTrue(result["score_only_matches_full_owner"])
        self.assertTrue(result["alignment_exact"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_threshold_margin_probe_recovers_high_mismatch_false_negative(self) -> None:
        result = _high_mismatch_case(1, gamma=0.8, noise_std=0.012)
        by_threshold = {row["threshold"]: row for row in result["threshold_results"]}

        self.assertFalse(by_threshold[0.006]["alignment_exact"])
        self.assertEqual(by_threshold[0.006]["false_negative"], 1)
        self.assertTrue(by_threshold[0.00625]["alignment_exact"])
        self.assertEqual(by_threshold[0.00625]["false_positive"], 0)
        self.assertGreater(result["best_false_residual"], 0.01)

    def test_aisb_threshold_margin_random_non_burst_rejects_candidate_threshold(self) -> None:
        result = _threshold_random_non_burst_case(4, gamma=0.8, noise_std=0.012)
        by_threshold = {row["threshold"]: row for row in result["threshold_results"]}

        self.assertEqual(by_threshold[0.00625]["accepted_count"], 0)
        self.assertGreater(result["best_residual"], 0.01)

    def test_aisb_exact_search_accepts_predeclared_threshold_candidate(self) -> None:
        result = _exact_scale_case(
            1,
            burst_count=12,
            message_space_size=16,
            wrong_key_count=12,
            gamma=0.8,
            noise_std=0.012,
            residual_threshold=0.00625,
        )

        self.assertTrue(result["pass"])
        self.assertTrue(result["alignment_exact"])
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertGreater(result["score_margin"], 0.02)

    def test_aisb_exact_search_parallel_matches_serial(self) -> None:
        serial = _exact_scale_case(
            0,
            burst_count=8,
            message_space_size=8,
            wrong_key_count=4,
            gamma=0.5,
            noise_std=0.012,
            residual_threshold=0.006,
            workers=1,
        )
        parallel = _exact_scale_case(
            0,
            burst_count=8,
            message_space_size=8,
            wrong_key_count=4,
            gamma=0.5,
            noise_std=0.012,
            residual_threshold=0.006,
            workers=2,
        )

        self.assertEqual(parallel["parallel_workers"], 2)
        self.assertEqual(serial["owner_best_message"], parallel["owner_best_message"])
        self.assertEqual(serial["best_wrong_message"], parallel["best_wrong_message"])
        self.assertAlmostEqual(serial["owner_best_score"], parallel["owner_best_score"], places=15)
        self.assertAlmostEqual(serial["best_wrong_score"], parallel["best_wrong_score"], places=15)
        self.assertEqual(serial["pass"], parallel["pass"])

    def test_aisb_threshold_candidate_exact_summary_counts_passes(self) -> None:
        cases = [
            {
                "pass": True,
                "alignment_exact": True,
                "owner_best_message": "message_0",
                "true_message": "message_0",
                "score_margin": 0.04,
                "candidate_count": 4160,
                "total_exact_score_count": 4160,
                "bounded_abandoned_count": 10,
            },
            {
                "pass": True,
                "alignment_exact": True,
                "owner_best_message": "message_1",
                "true_message": "message_1",
                "score_margin": 0.08,
                "candidate_count": 4160,
                "total_exact_score_count": 4160,
                "bounded_abandoned_count": 20,
            },
        ]
        summary = _threshold_candidate_exact_summarize(cases)

        self.assertEqual(summary["pass_count"], 2)
        self.assertEqual(summary["alignment_exact_count"], 2)
        self.assertEqual(summary["owner_message_recovery_count"], 2)
        self.assertEqual(summary["score_margin_min"], 0.04)

    def test_aisb_sequence_consistency_filters_isolated_false_bursts(self) -> None:
        thresholds = [0.0125]
        high_cases = [
            _high_mismatch_sequence_case(
                index,
                gamma=50.0,
                noise_std=0.012,
                thresholds=thresholds,
                burst_count=12,
                min_sequence_support=12,
            )
            for index in range(2)
        ]
        random_cases = [
            _random_non_burst_sequence_case(
                index,
                gamma=50.0,
                noise_std=0.012,
                thresholds=thresholds,
                min_sequence_support=12,
            )
            for index in range(16)
        ]

        high_summary = _sequence_summarize_high(high_cases, thresholds=thresholds)
        random_summary = _sequence_summarize_random(random_cases, thresholds=thresholds)

        self.assertEqual(high_summary["by_threshold"]["0.0125"]["sequence_alignment_exact_count"], 2)
        self.assertEqual(high_summary["by_threshold"]["0.0125"]["sequence_false_negative_total"], 0)
        self.assertEqual(random_summary["by_threshold"]["0.0125"]["sequence_false_positive_total"], 0)
        self.assertLessEqual(
            random_summary["by_threshold"]["0.0125"]["sequence_support_max"],
            1,
        )

    def test_aisb_sequence_supported_exact_probe_closes_scoring_chain(self) -> None:
        result = _sequence_supported_exact_case(
            0,
            burst_count=8,
            message_space_size=8,
            wrong_key_count=4,
            gamma=5.0,
            noise_std=0.012,
            residual_threshold=0.0125,
            filler_multiplier=2,
            min_sequence_support=8,
            workers=1,
        )

        self.assertTrue(result["alignment_exact"])
        self.assertGreaterEqual(result["sequence_support_count"], 8)
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertEqual(result["global_best_role"], "owner")
        self.assertGreater(result["score_margin"], 0.02)
        self.assertTrue(result["pass"])

    def test_aisb_sequence_ambiguity_exact_probe_handles_nonunique_alignment(self) -> None:
        result = _sequence_ambiguity_exact_case(
            4,
            burst_count=12,
            message_space_size=24,
            wrong_key_count=4,
            gamma=50.0,
            noise_std=0.012,
            residual_threshold=0.0125,
            near_tie_ratio=3.0,
            per_cluster_limit=3,
            max_sequences=128,
            filler_multiplier=3,
            min_sequence_support=12,
            workers=1,
        )

        self.assertTrue(result["truth_sequence_in_ambiguity"])
        self.assertGreater(result["ambiguity_sequence_count"], 1)
        self.assertEqual(result["owner_best_message"], result["true_message"])
        self.assertEqual(result["global_best_role"], "owner")
        self.assertGreater(result["score_margin"], 0.02)
        self.assertTrue(result["pass"])

    def test_aisb_sequence_ambiguity_top_two_scan_recovers_close_deletion_candidate(self) -> None:
        case_index = 50
        burst_count = 16
        message_space_size = 64
        owner_key = f"owner_pruned_{burst_count}_{message_space_size}_{case_index}"
        states, burst_plan = _build_long_sequence(
            case_index,
            owner_key=owner_key,
            message=f"message_{case_index}",
            burst_count=burst_count,
            filler_multiplier=7,
        )
        observations = generate_observations(
            states,
            make_random_channel(141000 + 100 * message_space_size + case_index, relation_count=16, noise_std=0.22),
            seed=142000 + 100 * message_space_size + case_index,
        )
        observations = _apply_quadratic_mismatch(observations, states, gamma=100.0)
        edited, _, truth = _apply_long_edits(observations, burst_plan, case_index=case_index)
        truth_set = set(truth)

        top_one = scan_burst_candidates(
            edited,
            make_redundant_templates(),
            allow_single_deletion=True,
            top_k_per_start=1,
        )
        top_two = scan_burst_candidates(
            edited,
            make_redundant_templates(),
            allow_single_deletion=True,
            top_k_per_start=2,
        )

        top_one_sequences = _supported_ambiguity_sequences(
            top_one,
            residual_threshold=0.0125,
            near_tie_ratio=5.0,
            per_cluster_limit=3,
            max_sequences=512,
            min_sequence_support=12,
        )
        top_two_sequences = _supported_ambiguity_sequences(
            top_two,
            residual_threshold=0.0125,
            near_tie_ratio=5.0,
            per_cluster_limit=3,
            max_sequences=512,
            min_sequence_support=12,
        )

        self.assertFalse(any(truth_set <= {_candidate_key(candidate) for candidate in sequence} for sequence in top_one_sequences))
        self.assertTrue(any(truth_set <= {_candidate_key(candidate) for candidate in sequence} for sequence in top_two_sequences))
        self.assertIn((284, "redundant_beta", 6), {_candidate_key(candidate) for candidate in top_two})

    def test_diagnostic_pruned_selector_keeps_all_owner_and_one_per_wrong_key(self) -> None:
        cheap_scored = [
            (("owner", "owner_key", "message_0"), -10.0),
            (("owner", "owner_key", "message_1"), -20.0),
            (("wrong", "wrong_a", "message_0"), -1.0),
            (("wrong", "wrong_a", "message_1"), -2.0),
            (("wrong", "wrong_b", "message_0"), -3.0),
            (("wrong", "wrong_b", "message_1"), -4.0),
        ]
        selected = _select_diagnostic_pruned_candidates(
            cheap_scored,
            top_k_global=0,
            top_k_owner=2,
            top_k_per_wrong_key=1,
        )

        self.assertEqual(
            set(selected),
            {
                ("owner", "owner_key", "message_0"),
                ("owner", "owner_key", "message_1"),
                ("wrong", "wrong_a", "message_0"),
                ("wrong", "wrong_b", "message_0"),
            },
        )

    def test_diagnostic_pruned_parallel_matches_serial(self) -> None:
        common_kwargs = dict(
            burst_count=12,
            message_space_size=16,
            wrong_key_count=8,
            gamma=40.0,
            noise_std=0.014,
            residual_threshold=0.0125,
            near_tie_ratio=3.0,
            per_cluster_limit=3,
            max_sequences=128,
            filler_multiplier=3,
            min_sequence_support=12,
            scoring_mode="diagnostic_pruned_c",
            diagnostic_top_k_global=8,
            diagnostic_top_k_owner=8,
            diagnostic_top_k_per_wrong_key=1,
        )

        serial = _sequence_ambiguity_exact_case(5, workers=1, **common_kwargs)
        parallel = _sequence_ambiguity_exact_case(5, workers=2, **common_kwargs)

        self.assertEqual(parallel["parallel_workers"], 2)
        self.assertEqual(parallel["owner_best_message"], serial["owner_best_message"])
        self.assertEqual(parallel["best_wrong_message"], serial["best_wrong_message"])
        self.assertEqual(parallel["global_best_role"], serial["global_best_role"])
        self.assertEqual(parallel["global_best_message"], serial["global_best_message"])
        self.assertEqual(parallel["pass"], serial["pass"])
        self.assertAlmostEqual(parallel["owner_best_score"], serial["owner_best_score"], places=15)
        self.assertAlmostEqual(parallel["best_wrong_score"], serial["best_wrong_score"], places=15)
        self.assertAlmostEqual(parallel["score_margin"], serial["score_margin"], places=15)
        self.assertEqual(parallel["total_exact_score_count"], serial["total_exact_score_count"])
        self.assertEqual(parallel["bounded_abandoned_count"], serial["bounded_abandoned_count"])

    def test_aisb_sequence_ambiguity_ordered_bounded_matches_exact_winner(self) -> None:
        common_kwargs = dict(
            burst_count=12,
            message_space_size=24,
            wrong_key_count=12,
            gamma=50.0,
            noise_std=0.012,
            residual_threshold=0.0125,
            near_tie_ratio=3.0,
            per_cluster_limit=3,
            max_sequences=128,
            filler_multiplier=3,
            min_sequence_support=12,
            workers=1,
        )
        exact = _sequence_ambiguity_exact_case(
            4,
            scoring_mode="candidate_parallel",
            **common_kwargs,
        )
        ordered = _sequence_ambiguity_exact_case(
            4,
            scoring_mode="ordered_bounded",
            **common_kwargs,
        )
        native_ordered = _sequence_ambiguity_exact_case(
            4,
            scoring_mode="ordered_bounded_c",
            **common_kwargs,
        )
        native_ordered_parallel = _sequence_ambiguity_exact_case(
            4,
            scoring_mode="ordered_bounded_c",
            **{**common_kwargs, "workers": 2},
        )
        native_global_ordered = _sequence_ambiguity_exact_case(
            4,
            scoring_mode="ordered_bounded_global_c",
            **common_kwargs,
        )
        margin_proof = _sequence_ambiguity_exact_case(
            4,
            scoring_mode="margin_proof_c",
            **common_kwargs,
        )

        self.assertEqual(ordered["owner_best_message"], exact["owner_best_message"])
        self.assertEqual(ordered["best_wrong_message"], exact["best_wrong_message"])
        self.assertEqual(ordered["global_best_role"], exact["global_best_role"])
        self.assertEqual(ordered["global_best_message"], exact["global_best_message"])
        self.assertEqual(ordered["pass"], exact["pass"])
        self.assertAlmostEqual(ordered["owner_best_score"], exact["owner_best_score"], places=15)
        self.assertAlmostEqual(ordered["best_wrong_score"], exact["best_wrong_score"], places=15)
        self.assertAlmostEqual(ordered["score_margin"], exact["score_margin"], places=15)
        self.assertGreater(ordered["bounded_abandoned_count"], 0)
        self.assertEqual(native_ordered["owner_best_message"], exact["owner_best_message"])
        self.assertEqual(native_ordered["best_wrong_message"], exact["best_wrong_message"])
        self.assertEqual(native_ordered["global_best_role"], exact["global_best_role"])
        self.assertEqual(native_ordered["global_best_message"], exact["global_best_message"])
        self.assertEqual(native_ordered["pass"], exact["pass"])
        self.assertAlmostEqual(native_ordered["owner_best_score"], exact["owner_best_score"], places=15)
        self.assertAlmostEqual(native_ordered["best_wrong_score"], exact["best_wrong_score"], places=15)
        self.assertAlmostEqual(native_ordered["score_margin"], exact["score_margin"], places=15)
        self.assertGreater(native_ordered["bounded_abandoned_count"], 0)
        self.assertEqual(native_ordered_parallel["parallel_workers"], 2)
        self.assertEqual(native_ordered_parallel["owner_best_message"], exact["owner_best_message"])
        self.assertEqual(native_ordered_parallel["best_wrong_message"], exact["best_wrong_message"])
        self.assertEqual(native_ordered_parallel["global_best_role"], exact["global_best_role"])
        self.assertEqual(native_ordered_parallel["global_best_message"], exact["global_best_message"])
        self.assertEqual(native_ordered_parallel["pass"], exact["pass"])
        self.assertAlmostEqual(native_ordered_parallel["owner_best_score"], exact["owner_best_score"], places=15)
        self.assertAlmostEqual(native_ordered_parallel["best_wrong_score"], exact["best_wrong_score"], places=15)
        self.assertAlmostEqual(native_ordered_parallel["score_margin"], exact["score_margin"], places=15)
        self.assertGreater(native_ordered_parallel["bounded_abandoned_count"], 0)
        self.assertEqual(native_global_ordered["owner_best_message"], exact["owner_best_message"])
        self.assertEqual(native_global_ordered["best_wrong_message"], exact["best_wrong_message"])
        self.assertEqual(native_global_ordered["global_best_role"], exact["global_best_role"])
        self.assertEqual(native_global_ordered["global_best_message"], exact["global_best_message"])
        self.assertEqual(native_global_ordered["pass"], exact["pass"])
        self.assertAlmostEqual(native_global_ordered["owner_best_score"], exact["owner_best_score"], places=15)
        self.assertAlmostEqual(native_global_ordered["best_wrong_score"], exact["best_wrong_score"], places=15)
        self.assertAlmostEqual(native_global_ordered["score_margin"], exact["score_margin"], places=15)
        self.assertGreater(native_global_ordered["bounded_abandoned_count"], 0)
        self.assertEqual(margin_proof["owner_best_message"], exact["owner_best_message"])
        self.assertEqual(margin_proof["global_best_role"], "owner")
        self.assertEqual(margin_proof["global_best_message"], exact["global_best_message"])
        self.assertEqual(margin_proof["pass"], exact["pass"])
        self.assertAlmostEqual(margin_proof["owner_best_score"], exact["owner_best_score"], places=15)
        self.assertGreater(margin_proof["score_margin"], 0.02)
        self.assertTrue(margin_proof["score_margin_is_lower_bound"])
        self.assertTrue(margin_proof["best_wrong_score_is_upper_bound"])
        self.assertEqual(margin_proof["candidate_count"], exact["candidate_count"])
        self.assertGreater(margin_proof["bounded_abandoned_count"], 0)

    def test_aisb_sequence_ambiguity_progress_jsonl_is_diagnostic_only(self) -> None:
        progress_events: list[dict[str, object]] = []
        result = _sequence_ambiguity_exact_case(
            4,
            burst_count=12,
            message_space_size=8,
            wrong_key_count=4,
            gamma=50.0,
            noise_std=0.012,
            residual_threshold=0.0125,
            near_tie_ratio=3.0,
            per_cluster_limit=3,
            max_sequences=128,
            filler_multiplier=3,
            min_sequence_support=12,
            workers=1,
            scoring_mode="ordered_bounded_global_c",
            progress_callback=progress_events.append,
            progress_interval=1,
        )

        event_names = [event["event"] for event in progress_events]
        self.assertTrue(result["pass"])
        self.assertIn("case_start", event_names)
        self.assertIn("case_scoring_start", event_names)
        self.assertIn("scoring_candidate_start", event_names)
        self.assertIn("scoring_candidate_finish", event_names)
        self.assertNotIn("paper_claim", progress_events[0])

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "progress" / "events.jsonl"
            _append_jsonl(output, {"event": "case_result", "pass": result["pass"]})
            parsed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(parsed, [{"event": "case_result", "pass": True}])

    def test_aisb_stress_grid_summary_reports_alignment_and_margin(self) -> None:
        cases = [
            {
                "pass": True,
                "alignment_exact": True,
                "owner_message_recovered": True,
                "score_margin": 0.04,
            },
            {
                "pass": False,
                "alignment_exact": False,
                "owner_message_recovered": True,
                "score_margin": 0.01,
            },
        ]
        summary = _summarize_grid_cell(cases)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["pass_count"], 1)
        self.assertEqual(summary["alignment_exact_count"], 1)
        self.assertEqual(summary["owner_message_recovery_count"], 2)
        self.assertEqual(summary["score_margin_min"], 0.01)

    def test_aisb_stress_grid_smoke_case_runs_cpu_only(self) -> None:
        grid = _stress_grid(
            gammas=[0.5],
            noise_stds=[0.012],
            message_space_size=8,
            wrong_key_count=4,
            case_count=1,
        )
        cell = grid["gamma_0.5_noise_0.012"]

        self.assertEqual(cell["case_count"], 1)
        self.assertEqual(cell["pass_count"], 1)
        self.assertEqual(cell["alignment_exact_count"], 1)
        self.assertGreater(cell["score_margin_min"], 0.02)

    def test_aisb_stress_grid_ambiguity_set_smoke_case_runs_cpu_only(self) -> None:
        grid = _ambiguity_scoring_grid(
            gammas=[50.0],
            noise_stds=[0.012],
            message_space_size=8,
            wrong_key_count=4,
            case_count=1,
            workers=1,
        )
        cell = grid["gamma_50_noise_0.012"]

        self.assertEqual(cell["case_count"], 1)
        self.assertEqual(cell["pass_count"], 1)
        self.assertEqual(cell["truth_sequence_covered_count"], 1)
        self.assertEqual(cell["owner_message_recovery_count"], 1)
        self.assertGreater(cell["score_margin_min"], 0.02)

    def test_aisb_stress_grid_ambiguity_set_does_not_truncate_case_count(self) -> None:
        grid = _ambiguity_scoring_grid(
            gammas=[50.0],
            noise_stds=[0.012],
            message_space_size=8,
            wrong_key_count=4,
            case_count=4,
            workers=1,
        )
        cell = grid["gamma_50_noise_0.012"]

        self.assertEqual(cell["case_count"], 4)
        self.assertEqual(len(cell["cases"]), 4)
        self.assertEqual(cell["pass_count"], 4)
        self.assertEqual(cell["truth_sequence_covered_count"], 4)
        self.assertEqual(cell["owner_message_recovery_count"], 4)
        self.assertGreater(cell["score_margin_min"], 0.02)


if __name__ == "__main__":
    unittest.main()
