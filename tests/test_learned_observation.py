from __future__ import annotations

import copy
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

from src.sc_sstw_feasibility.learned_observation import (  # noqa: E402
    ACQUISITION_PROTOCOL_SHA256,
    CALIBRATION_INDICES,
    CONFIG_CANONICAL_SHA256,
    FORBIDDEN_MP4_SHA256,
    FrozenObservationFrontend,
    HELD_OUT_IDS,
    L1_IDS,
    NULL_IDS,
    PER_VIDEO_HELD_OUT_INDICES,
    TEMPORAL_POINTS,
    TRAIN_IDS,
    VALIDATION_IDS,
    acquire_and_freeze_ambiguity,
    assert_stage_dataset_access,
    audit_truth_success_after_freeze,
    calibrate_from_frozen_ambiguity,
    canonical_json_bytes,
    fit_train_only_normalizer,
    load_frozen_frontend,
    read_frozen_ambiguity,
    sha256_bytes,
    static_contract_report,
    validate_learned_observation_config,
    validate_new_dataset_artifact,
    watermarked_gate_checks,
)


class LearnedObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs" / "learned_observation_frontend.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_single_config_freezes_carrier_dataset_and_acquisition_digest(self) -> None:
        validate_learned_observation_config(self.config)
        protocol = self.config["acquisition"]["protocol"]
        self.assertEqual(sha256_bytes(canonical_json_bytes(protocol)), ACQUISITION_PROTOCOL_SHA256)
        self.assertEqual(self.config["acquisition"]["protocol_sha256"], ACQUISITION_PROTOCOL_SHA256)
        self.assertEqual(tuple(self.config["carrier"]["public_calibration_indices"]), CALIBRATION_INDICES)
        self.assertEqual(tuple(self.config["carrier"]["per_video_calibration_held_out_indices"]), PER_VIDEO_HELD_OUT_INDICES)
        self.assertEqual(tuple(self.config["dataset"]["l1_permitted_ids"]), TRAIN_IDS + VALIDATION_IDS)
        self.assertEqual(tuple(self.config["dataset"]["l2_permitted_only_after_l1_pass_ids"]), HELD_OUT_IDS + NULL_IDS)

    def test_all_declared_semantic_config_mutations_fail_closed(self) -> None:
        mutations = []
        paths = [
            (("carrier", "basis_formula_x"), "changed_basis"),
            (("carrier", "apply_to"), "scheduler_oracle"),
            (("generation", "negative_prompt"), "changed"),
            (("extractor", "architecture"), ["direct_key_decoder"]),
            (("training", "learning_rate"), 0.01),
            (("training", "outer_relation_loss", "formula"), "fit_all_points"),
            (("features", "color_transform", "Y"), [1.0, 0.0, 0.0]),
            (("quarantine", "forbidden_local_roots"), []),
            (("dataset", "l2_permitted_only_after_l1_pass_ids"), [41007]),
            (("history_lock", "overall_method_conclusion"), "PASS"),
            (("gate_thresholds", "maximum_aisb_residual"), 0.3),
        ]
        for path, value in paths:
            changed = copy.deepcopy(self.config)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["dataset"]["items"][0]["prompt"] = "changed prompt"
        mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(Exception):
                validate_learned_observation_config(changed)
            with self.assertRaises(Exception):
                static_contract_report(changed)
        self.assertEqual(sha256_bytes(canonical_json_bytes(self.config)), CONFIG_CANONICAL_SHA256)

    def test_forward_and_observe_signatures_have_no_metadata_or_truth_inputs(self) -> None:
        self.assertEqual(tuple(inspect.signature(FrozenObservationFrontend.forward).parameters), ("self", "standardized_feature_matrix"))
        self.assertEqual(tuple(inspect.signature(FrozenObservationFrontend.observe_saved_mp4).parameters), ("self", "mp4_path"))
        forbidden = set(self.config["extractor"]["forbidden_forward_inputs"])
        self.assertTrue({"dataset_id", "q", "key", "message", "alignment", "truth", "source_index"} <= forbidden)

    def test_shared_frontend_is_identical_input_equal_and_permutation_equivariant(self) -> None:
        first_weight = tuple(tuple((row + column + 1) * 1e-4 for column in range(30)) for row in range(16))
        second_weight = tuple(tuple((row + column + 1) * 2e-3 for column in range(16)) for row in range(2))
        frontend = FrozenObservationFrontend(
            mean=(0.0,) * 30, std=(1.0,) * 30,
            first_weight=first_weight, first_bias=(0.0,) * 16,
            second_weight=second_weight, second_bias=(0.0,) * 2,
            artifact_sha256="0" * 64,
        )
        identical = [[0.25] * 30 for _ in range(13)]
        identical_output = frontend.forward(identical)
        self.assertTrue(all(row == identical_output[0] for row in identical_output))
        features = [[float(index + column) / 100.0 for column in range(30)] for index in range(13)]
        permutation = [12, 0, 5, 1, 9, 2, 8, 3, 7, 4, 6, 10, 11]
        original = frontend.forward(features)
        permuted = frontend.forward([features[index] for index in permutation])
        self.assertEqual(permuted, [original[index] for index in permutation])

    def test_normalizer_reads_exactly_four_train_videos(self) -> None:
        features = {
            dataset_id: [[float(dataset_id + row + column) for column in range(30)] for row in range(13)]
            for dataset_id in TRAIN_IDS
        }
        means, stds = fit_train_only_normalizer(features)
        self.assertEqual(len(means), 30)
        self.assertEqual(len(stds), 30)
        with self.assertRaises(Exception):
            fit_train_only_normalizer({**features, VALIDATION_IDS[0]: features[TRAIN_IDS[0]]})
        missing = dict(features)
        missing.pop(TRAIN_IDS[-1])
        with self.assertRaises(Exception):
            fit_train_only_normalizer(missing)

    def test_l1_cannot_touch_heldout_or_null_and_l2_requires_pass(self) -> None:
        assert_stage_dataset_access("gpu_train_validation", L1_IDS)
        for invalid in ([L1_IDS[0]], list(reversed(L1_IDS)), [*L1_IDS, HELD_OUT_IDS[0]]):
            with self.assertRaises(Exception):
                assert_stage_dataset_access("gpu_train_validation", invalid)
        with self.assertRaises(Exception):
            assert_stage_dataset_access("gpu_public_held_out_and_null", HELD_OUT_IDS + NULL_IDS, l1_gate_pass=False)
        assert_stage_dataset_access("gpu_public_held_out_and_null", HELD_OUT_IDS + NULL_IDS, l1_gate_pass=True)
        with self.assertRaises(Exception):
            assert_stage_dataset_access("gpu_public_held_out_and_null", list(reversed(HELD_OUT_IDS + NULL_IDS)), l1_gate_pass=True)

    def test_acquisition_is_frozen_before_c_only_calibration_and_truth_audit(self) -> None:
        matrix = ((1.2, 0.2), (-0.1, 0.9))
        bias = (0.15, -0.08)
        observation = [
            [matrix[0][0] * q[0] + matrix[0][1] * q[1] + bias[0], matrix[1][0] * q[0] + matrix[1][1] * q[1] + bias[1]]
            for q in TEMPORAL_POINTS
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "ambiguity.json"
            frozen = acquire_and_freeze_ambiguity(observation, artifact, self.config)
            expected_artifact_sha256 = frozen["artifact_sha256"]
            envelope = read_frozen_ambiguity(artifact, expected_artifact_sha256=expected_artifact_sha256)
            self.assertEqual(envelope["payload"]["protocol_sha256"], ACQUISITION_PROTOCOL_SHA256)
            self.assertTrue(audit_truth_success_after_freeze(artifact, expected_artifact_sha256, self.config))
            from src.sc_sstw_feasibility.calibration import calibrate_from_pilot_pairs as actual_fit
            with mock.patch("src.sc_sstw_feasibility.learned_observation.calibrate_from_pilot_pairs", wraps=actual_fit) as fitted:
                metrics = calibrate_from_frozen_ambiguity(observation, artifact, expected_artifact_sha256, self.config)
            pairs = fitted.call_args.args[0]
            self.assertEqual([tuple(pair[0]) for pair in pairs], [TEMPORAL_POINTS[index] for index in CALIBRATION_INDICES])
            self.assertEqual(len(pairs), 4)
            self.assertLessEqual(metrics["public_calibration_held_out_mse"], 0.02)
            self.assertGreaterEqual(metrics["fitted_affine_second_singular_value"], 0.05)
            with self.assertRaises(FileExistsError):
                acquire_and_freeze_ambiguity(observation, artifact, self.config)

    def test_complete_watermarked_cpu_relation_fixture_passes_all_frozen_checks(self) -> None:
        matrix = ((1.1, -0.15), (0.25, 0.95))
        bias = (-0.04, 0.12)
        observation = [
            [matrix[0][0] * q[0] + matrix[0][1] * q[1] + bias[0], matrix[1][0] * q[0] + matrix[1][1] * q[1] + bias[1]]
            for q in TEMPORAL_POINTS
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            result = watermarked_gate_checks(observation, Path(tmpdir) / "ambiguity.json", self.config)
        self.assertTrue(result["case_pass"], result)
        self.assertTrue(result["truth_acquisition_success"])

    def test_calibration_rejects_different_observation_than_acquisition(self) -> None:
        observation = [[float(q[0]), float(q[1])] for q in TEMPORAL_POINTS]
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "ambiguity.json"
            frozen = acquire_and_freeze_ambiguity(observation, artifact, self.config)
            expected_artifact_sha256 = frozen["artifact_sha256"]
            changed = copy.deepcopy(observation)
            changed[12][0] += 0.001
            with self.assertRaises(Exception):
                calibrate_from_frozen_ambiguity(changed, artifact, expected_artifact_sha256, self.config)
            envelope = json.loads(artifact.read_text(encoding="utf-8"))
            envelope["payload"]["accepted_candidates"][0]["residual"] += 0.001
            envelope["payload_sha256"] = sha256_bytes(canonical_json_bytes(envelope["payload"]))
            artifact.write_text(json.dumps(envelope, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(Exception):
                calibrate_from_frozen_ambiguity(observation, artifact, expected_artifact_sha256, self.config)

    def test_generation_manifest_binds_config_commit_digest_and_no_derivative(self) -> None:
        old_path = Path("/tmp/sc_sstw_audit_20260807T022258Z/20260807T022258Z_cfff6893/artifacts/watermarked.mp4")
        with self.assertRaises(Exception):
            validate_new_dataset_artifact(old_path, Path("missing.json"), self.config, expected_repository_commit="a" * 40, expected_manifest_sha256="b" * 64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "new.mp4"
            manifest_path = Path(tmpdir) / "generation_manifest.json"
            path.write_bytes(b"new public fixture")
            item = self.config["dataset"]["items"][0]
            manifest = {
                "protocol_id": self.config["protocol_id"], "config_sha256": CONFIG_CANONICAL_SHA256,
                "repository_commit": "a" * 40, "dataset_id": item["dataset_id"],
                "prompt": item["prompt"], "seed": item["seed"], "carrier": item["carrier"],
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "derived_from": None,
                "generation_call": "repository_formal_cli", "injection_records_sha256": "c" * 64,
            }
            def write_manifest() -> str:
                manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
                return hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            expected_manifest_sha256 = write_manifest()
            validate_new_dataset_artifact(path, manifest_path, self.config, expected_repository_commit="a" * 40, expected_manifest_sha256=expected_manifest_sha256)
            manifest["derived_from"] = "old-run"
            expected_manifest_sha256 = write_manifest()
            with self.assertRaises(Exception):
                validate_new_dataset_artifact(path, manifest_path, self.config, expected_repository_commit="a" * 40, expected_manifest_sha256=expected_manifest_sha256)
            manifest["derived_from"] = None
            expected_manifest_sha256 = write_manifest()
            with self.assertRaises(Exception):
                validate_new_dataset_artifact(path, manifest_path, self.config, expected_repository_commit="d" * 40, expected_manifest_sha256=expected_manifest_sha256)
            with self.assertRaises(Exception):
                validate_new_dataset_artifact(path, manifest_path, self.config, expected_repository_commit="a" * 40, expected_manifest_sha256="e" * 64)
        self.assertEqual(set(self.config["quarantine"]["forbidden_mp4_sha256"]), set(FORBIDDEN_MP4_SHA256))

    def test_weights_require_external_sha_config_and_strict_finite_schema(self) -> None:
        payload = {
            "artifact_kind": "public_relation_observation_frontend_step_2000",
            "config_sha256": CONFIG_CANONICAL_SHA256,
            "train_artifact_sha256_by_dataset_id": {str(dataset_id): format(dataset_id, "064x") for dataset_id in TRAIN_IDS},
            "normalizer_fit_dataset_ids": list(TRAIN_IDS), "optimizer_steps": 2000,
            "final_training_loss": 0.5,
            "normalizer": {"mean": [0.0] * 30, "std": [1.0] * 30},
            "model": {"first_weight": [[0.0] * 30 for _ in range(16)], "first_bias": [0.0] * 16, "second_weight": [[0.0] * 16 for _ in range(2)], "second_bias": [0.0] * 2},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.json"
            path.write_bytes(canonical_json_bytes(payload) + b"\n")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = load_frozen_frontend(path, expected_weights_sha256=expected, expected_config_sha256=CONFIG_CANONICAL_SHA256)
            self.assertEqual(loaded.artifact_sha256, expected)
            with self.assertRaises(Exception):
                load_frozen_frontend(path, expected_weights_sha256="0" * 64, expected_config_sha256=CONFIG_CANONICAL_SHA256)
            with self.assertRaises(Exception):
                load_frozen_frontend(path, expected_weights_sha256=expected, expected_config_sha256="0" * 64)
            payload["model"]["second_bias"] = [float("nan"), 0.0]
            path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8") + b"\n")
            changed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(Exception):
                load_frozen_frontend(path, expected_weights_sha256=changed_sha, expected_config_sha256=CONFIG_CANONICAL_SHA256)

    def test_decoder_encoder_and_training_are_single_path_static_contracts(self) -> None:
        source = (ROOT / "src" / "sc_sstw_feasibility" / "learned_observation.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('iio.imiter(mp4_path, plugin="FFMPEG")'), 1)
        self.assertIn("export_to_video(frames, str(mp4_path), fps=8, quality=5.0, bitrate=None, macro_block_size=16)", source)
        self.assertNotIn("cv2", source)
        self.assertIn("for _step in range(2000):", source)
        self.assertIn("optimizer_steps != 2000", source)
        self.assertIn('with output_path.open("xb")', source)
        self.assertIn("expected_artifact_sha256", source)

    def test_static_report_passes_but_never_admits_gpu(self) -> None:
        report = static_contract_report(self.config)
        self.assertTrue(report["gate_pass"])
        self.assertFalse(report["gpu_admission"])
        self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
