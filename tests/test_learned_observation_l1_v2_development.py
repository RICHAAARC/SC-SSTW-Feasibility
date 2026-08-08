from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sc_sstw_feasibility.learned_observation_l1_v2 as gate


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def good_features() -> dict[int, np.ndarray]:
    base = np.zeros((13, 30), dtype=float)
    base[:, :2] = np.asarray(gate.TEMPORAL_POINTS)
    random = np.random.default_rng(7)
    result: dict[int, np.ndarray] = {}
    for dataset_id in gate.FIT_IDS:
        value = base.copy()
        value[:, :2] += 1e-3 * random.normal(size=(13, 2))
        result[dataset_id] = value
    for dataset_id in gate.DEVELOPMENT_IDS:
        result[dataset_id] = base.copy()
    return result


class CliHarness:
    def __init__(self, owner: unittest.TestCase, features: dict[int, np.ndarray] | None = None):
        self.owner = owner
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        for relative_path in gate.REQUIRED_SOURCE_PATHS:
            destination = self.repo / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative_path, destination)
        package_init = self.repo / "src" / "sc_sstw_feasibility" / "__init__.py"
        package_init.write_text("", encoding="utf-8")
        self.git("init", "-q")
        self.git("config", "user.name", "Synthetic Test")
        self.git("config", "user.email", "synthetic@example.invalid")
        self.git("add", ".")
        self.git("commit", "-qm", "synthetic source snapshot")
        self.inputs = self.root / "synthetic-inputs"
        self.inputs.mkdir()
        self.write_features(features or good_features())
        self.manifest_path = self.root / "authorization-manifest.json"
        self.output_index = 0
        self.manifest = self.build_manifest()
        self.write_manifest()

    def close(self) -> None:
        self.temporary.cleanup()

    def git(self, *arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def write_features(self, features: dict[int, np.ndarray]) -> None:
        self.feature_paths: dict[int, Path] = {}
        for index, dataset_id in enumerate(gate.PERMITTED_INPUT_IDS):
            path = self.inputs / f"synthetic-feature-{index}.json"
            path.write_bytes(gate.canonical_json_bytes({"fixture": "pure_synthetic", "features": features[dataset_id].tolist()}))
            self.feature_paths[dataset_id] = path

    def build_manifest(self) -> dict[str, object]:
        head = self.git("rev-parse", "HEAD")
        tree = self.git("rev-parse", "HEAD^{tree}")
        source_files = {
            relative_path: sha256_file(self.repo / relative_path)
            for relative_path in gate.REQUIRED_SOURCE_PATHS
        }
        return {
            "schema_version": gate.MANIFEST_SCHEMA_VERSION,
            "protocol_id": gate.PROTOCOL_ID,
            "expected_source": {"head": head, "tree": tree},
            "source_files": source_files,
            "config": {"path": gate.CONFIG_PATH, "sha256": source_files[gate.CONFIG_PATH]},
            "inputs": [
                {
                    "dataset_id": dataset_id,
                    "path": str(self.feature_paths[dataset_id]),
                    "sha256": sha256_file(self.feature_paths[dataset_id]),
                }
                for dataset_id in gate.PERMITTED_INPUT_IDS
            ],
            "command_schema": {
                "runner": gate.RUNNER_PATH,
                "required_arguments": ["--manifest", "--output"],
                "optional_arguments": ["--source-commit"],
            },
            "output_schema": gate.OUTPUT_SCHEMA,
        }

    def refresh_manifest(self) -> None:
        self.manifest = self.build_manifest()
        self.write_manifest()

    def write_manifest(self) -> None:
        self.manifest_path.write_bytes(gate.canonical_json_bytes(self.manifest))

    def run(self, *, source_commit: str | None = None) -> tuple[subprocess.CompletedProcess[str], dict[str, object], Path]:
        self.output_index += 1
        output = self.root / f"output-{self.output_index}"
        command = [
            sys.executable,
            "-B",
            str(self.repo / gate.RUNNER_PATH),
            "--manifest",
            str(self.manifest_path),
            "--output",
            str(output),
        ]
        if source_commit is not None:
            command.extend(("--source-commit", source_commit))
        completed = subprocess.run(command, cwd=self.repo, capture_output=True, text=True)
        audit_path = output / "audit.json"
        self.owner.assertTrue(
            audit_path.is_file(),
            f"runner did not create audit package\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        return completed, audit, output


class TestFrozenScience(unittest.TestCase):
    def test_config_matches_single_frozen_source(self) -> None:
        config = json.loads((ROOT / gate.CONFIG_PATH).read_text(encoding="utf-8"))
        gate.validate_frozen_config(config)
        self.assertEqual(config["candidate_order"], ["A1", "A2"])
        self.assertEqual(config["states"]["success"], gate.SUCCESS_STATUS)
        self.assertEqual(config["states"]["invalid"], gate.INVALID_STATUS)

    def test_a1_median_mad_and_clip(self) -> None:
        features = np.tile(np.arange(13, dtype=float)[:, None], (1, 30))
        transformed = gate.transform(features, "A1")
        self.assertTrue(np.allclose(np.median(transformed, axis=0), 0))
        self.assertLessEqual(abs(transformed).max(), 6)

    def test_a2_boundary_and_length(self) -> None:
        features = np.tile(np.arange(13, dtype=float)[:, None], (1, 30))
        transformed = gate.transform(features, "A2")
        self.assertEqual(transformed.shape, (13, 30))
        self.assertTrue(np.allclose(transformed[1:-1], 0))
        self.assertTrue(np.allclose(transformed[0], -transformed[-1]))

    def test_absolute_boundary_values_pass(self) -> None:
        metrics = [copy.deepcopy(gate.ABSOLUTE_THRESHOLDS) for _ in gate.FIT_IDS]
        converted = [
            {
                "residual": item["max_residual"],
                "global_s2": item["min_global_s2"],
                "affine_s2": item["min_affine_s2"],
                "condition": item["max_condition"],
                "held_out_mse": item["max_held_out_mse"],
            }
            for item in metrics
        ]
        self.assertEqual(gate.derive_envelope(converted), gate.ABSOLUTE_THRESHOLDS)

    def test_derived_envelope_is_stricter(self) -> None:
        value = {
            "residual": 0.20,
            "global_s2": 0.20,
            "affine_s2": 0.08,
            "condition": 8.0,
            "held_out_mse": 0.01,
        }
        envelope = gate.derive_envelope([value] * 4)
        self.assertEqual(
            envelope,
            {
                "max_residual": 0.20,
                "min_global_s2": 0.20,
                "min_affine_s2": 0.08,
                "max_condition": 8.0,
                "max_held_out_mse": 0.01,
            },
        )

    def test_bad_training_value_cannot_derive_envelope(self) -> None:
        value = {
            "residual": 0.2500001,
            "global_s2": 0.20,
            "affine_s2": 0.08,
            "condition": 8.0,
            "held_out_mse": 0.01,
        }
        with self.assertRaisesRegex(gate.InvalidExperiment, "absolute threshold"):
            gate.derive_envelope([value] * 4)

    def test_degenerate_training_stops_before_development(self) -> None:
        zeros = {dataset_id: np.zeros((13, 30)) for dataset_id in gate.PERMITTED_INPUT_IDS}
        results, selected = gate.evaluate_gate(zeros)
        self.assertIsNone(selected)
        self.assertEqual([item["candidate"] for item in results], ["A1", "A2"])
        self.assertTrue(all(not item["training_absolute_gate_pass"] for item in results))
        self.assertTrue(all(not item["development_evaluated"] for item in results))

    def test_candidate_injection_is_rejected(self) -> None:
        with self.assertRaisesRegex(gate.InvalidExperiment, "only A1 and A2"):
            gate.transform(np.zeros((13, 30)), "A3")


class TestRealRunner(unittest.TestCase):
    def make_harness(self, features: dict[int, np.ndarray] | None = None) -> CliHarness:
        harness = CliHarness(self, features)
        self.addCleanup(harness.close)
        return harness

    def assert_invalid(self, audit: dict[str, object], reason: str) -> None:
        self.assertEqual(audit["status"], gate.INVALID_STATUS)
        self.assertFalse(audit["valid_experiment"])
        self.assertEqual(audit["reason_code"], reason)
        self.assertFalse(audit["science_metrics_present"])
        self.assertNotIn("candidates", audit)
        self.assertNotIn("selected_candidate", audit)

    def test_success_package_and_a1_short_circuit(self) -> None:
        harness = self.make_harness()
        completed, audit, output = harness.run()
        self.assertEqual(completed.returncode, 0, f"{completed.stderr}\naudit={audit}")
        self.assertEqual(audit["status"], gate.SUCCESS_STATUS)
        self.assertTrue(audit["valid_experiment"])
        self.assertEqual(audit["candidate_attempts"], ["A1"])
        self.assertEqual(audit["selected_candidate"], "A1")
        self.assertFalse(audit["formal_result"])
        self.assertFalse(audit["stage_progression_allowed"])
        self.assertFalse(audit["fresh_held_out_read"])
        self.assertEqual(set(audit["input_sha256"]), {str(item) for item in gate.PERMITTED_INPUT_IDS})
        self.assertEqual(set(audit["source_file_sha256"]), set(gate.REQUIRED_SOURCE_PATHS))
        self.assertTrue((output / "authorization_manifest.json").is_file())
        self.assertTrue((output / "config.json").is_file())
        self.assertTrue((output / "checksums.sha256").is_file())

    def test_scientific_failure_uses_a1_then_a2_and_no_third_candidate(self) -> None:
        zeros = {dataset_id: np.zeros((13, 30)) for dataset_id in gate.PERMITTED_INPUT_IDS}
        harness = self.make_harness(zeros)
        completed, audit, _ = harness.run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(audit["status"], gate.FAILURE_STATUS)
        self.assertTrue(audit["valid_experiment"])
        self.assertEqual(audit["candidate_attempts"], ["A1", "A2"])
        self.assertIsNone(audit["selected_candidate"])
        self.assertTrue(all(not item["development_evaluated"] for item in audit["candidates"]))

    def test_fake_cli_commit_is_rejected(self) -> None:
        harness = self.make_harness()
        completed, audit, _ = harness.run(source_commit="0" * 40)
        self.assertEqual(completed.returncode, 2)
        self.assert_invalid(audit, "SOURCE_COMMIT_DECLARATION_MISMATCH")

    def test_head_mismatch_is_rejected(self) -> None:
        harness = self.make_harness()
        harness.manifest["expected_source"]["head"] = "0" * 40
        harness.write_manifest()
        _, audit, _ = harness.run()
        self.assert_invalid(audit, "HEAD_MISMATCH")

    def test_tree_mismatch_is_rejected(self) -> None:
        harness = self.make_harness()
        harness.manifest["expected_source"]["tree"] = "0" * 40
        harness.write_manifest()
        _, audit, _ = harness.run()
        self.assert_invalid(audit, "TREE_MISMATCH")

    def test_dirty_worktree_is_rejected_before_input_hashing(self) -> None:
        harness = self.make_harness()
        (harness.repo / gate.PROTOCOL_PATH).write_text("dirty\n", encoding="utf-8")
        _, audit, _ = harness.run()
        self.assert_invalid(audit, "DIRTY_WORKTREE")

    def test_unreadable_git_is_rejected(self) -> None:
        harness = self.make_harness()
        shutil.move(harness.repo / ".git", harness.root / "disabled-git-metadata")
        _, audit, _ = harness.run()
        self.assert_invalid(audit, "GIT_STATE_UNREADABLE")

    def test_source_and_protocol_hash_mismatch_are_rejected(self) -> None:
        for relative_path in (gate.RUNNER_PATH, gate.PROTOCOL_PATH, gate.LIBRARY_PATH):
            with self.subTest(relative_path=relative_path):
                harness = self.make_harness()
                harness.manifest["source_files"][relative_path] = "0" * 64
                harness.write_manifest()
                _, audit, _ = harness.run()
                self.assert_invalid(audit, "SOURCE_FILE_HASH_MISMATCH")

    def test_config_tampering_cannot_be_authorized_by_new_hashes(self) -> None:
        harness = self.make_harness()
        path = harness.repo / gate.CONFIG_PATH
        config = json.loads(path.read_text(encoding="utf-8"))
        config["absolute_thresholds"]["max_residual"] = 0.26
        path.write_bytes(gate.canonical_json_bytes(config))
        harness.git("add", gate.CONFIG_PATH)
        harness.git("commit", "-qm", "synthetic config tamper")
        harness.refresh_manifest()
        _, audit, _ = harness.run()
        self.assert_invalid(audit, "CONFIG_NOT_FROZEN")

    def test_input_replacement_is_rejected(self) -> None:
        harness = self.make_harness()
        harness.feature_paths[41006].write_text('{"features": []}', encoding="utf-8")
        _, audit, _ = harness.run()
        self.assert_invalid(audit, "INPUT_HASH_MISMATCH")

    def test_forbidden_heldout_is_rejected_without_path_access(self) -> None:
        harness = self.make_harness()
        harness.manifest["inputs"][-1] = {
            "dataset_id": 41007,
            "path": str(harness.root / "must-not-be-read"),
            "sha256": "0" * 64,
        }
        harness.write_manifest()
        _, audit, _ = harness.run()
        self.assert_invalid(audit, "FORBIDDEN_HELD_OUT_INPUT")

    def test_disguised_heldout_path_is_rejected_without_access(self) -> None:
        harness = self.make_harness()
        harness.manifest["inputs"][0]["path"] = str(harness.root / "41007" / "features.json")
        harness.manifest["inputs"][0]["sha256"] = "0" * 64
        harness.write_manifest()
        _, audit, _ = harness.run()
        self.assert_invalid(audit, "FORBIDDEN_HELD_OUT_PATH")

    def test_extra_input_is_rejected(self) -> None:
        harness = self.make_harness()
        harness.manifest["inputs"].append(
            {"dataset_id": 41009, "path": str(harness.root / "extra"), "sha256": "0" * 64}
        )
        harness.write_manifest()
        _, audit, _ = harness.run()
        self.assert_invalid(audit, "INPUT_SET_MISMATCH")

    def test_manifest_threshold_or_candidate_injection_is_rejected(self) -> None:
        for key, value in (("thresholds", {"max_residual": 1.0}), ("candidates", ["A3"])):
            with self.subTest(key=key):
                harness = self.make_harness()
                harness.manifest[key] = value
                harness.write_manifest()
                _, audit, _ = harness.run()
                self.assert_invalid(audit, "MANIFEST_SCHEMA_MISMATCH")

    def test_legacy_overclaim_states_are_absent(self) -> None:
        config_text = (ROOT / gate.CONFIG_PATH).read_text(encoding="utf-8")
        protocol_text = (ROOT / gate.PROTOCOL_PATH).read_text(encoding="utf-8")
        runner_text = (ROOT / gate.RUNNER_PATH).read_text(encoding="utf-8")
        for legacy in ("GPU_READY", "STOP_NOT_GPU_READY"):
            self.assertNotIn(legacy, config_text)
            self.assertNotIn(legacy, protocol_text)
            self.assertNotIn(legacy, runner_text)


if __name__ == "__main__":
    unittest.main()
