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
        # Atomic no-replace directory publication is unavailable on WSL DrvFS;
        # the protocol intentionally fails closed there. Exercise the supported
        # Linux filesystem path for positive package tests.
        self.temporary = tempfile.TemporaryDirectory(dir="/tmp")
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
        completed, response = self.run_at(output, source_commit=source_commit)
        self.owner.assertTrue(response["package_written"], f"runner did not write a package: {response}")
        actual = Path(response["actual_package_path"])
        self.owner.assertEqual(actual, output)
        audit_path = actual / "audit.json"
        self.owner.assertTrue(
            audit_path.is_file(),
            f"runner did not create audit package\nstdout={completed.stdout}\nstderr={completed.stderr}",
        )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        return completed, audit, actual

    def run_at(
        self,
        output: Path,
        *,
        source_commit: str | None = None,
        fault_point: str | None = None,
        exception_type: str = "OSError",
        competitor_kind: str = "none",
        _expect_json: bool = True,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
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
        if fault_point is not None:
            wrapper = self.root / "fault-runner.py"
            wrapper.write_text(
                """from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

runner, manifest, output, point, exception_type, competitor_kind = sys.argv[1:]
spec = importlib.util.spec_from_file_location("g0_fault_runner", runner)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
fired = False

def inject(name, staging, target):
    global fired
    if fired or name != point:
        return
    fired = True
    if name.endswith("publish_race"):
        if competitor_kind == "file":
            target.write_bytes(b"competitor-file")
        else:
            target.mkdir()
            if competitor_kind == "nonempty_dir":
                (target / "sentinel.bin").write_bytes(b"competitor-directory")
        return
    errors = {"OSError": OSError, "RuntimeError": RuntimeError, "TypeError": TypeError}
    if exception_type == "KeyboardInterrupt":
        raise KeyboardInterrupt("injected interrupt")
    if exception_type == "SystemExit":
        raise SystemExit(71)
    raise errors[exception_type](f"injected {name}")

raise SystemExit(module.main(["--manifest", manifest, "--output", output], _test_fault=inject))
""",
                encoding="utf-8",
            )
            command = [
                sys.executable,
                "-B",
                str(wrapper),
                str(self.repo / gate.RUNNER_PATH),
                str(self.manifest_path),
                str(output),
                fault_point,
                exception_type,
                competitor_kind,
            ]
        completed = subprocess.run(command, cwd=self.repo, capture_output=True, text=True)
        if not _expect_json:
            return completed, {}
        lines = completed.stdout.splitlines()
        self.owner.assertEqual(len(lines), 1, f"runner stdout is not one canonical JSON line: {completed.stdout!r}")
        response = json.loads(lines[0])
        self.owner.assertEqual(lines[0], gate.canonical_json_bytes(response).decode("utf-8"))
        return completed, response

    def run_at_without_json(
        self,
        output: Path,
        *,
        fault_point: str,
        exception_type: str,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        return self.run_at(
            output,
            fault_point=fault_point,
            exception_type=exception_type,
            _expect_json=False,
        )


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

    def assert_minimal_invalid_package(
        self,
        completed: subprocess.CompletedProcess[str],
        response: dict[str, object],
        reason: str,
    ) -> Path:
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(response["status"], gate.INVALID_STATUS)
        self.assertEqual(response["reason_code"], reason)
        self.assertTrue(response["package_written"])
        package = Path(str(response["actual_package_path"]))
        audit = json.loads((package / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], gate.INVALID_STATUS)
        self.assertFalse(audit["formal_result"])
        self.assertFalse(audit["stage_progression_allowed"])
        self.assertFalse(audit["science_metrics_present"])
        self.assertEqual(audit["actual_package_path"], str(package))
        for forbidden in ("candidate", "candidate_order", "candidates", "selected_candidate", "frontend", "readout"):
            self.assertNotIn(forbidden, audit)
        self.assertFalse((package / "frozen_frontend.json").exists())
        self.assertFalse((package / "readout.json").exists())
        self.assertEqual(set(path.name for path in package.iterdir()), {"audit.json", "command.txt", "checksums.sha256"})
        return package

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
        self.assertTrue((output / "frozen_frontend.json").is_file())
        self.assertTrue((output / "readout.json").is_file())
        self.assertTrue((output / "checksums.sha256").is_file())
        declared = {line.split("  ", 1)[1] for line in (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(declared, {"audit.json", "authorization_manifest.json", "command.txt", "config.json", "frozen_frontend.json", "readout.json"})
        produced = np.asarray(json.loads((output / "readout.json").read_text(encoding="utf-8"))["coefficients"])
        raw = good_features()
        expected = gate.fit_readout([gate.transform(raw[dataset_id], "A1") for dataset_id in gate.FIT_IDS])
        self.assertTrue(np.array_equal(produced, expected))

    def test_scientific_failure_uses_a1_then_a2_and_no_third_candidate(self) -> None:
        zeros = {dataset_id: np.zeros((13, 30)) for dataset_id in gate.PERMITTED_INPUT_IDS}
        harness = self.make_harness(zeros)
        completed, audit, output = harness.run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(audit["status"], gate.FAILURE_STATUS)
        self.assertTrue(audit["valid_experiment"])
        self.assertEqual(audit["candidate_attempts"], ["A1", "A2"])
        self.assertIsNone(audit["selected_candidate"])
        self.assertTrue(all(not item["development_evaluated"] for item in audit["candidates"]))
        self.assertFalse((output / "frozen_frontend.json").exists())
        self.assertFalse((output / "readout.json").exists())

    def test_preexisting_targets_are_untouched_and_rejected_before_feature_access(self) -> None:
        for kind in ("empty_dir", "nonempty_dir", "file", "symlink"):
            with self.subTest(kind=kind):
                harness = self.make_harness()
                output = harness.root / f"preexisting-{kind}"
                symlink_target = harness.root / "symlink-target.bin"
                if kind == "empty_dir":
                    output.mkdir()
                elif kind == "nonempty_dir":
                    output.mkdir()
                    (output / "sentinel.bin").write_bytes(b"directory-sentinel")
                elif kind == "file":
                    output.write_bytes(b"file-sentinel")
                else:
                    symlink_target.write_bytes(b"symlink-target-sentinel")
                    output.symlink_to(symlink_target)
                inode_before = output.lstat().st_ino
                bytes_before = (
                    output.read_bytes()
                    if kind == "file"
                    else (symlink_target.read_bytes() if kind == "symlink" else None)
                )
                entries_before = (
                    {path.name: path.read_bytes() for path in output.iterdir()}
                    if kind in {"empty_dir", "nonempty_dir"}
                    else None
                )
                # Missing inputs are a read sentinel: output ownership must win.
                for path in harness.feature_paths.values():
                    path.unlink()
                completed, response = harness.run_at(output)
                package = self.assert_minimal_invalid_package(completed, response, "OUTPUT_TARGET_EXISTS")
                self.assertNotEqual(package, output)
                self.assertEqual(output.lstat().st_ino, inode_before)
                if kind == "file":
                    self.assertEqual(output.read_bytes(), bytes_before)
                elif kind == "symlink":
                    self.assertTrue(output.is_symlink())
                    self.assertEqual(symlink_target.read_bytes(), bytes_before)
                else:
                    self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, entries_before)

    def test_main_writer_fault_matrix_never_publishes_partial_success(self) -> None:
        points = (
            "main_staging_mkdir",
            "main_audit_write",
            "main_command_write",
            "main_manifest_write",
            "main_config_write",
            "main_frontend_write",
            "main_readout_write",
            "main_checksum_write",
            "main_checksum_self_check",
            "main_publish",
        )
        for point in points:
            with self.subTest(point=point):
                harness = self.make_harness()
                output = harness.root / f"fault-{point}"
                completed, response = harness.run_at(output, fault_point=point)
                package = self.assert_minimal_invalid_package(completed, response, "PACKAGE_WRITE_FAILED")
                self.assertNotEqual(package, output)
                self.assertFalse(output.exists())
                self.assertFalse(any(path.name.startswith(".g0-package-staging-") for path in output.parent.iterdir()))

    def test_main_publish_races_never_replace_competitor(self) -> None:
        for kind in ("file", "empty_dir", "nonempty_dir"):
            with self.subTest(kind=kind):
                harness = self.make_harness()
                output = harness.root / f"race-{kind}"
                completed, response = harness.run_at(output, fault_point="main_publish_race", competitor_kind=kind)
                self.assert_minimal_invalid_package(completed, response, "PACKAGE_WRITE_FAILED")
                if kind == "file":
                    self.assertEqual(output.read_bytes(), b"competitor-file")
                elif kind == "empty_dir":
                    self.assertTrue(output.is_dir())
                    self.assertEqual(list(output.iterdir()), [])
                else:
                    self.assertEqual((output / "sentinel.bin").read_bytes(), b"competitor-directory")

    def test_fallback_fault_matrix_is_nonrecursive_and_traceback_free(self) -> None:
        points = (
            "fallback_staging_mkdir",
            "fallback_audit_write",
            "fallback_command_write",
            "fallback_checksum_write",
            "fallback_publish",
        )
        for point in points:
            with self.subTest(point=point):
                harness = self.make_harness()
                output = harness.root / f"fallback-fault-{point}"
                output.write_bytes(b"owned-target")
                completed, response = harness.run_at(output, fault_point=point)
                self.assertEqual(completed.returncode, 2)
                self.assertNotIn("Traceback", completed.stderr)
                self.assertEqual(completed.stderr, "")
                self.assertEqual(response["status"], gate.INVALID_STATUS)
                self.assertEqual(response["reason_code"], "INVALID_PACKAGE_WRITE_FAILED")
                self.assertFalse(response["package_written"])
                self.assertIsNone(response["actual_package_path"])
                self.assertEqual(output.read_bytes(), b"owned-target")

    def test_fallback_publish_race_uses_next_exclusive_sibling(self) -> None:
        harness = self.make_harness()
        output = harness.root / "fallback-race"
        output.write_bytes(b"owned-target")
        completed, response = harness.run_at(output, fault_point="fallback_publish_race", competitor_kind="nonempty_dir")
        package = self.assert_minimal_invalid_package(completed, response, "OUTPUT_TARGET_EXISTS")
        first = output.with_name(f"{output.name}.invalid.0001")
        self.assertEqual((first / "sentinel.bin").read_bytes(), b"competitor-directory")
        self.assertEqual(package, output.with_name(f"{output.name}.invalid.0002"))
        self.assertEqual(output.read_bytes(), b"owned-target")

    def test_unexpected_exception_types_are_wrapped_once_without_science(self) -> None:
        for exception_type in ("OSError", "RuntimeError", "TypeError"):
            with self.subTest(exception_type=exception_type):
                harness = self.make_harness()
                output = harness.root / f"unexpected-{exception_type}"
                completed, response = harness.run_at(
                    output,
                    fault_point="before_evaluate",
                    exception_type=exception_type,
                )
                package = self.assert_minimal_invalid_package(completed, response, "UNEXPECTED_RUNTIME_FAILURE")
                audit = json.loads((package / "audit.json").read_text(encoding="utf-8"))
                self.assertEqual(audit["exception_type"], exception_type)

    def test_keyboard_interrupt_and_system_exit_are_not_swallowed(self) -> None:
        for exception_type in ("KeyboardInterrupt", "SystemExit"):
            with self.subTest(exception_type=exception_type):
                harness = self.make_harness()
                output = harness.root / f"interrupt-{exception_type}"
                completed, _ = harness.run_at_without_json(
                    output,
                    fault_point="before_preflight",
                    exception_type=exception_type,
                )
                self.assertNotEqual(completed.returncode, 2)
                self.assertFalse(output.exists())

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
        _, audit, output = harness.run()
        self.assert_invalid(audit, "INPUT_HASH_MISMATCH")
        self.assertFalse((output / "frozen_frontend.json").exists())
        self.assertFalse((output / "readout.json").exists())

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
