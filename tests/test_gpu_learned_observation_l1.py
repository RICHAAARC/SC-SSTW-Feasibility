from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

from src.sc_sstw_feasibility.gpu_internal_challenger import EXPECTED_BRANCHES, EXPECTED_INTERNAL_SHAPE, EXPECTED_TIMESTEPS
from src.sc_sstw_feasibility.gpu_learned_observation_l1 import (
    EXPECTED_STRUCTURE,
    finalize_l1_success_package,
    inspect_saved_mp4_codec,
    validate_injection_records,
    write_l1_failure_package,
)
from src.sc_sstw_feasibility.learned_observation import L1_IDS, TRAIN_IDS, VALIDATION_IDS, validate_learned_observation_config


class GpuLearnedObservationL1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs" / "learned_observation_frontend.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_l1_uses_frozen_config_and_exact_public_ids(self) -> None:
        validate_learned_observation_config(self.config)
        self.assertEqual(L1_IDS, TRAIN_IDS + VALIDATION_IDS)
        self.assertEqual(L1_IDS, tuple(self.config["dataset"]["l1_permitted_ids"]))
        self.assertEqual(EXPECTED_STRUCTURE["block_count"], 30)

    def test_persisted_injection_record_validator_is_fail_closed(self) -> None:
        records = []
        for index, (label, timestep) in enumerate(zip(EXPECTED_BRANCHES, EXPECTED_TIMESTEPS, strict=True)):
            records.append({
                "call_index": index, "step_index": index // 2,
                "branch": "cond" if index % 2 == 0 else "uncond",
                "branch_label": label, "timestep_value": timestep,
                "module_path": "transformer.blocks[29].attn1", "output_shape": EXPECTED_INTERNAL_SHAPE,
                "output_dtype": "torch.bfloat16", "original_tensor_version_before": 0,
                "original_tensor_version_after": 0, "original_tensor_version_unchanged": True,
                "modified_has_distinct_storage": True, "gain_solver_converged": True,
                "raw_float_rms": 1.0, "output_rms": 1.0, "initial_gain": 0.03,
                "applied_gain": 0.03, "effective_relative_rms": 0.03,
                "effective_absolute_error": 0.0, "residual_q_reconstruction_mse": 0.0,
            })
        self.assertTrue(validate_injection_records(records, self.config))
        for mutation in ("call_index", "step_index", "branch", "branch_label", "module_path", "output_shape", "original_tensor_version_after", "effective_relative_rms", "residual_q_reconstruction_mse"):
            changed = copy.deepcopy(records)
            changed[0][mutation] = {
                "call_index": 1, "step_index": 1, "branch": "uncond",
                "branch_label": "step_0_uncond", "module_path": "scheduler",
                "output_shape": [1], "original_tensor_version_after": 1,
                "effective_relative_rms": 0.04, "residual_q_reconstruction_mse": 0.1,
            }[mutation]
            self.assertFalse(validate_injection_records(changed, self.config), mutation)

    def test_codec_probe_requires_exact_h264_yuv420p_geometry(self) -> None:
        good = json.dumps({"streams": [{"codec_name": "h264", "pix_fmt": "yuv420p", "width": 512, "height": 320}]})
        with mock.patch("src.sc_sstw_feasibility.gpu_learned_observation_l1.shutil.which", return_value="/usr/bin/ffprobe"), mock.patch(
            "src.sc_sstw_feasibility.gpu_learned_observation_l1.subprocess.run",
            return_value=mock.Mock(stdout=good),
        ):
            self.assertEqual(inspect_saved_mp4_codec(Path("video.mp4"))["codec_name"], "h264")
        bad = json.dumps({"streams": [{"codec_name": "hevc", "pix_fmt": "yuv420p", "width": 512, "height": 320}]})
        with mock.patch("src.sc_sstw_feasibility.gpu_learned_observation_l1.shutil.which", return_value="/usr/bin/ffprobe"), mock.patch(
            "src.sc_sstw_feasibility.gpu_learned_observation_l1.subprocess.run",
            return_value=mock.Mock(stdout=bad),
        ):
            with self.assertRaises(Exception):
                inspect_saved_mp4_codec(Path("video.mp4"))

    def test_formal_runner_cross_checks_real_files_and_excludes_l2(self) -> None:
        source = (ROOT / "src" / "sc_sstw_feasibility" / "gpu_learned_observation_l1.py").read_text(encoding="utf-8")
        self.assertIn('assert_stage_dataset_access("gpu_train_validation", L1_IDS)', source)
        self.assertIn('manifest["injection_records_sha256"] != injection_sha', source)
        self.assertIn('weights_payload["train_artifact_sha256_by_dataset_id"] != expected_train_digests', source)
        self.assertIn("frontend.observe_saved_mp4(mp4_path)", source)
        self.assertIn("training_blind_observations.json", source)
        self.assertIn("execution_integrity.json", source)
        self.assertIn("six_execution_integrity_records_pass", source)
        self.assertIn("calibration_evidence", source)
        self.assertIn('"predeclared_conditions": {key: True for key in checks}', source)
        self.assertIn('"gpu_l2_admission": False', source)
        self.assertIn('"l2_candidate": gate_pass', source)
        train_loop = source.index("for dataset_id in TRAIN_IDS:")
        freeze = source.index('artifacts / "weights_freeze.json"')
        validation_loop = source.index("for dataset_id in VALIDATION_IDS:")
        self.assertLess(train_loop, freeze)
        self.assertLess(freeze, validation_loop)
        self.assertIn("acquire_and_freeze_ambiguity", source)
        self.assertIn("calibrate_from_frozen_ambiguity", source)
        self.assertNotIn("HELD_OUT_IDS", source)
        self.assertNotIn("NULL_IDS", source)
        self.assertNotIn("source_index", source)
        self.assertNotIn("clean.mp4", source)
        self.assertNotIn("paired_difference", source)

    def test_failure_package_is_complete_and_non_claiming(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            write_l1_failure_package(self.config_path, output, "a" * 40, "forced failure", stdout_text="stdout before failure\n", stderr_text="forced failure\n")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(metrics["gate_pass"])
            self.assertFalse(metrics["gpu_l2_admission"])
            self.assertFalse(metrics["method_claim"])
            self.assertEqual((output / "stdout.log").read_text(encoding="utf-8"), "stdout before failure\n")
            self.assertEqual((output / "stderr.log").read_text(encoding="utf-8"), "forced failure\n")
            self.assertTrue((output / "checksums.sha256").is_file())
            self.assertTrue(output.with_suffix(".tar.gz").is_file())


    def test_actual_cli_subprocess_preserves_prefailure_stdout_and_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            script = f"""
import sys
from pathlib import Path
from unittest import mock
from experiments import run_gpu_learned_observation_l1 as cli

def forced_failure(*args, **kwargs):
    print("stdout-before-failure", flush=True)
    raise RuntimeError("forced-after-stdout")

sys.argv = ["run_gpu_learned_observation_l1.py", "--config", {str(self.config_path)!r}, "--output-dir", {str(output)!r}, "--expected-commit", {('a' * 40)!r}]
with mock.patch.object(cli, "run_gpu_learned_observation_l1", side_effect=forced_failure):
    raise SystemExit(cli.main())
"""
            completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(completed.stdout, "stdout-before-failure\n")
            self.assertEqual(completed.stderr, "RuntimeError: forced-after-stdout\n")
            self.assertEqual((output / "stdout.log").read_text(encoding="utf-8"), completed.stdout)
            self.assertEqual((output / "stderr.log").read_text(encoding="utf-8"), completed.stderr)
            checksums = (output / "checksums.sha256").read_text(encoding="utf-8")
            self.assertIn("stdout.log", checksums)
            self.assertIn("stderr.log", checksums)
            self.assertTrue(output.with_suffix(".tar.gz").is_file())

    def test_success_finalizer_preserves_exact_cli_streams_before_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            output.mkdir()
            (output / "metrics.json").write_text("{}\n", encoding="utf-8")
            finalize_l1_success_package(output, stdout_text="actual stdout\n", stderr_text="actual stderr\n")
            self.assertEqual((output / "stdout.log").read_text(encoding="utf-8"), "actual stdout\n")
            self.assertEqual((output / "stderr.log").read_text(encoding="utf-8"), "actual stderr\n")
            checksums = (output / "checksums.sha256").read_text(encoding="utf-8")
            self.assertIn("stdout.log", checksums)
            self.assertIn("stderr.log", checksums)

    def test_thin_notebook_calls_only_formal_repository_cli(self) -> None:
        notebook = json.loads((ROOT / "notebooks" / "sc_sstw_gpu_learned_observation_l1.ipynb").read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 5)
        joined = "\n".join("".join(cell["source"]) for cell in code_cells)
        self.assertIn("run_gpu_learned_observation_l1.py", joined)
        self.assertIn("learned_observation_frontend.json", joined)
        self.assertIn("gpu_learned_observation_l1", joined)
        self.assertIn("userdata.get", joined)
        for forbidden in ("register_forward_hook", "train_public_relation_frontend", "acquire_and_freeze_ambiguity", "calibrate_from_frozen_ambiguity", "TEMPORAL_POINTS"):
            self.assertNotIn(forbidden, joined)
        self.assertTrue(all(cell["execution_count"] is None and not cell["outputs"] for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
