from __future__ import annotations

import ast
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
    GPU_SOFTWARE,
    LearnedObservationL1Error,
    _build_internal_preflight_config,
    _run_internal_preflight_adapter,
    finalize_l1_success_package,
    run_gpu_learned_observation_l1,
    inspect_saved_mp4_codec,
    validate_injection_records,
    write_l1_failure_package,
)
from src.sc_sstw_feasibility.learned_observation import CONFIG_CANONICAL_SHA256, L1_IDS, TRAIN_IDS, VALIDATION_IDS, canonical_json_bytes, sha256_bytes, validate_learned_observation_config


class GpuLearnedObservationL1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs" / "learned_observation_frontend.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_l1_uses_frozen_config_and_exact_public_ids(self) -> None:
        validate_learned_observation_config(self.config)
        self.assertEqual(L1_IDS, TRAIN_IDS + VALIDATION_IDS)
        self.assertEqual(L1_IDS, tuple(self.config["dataset"]["l1_permitted_ids"]))
        self.assertEqual(EXPECTED_STRUCTURE["block_count"], 30)

    def test_preflight_adapter_adds_only_frozen_helper_fields_without_mutation(self) -> None:
        original = copy.deepcopy(self.config)
        original_digest = sha256_bytes(canonical_json_bytes(self.config))
        adapted = _build_internal_preflight_config(self.config)
        self.assertEqual(self.config, original)
        self.assertEqual(original_digest, CONFIG_CANONICAL_SHA256)
        self.assertEqual(sha256_bytes(canonical_json_bytes(self.config)), CONFIG_CANONICAL_SHA256)
        added = set(adapted["carrier"]) - set(self.config["carrier"])
        self.assertEqual(added, {"required_runtime_dtype", "gain_solver_max_iterations"})
        self.assertEqual(adapted["carrier"]["required_runtime_dtype"], "torch.bfloat16")
        self.assertEqual(adapted["carrier"]["gain_solver_max_iterations"], 12)
        for key in self.config["carrier"]:
            self.assertEqual(adapted["carrier"][key], self.config["carrier"][key])
        for forbidden_key, value in (("required_runtime_dtype", "torch.bfloat16"), ("gain_solver_max_iterations", 12)):
            changed = copy.deepcopy(self.config)
            changed["carrier"][forbidden_key] = value
            with self.assertRaises(Exception):
                validate_learned_observation_config(changed)

    def test_preflight_adapter_passes_literal_contract_and_actual_dtype_argument(self) -> None:
        fake_dtype = object()
        fake_torch = mock.Mock(bfloat16=fake_dtype)
        with mock.patch("src.sc_sstw_feasibility.gpu_learned_observation_l1.run_internal_preflight", return_value={"passed": True, "records": []}) as helper:
            result = _run_internal_preflight_adapter(fake_torch, self.config)
        self.assertTrue(result["passed"])
        received_torch, received_config, received_dtype = helper.call_args.args
        self.assertIs(received_torch, fake_torch)
        self.assertIs(received_dtype, fake_dtype)
        added = set(received_config["carrier"]) - set(self.config["carrier"])
        self.assertEqual(added, {"required_runtime_dtype", "gain_solver_max_iterations"})
        self.assertEqual(received_config["carrier"]["required_runtime_dtype"], "torch.bfloat16")
        self.assertEqual(received_config["carrier"]["gain_solver_max_iterations"], 12)

    def test_preflight_failure_or_exception_blocks_model_generation_and_training(self) -> None:
        expected_commit = "a" * 40
        fake_torch = mock.Mock()
        fake_torch.bfloat16 = object()
        fake_torch.__version__ = "2.6.0+cu124"
        fake_torch.version = mock.Mock(cuda="12.4")
        fake_torch.cuda.get_device_name.return_value = "NVIDIA A100-SXM4-40GB"
        model_load = mock.Mock()
        runtime = {
            "torch": fake_torch,
            "versions": GPU_SOFTWARE,
            "imageio_ffmpeg": mock.Mock(get_ffmpeg_version=mock.Mock(return_value="6.1")),
            "WanPipeline": mock.Mock(from_pretrained=model_load),
            "model_info": mock.Mock(return_value=mock.Mock(sha=self.config["model"]["revision"])),
        }
        def fake_git(_repo, *args):
            if args == ("rev-parse", "HEAD"):
                return expected_commit
            if args == ("status", "--porcelain"):
                return ""
            if args == ("remote", "get-url", "origin"):
                return "git@example.invalid/repo.git"
            raise AssertionError(args)
        outcomes = ({"passed": False, "records": [], "reason": "forced"}, RuntimeError("forced helper exception"))
        for outcome in outcomes:
            model_load.reset_mock()
            with tempfile.TemporaryDirectory() as tmpdir:
                output = Path(tmpdir) / "run"
                preflight_patch = mock.patch("src.sc_sstw_feasibility.gpu_learned_observation_l1.run_internal_preflight")
                with preflight_patch as preflight, mock.patch("src.sc_sstw_feasibility.gpu_learned_observation_l1._runtime_imports", return_value=runtime), mock.patch("src.sc_sstw_feasibility.gpu_learned_observation_l1._git", side_effect=fake_git), mock.patch("src.sc_sstw_feasibility.gpu_learned_observation_l1._binary_version", return_value="version"), mock.patch("src.sc_sstw_feasibility.gpu_learned_observation_l1.generate_l1_video") as generate, mock.patch("src.sc_sstw_feasibility.gpu_learned_observation_l1.train_public_relation_frontend") as train:
                    if isinstance(outcome, Exception):
                        preflight.side_effect = outcome
                    else:
                        preflight.return_value = outcome
                    with self.assertRaises((LearnedObservationL1Error, RuntimeError)):
                        run_gpu_learned_observation_l1(self.config_path, output, expected_commit)
                model_load.assert_not_called()
                generate.assert_not_called()
                train.assert_not_called()
        source = (ROOT / "src" / "sc_sstw_feasibility" / "gpu_learned_observation_l1.py").read_text(encoding="utf-8")
        self.assertLess(source.index("_run_internal_preflight_adapter(torch, config)"), source.index('runtime["WanPipeline"].from_pretrained'))

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
        runtime_cell = "".join(notebook["cells"][3]["source"])
        tree = ast.parse(runtime_cell)
        direct_imports = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertTrue({"torch", "numpy", "diffusers", "transformers", "accelerate"}.isdisjoint(direct_imports))
        self.assertIn("subprocess.run([sys.executable, '-c', probe_code]", runtime_cell)
        self.assertIn("fresh interpreter", runtime_cell)
        self.assertIn("_runtime_imports", runtime_cell)
        self.assertIn("cwd=REPO", runtime_cell)
        self.assertIn("probe_env['PYTHONPATH'] = str(REPO / 'src')", runtime_cell)

    def _execute_notebook_runtime_cell(self, probe_result=None, probe_exception=None, pip_exception=None, execute_cli_cell=False):
        notebook = json.loads((ROOT / "notebooks" / "sc_sstw_gpu_learned_observation_l1.ipynb").read_text(encoding="utf-8"))
        runtime_cell = "".join(notebook["cells"][3]["source"])
        cli_cell = "".join(notebook["cells"][4]["source"])
        calls = []
        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            if "-m" in args and "pip" in args:
                if pip_exception is not None:
                    raise pip_exception
                return mock.Mock(stdout="", stderr="", returncode=0)
            if "run_gpu_learned_observation_l1.py" in " ".join(map(str, args)):
                return mock.Mock(stdout="formal-cli", stderr="", returncode=0)
            if probe_exception is not None:
                raise probe_exception
            return mock.Mock(stdout=probe_result or "", stderr="", returncode=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            namespace = {
                "BOOTSTRAP_ERROR": None, "REPO": ROOT, "subprocess": mock.Mock(run=fake_run),
                "LOCAL_RUN": Path(tmpdir) / "run", "EXPECTED_COMMIT": "a" * 40,
            }
            exec(compile(runtime_cell, "notebook-runtime-cell", "exec"), namespace)
            if execute_cli_cell:
                exec(compile(cli_cell, "notebook-cli-cell", "exec"), namespace)
            return namespace, calls

    def test_runtime_probe_failures_execute_cell4_and_block_formal_cli(self) -> None:
        valid = {
            "versions": GPU_SOFTWARE, "torch": "2.6.0+cu124", "torch_major_2": True,
            "cuda_available": True, "bf16_supported": True, "gpu": "NVIDIA A100-SXM4-40GB",
            "ffprobe_path": "/usr/bin/ffprobe", "ffprobe_version": "ffprobe version 6.1",
        }
        marker = lambda payload: "SC_SSTW_RUNTIME_PROBE=" + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        failures = [
            {"probe_exception": subprocess.CalledProcessError(1, [sys.executable, "-c"])},
            {"probe_exception": subprocess.TimeoutExpired([sys.executable, "-c"], 300)},
            {"pip_exception": subprocess.TimeoutExpired([sys.executable, "-m", "pip"], 600)},
            {"probe_result": "malformed output"},
            {"probe_result": "SC_SSTW_RUNTIME_PROBE={not-json"},
            {"probe_result": marker(valid) + "\n" + marker(valid)},
        ]
        missing = copy.deepcopy(valid)
        missing.pop("gpu")
        extra = {**valid, "unexpected": True}
        failures.extend(({"probe_result": marker(missing)}, {"probe_result": marker(extra)}))
        for key, value in (
            ("versions", {**GPU_SOFTWARE, "numpy": "2.0.0"}),
            ("torch", ""), ("torch", 2.6), ("torch", "3.0.0"), ("torch", "1.13.1"),
            ("torch_major_2", False), ("cuda_available", False),
            ("bf16_supported", False), ("gpu", ""), ("ffprobe_path", None),
            ("ffprobe_version", ""),
        ):
            changed = copy.deepcopy(valid)
            changed[key] = value
            failures.append({"probe_result": marker(changed)})
        for kwargs in failures:
            namespace, calls = self._execute_notebook_runtime_cell(execute_cli_cell=True, **kwargs)
            self.assertIsNotNone(namespace["BOOTSTRAP_ERROR"])
            self.assertIsNone(namespace["RUNTIME_PROBE"])
            self.assertEqual(namespace["RETURN_CODE"], 2)
            formal_calls = [args for args, _call_kwargs in calls if "run_gpu_learned_observation_l1.py" in " ".join(map(str, args))]
            self.assertEqual(formal_calls, [])

    def test_runtime_probe_success_uses_exact_checkout_and_all_formal_imports(self) -> None:
        valid = {
            "versions": GPU_SOFTWARE, "torch": "2.6.0+cu124", "torch_major_2": True,
            "cuda_available": True, "bf16_supported": True, "gpu": "NVIDIA A100-SXM4-40GB",
            "ffprobe_path": "/usr/bin/ffprobe", "ffprobe_version": "ffprobe version 6.1",
        }
        marker = "SC_SSTW_RUNTIME_PROBE=" + json.dumps(valid, sort_keys=True, separators=(",", ":"))
        namespace, calls = self._execute_notebook_runtime_cell(marker)
        self.assertIsNone(namespace["BOOTSTRAP_ERROR"])
        self.assertEqual(namespace["RUNTIME_PROBE"], valid)
        checker_args, checker_kwargs = calls[1]
        self.assertEqual(checker_args[:2], [sys.executable, "-c"])
        self.assertEqual(checker_kwargs["cwd"], ROOT)
        self.assertEqual(checker_kwargs["env"]["PYTHONPATH"], str(ROOT / "src"))
        self.assertEqual(checker_kwargs["timeout"], 300)
        source = (ROOT / "src" / "sc_sstw_feasibility" / "gpu_learned_observation_l1.py").read_text(encoding="utf-8")
        runtime_imports = source[source.index("def _runtime_imports"):source.index("def run_gpu_learned_observation_l1")]
        self.assertIn("WanPipeline", runtime_imports)
        self.assertIn("model_info", runtime_imports)
        for dependency in GPU_SOFTWARE:
            self.assertIn(dependency, runtime_imports)


if __name__ == "__main__":
    unittest.main()
