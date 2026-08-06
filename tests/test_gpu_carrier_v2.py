from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.sc_sstw_feasibility.gpu_carrier import paired_difference_observation, saved_video_observation  # noqa: E402
from src.sc_sstw_feasibility.gpu_carrier_v2 import (  # noqa: E402
    construct_quantization_aware_residual,
    solve_quantized_scalar_gain,
    validate_config_v2,
    write_failure_package_v2,
)


class GpuCarrierV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs" / "gpu_carrier_propagation_v2.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_v2_contract_freezes_scientific_inputs_and_thresholds(self) -> None:
        validate_config_v2(self.config)
        carrier = self.config["carrier"]
        self.assertEqual(self.config["protocol_id"], "sc_sstw_gpu_carrier_propagation_v2")
        self.assertEqual(carrier["expected_tensor_shape"], [1, 16, 13, 40, 64])
        self.assertEqual(carrier["target_relative_rms"], 0.03)
        self.assertEqual(carrier["target_relative_rms_absolute_tolerance"], 0.00005)
        self.assertEqual(carrier["required_runtime_dtype"], "torch.bfloat16")
        self.assertEqual(self.config["relation_thresholds"], {"maximum_public_aisb_residual": 0.25, "maximum_public_condition_number": 10.0})
        self.assertEqual([i for group in self.config["diagnostic_readouts"]["temporal_frame_groups"] for i in group], list(range(49)))

    def test_contract_rejects_every_frozen_scientific_input_mutation(self) -> None:
        mutations = []

        changed = copy.deepcopy(self.config)
        changed["carrier"]["temporal_points"][12][1] = -0.5
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["carrier"]["temporal_point_roles"][7] = "tampered"
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["diagnostic_readouts"]["temporal_frame_groups"][1] = [1, 2, 3]
        mutations.append(changed)
        for field in ("analytic_basis", "apply_to", "injection_location", "required_runtime_dtype"):
            changed = copy.deepcopy(self.config)
            if isinstance(changed["carrier"][field], list):
                changed["carrier"][field][0] = "tampered"
            else:
                changed["carrier"][field] = "tampered"
            mutations.append(changed)

        for changed in mutations:
            with self.assertRaises(Exception):
                validate_config_v2(changed)

    def test_v1_files_are_byte_identical_to_frozen_commit(self) -> None:
        expected = {
            "configs/gpu_carrier_propagation_v1.json": "a567f80248197c64f6de3d0138dd5fde1b9da7678c1aee4ad7d85955a88b67a2",
            "src/sc_sstw_feasibility/gpu_carrier.py": "4b096476c1c0bf850befb1ed4095113c20a4d5b1aa8dc62ba2e897c1a83ee0e6",
            "experiments/run_gpu_carrier_propagation.py": "6653bf7c21c63e4f4fc61de30a12856ad7f1ce132dc802b959db46347b458791",
            "notebooks/sc_sstw_gpu_carrier_propagation_v1.ipynb": "9203f362fa681b699ddc0ddf0cbc6f8b5036ade600aa9c81030931c9dba6965b",
            "tests/test_gpu_carrier.py": "4dbfd7ca2f7ecff3165450e03df505b55894584a351b5a301e47def00f65063b",
        }
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)

    def test_quantization_aware_solver_checks_actual_bfloat16_effective_delta(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable in CPU contract environment")
        for seed, scale in ((7, 0.1), (11, 1.0), (19, 10.0)):
            torch.manual_seed(seed)
            sample = (scale * torch.randn((1, 16, 13, 40, 64), dtype=torch.float32)).to(torch.bfloat16)
            modified, effective_delta, record = construct_quantization_aware_residual(
                torch, sample, self.config["carrier"]["temporal_points"], 0.03, 0.00005, 12
            )
            measured = float((effective_delta.float().square().mean().sqrt() / sample.float().square().mean().sqrt()).item())
            self.assertEqual(effective_delta.dtype, sample.dtype)
            self.assertTrue(torch.equal(effective_delta, modified - sample))
            self.assertLessEqual(abs(measured - 0.03), 0.00005)
            self.assertAlmostEqual(measured, float(record["effective_relative_rms"]), places=8)
            self.assertTrue(record["gain_solver_converged"])

    def test_scalar_solver_closes_on_post_quantization_measurement_without_torch(self) -> None:
        measurements: list[float] = []

        def coarse_effective_measure(gain: float) -> float:
            measured = round(gain, 5)
            measurements.append(measured)
            return measured

        result = solve_quantized_scalar_gain(0.02992, 0.03, 0.00005, 12, coarse_effective_measure)
        self.assertGreaterEqual(len(measurements), 2)
        self.assertLessEqual(abs(float(result["effective_relative_rms"]) - 0.03), 0.00005)
        self.assertTrue(result["gain_solver_converged"])
        self.assertNotEqual(float(result["applied_gain"]), 0.02992)

    def test_best_candidate_objects_are_returned_without_recomputation(self) -> None:
        construct_source = inspect.getsource(construct_quantization_aware_residual)
        module_source = (ROOT / "src/sc_sstw_feasibility/gpu_carrier_v2.py").read_text(encoding="utf-8")
        self.assertEqual(construct_source.count("candidate ="), 1)
        self.assertIn("best = (error, modified, effective_delta, relative_rms, gain, iteration)", construct_source)
        self.assertIn("return best_modified, best_effective_delta", construct_source)
        self.assertNotIn("solution =", construct_source)
        self.assertIn("modified, effective_delta, energy = construct_quantization_aware_residual(", module_source)
        self.assertNotIn("modified = sample + effective_delta", module_source)

    def test_paired_rgb_and_blind_single_video_are_separate_calls(self) -> None:
        groups = self.config["diagnostic_readouts"]["temporal_frame_groups"]

        clean_frame = [[(20, 20, 20) for _x in range(16)] for _y in range(12)]
        marked_frame = [[(20 + int(4 * math.cos(2 * math.pi * (x + 0.5) / 16)), 20, 20) for x in range(16)] for _y in range(12)]
        clean, marked = [clean_frame] * 49, [marked_frame] * 49
        blind = saved_video_observation(marked, groups)
        paired = paired_difference_observation(clean, marked, groups)
        self.assertEqual([len(blind), len(blind[0])], [13, 2])
        self.assertEqual([len(paired), len(paired[0])], [13, 2])
        self.assertNotEqual(blind, paired)
        source = (ROOT / "src/sc_sstw_feasibility/gpu_carrier_v2.py").read_text(encoding="utf-8")
        self.assertIn("run_effective_delta_preflight", source)
        self.assertIn('"effective_delta_preflight_16_of_16"', source)
        self.assertLess(source.index("run_effective_delta_preflight(torch, config, dtype)"), source.index("WanPipeline.from_pretrained"))
        self.assertIn("effective_delta = modified - sample", source)
        self.assertIn('"single_saved_watermarked_mp4_only"', source)
        self.assertIn('"eligible_for_blind_pass": False', source)

    def test_pass_conditions_match_implementation_and_paired_cannot_substitute(self) -> None:
        source = (ROOT / "src/sc_sstw_feasibility/gpu_carrier_v2.py").read_text(encoding="utf-8")
        self.assertIn('set(checks) != set(config["pass_conditions"])', source)
        self.assertIn("execution_integrity_pass and blind_saved_mp4_relation_pass", source)
        self.assertNotIn("paired_propagation_diagnostic_only and", source)
        self.assertNotIn("source_index", source)
        self.assertNotIn("message", source)
        self.assertNotIn("scheduler.step =", source)

    def test_failure_package_is_traceable_and_non_claiming(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            write_failure_package_v2(self.config_path, output, "b" * 40, "forced failure")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(metrics["gate_pass"])
            self.assertFalse(metrics["public_relation_propagation_bridge_claim"])
            self.assertFalse(metrics["method_claim"])
            self.assertTrue((output / "checksums.sha256").exists())
            self.assertTrue(output.with_suffix(".tar.gz").exists())

    def test_thin_v2_notebook_calls_only_repository_cli(self) -> None:
        notebook = json.loads((ROOT / "notebooks/sc_sstw_gpu_carrier_propagation_v2.ipynb").read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 5)
        joined = "\n".join("".join(cell["source"]) for cell in code_cells)
        self.assertIn("run_gpu_carrier_propagation_v2.py", joined)
        self.assertIn("gpu_carrier_propagation_v2.json", joined)
        self.assertIn("gpu_carrier_propagation_v2", joined)
        self.assertIn("userdata.get", joined)
        self.assertNotIn("construct_quantization_aware_residual", joined)
        self.assertNotIn("saved_video_observation", joined)
        self.assertNotIn("scheduler.step", joined)
        self.assertTrue(all(cell["execution_count"] is None and not cell["outputs"] for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
