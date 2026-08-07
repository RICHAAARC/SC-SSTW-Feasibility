from __future__ import annotations

import copy
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.sc_sstw_feasibility.gpu_internal_challenger import (  # noqa: E402
    EXPECTED_BRANCHES,
    EXPECTED_INTERNAL_SHAPE,
    EXPECTED_PATCH_SHAPE,
    EXPECTED_TIMESTEPS,
    V2_FILE_DIGESTS,
    build_internal_basis,
    construct_internal_residual,
    project_internal_q,
    observed_cfg_sequence_matches,
    validate_internal_config,
    write_internal_failure_package,
)


class GpuInternalChallengerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs" / "gpu_internal_output_challenger_v1.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_contract_freezes_unique_internal_candidate_and_history(self) -> None:
        validate_internal_config(self.config)
        carrier = self.config["carrier"]
        self.assertEqual(carrier["module_path"], "transformer.blocks[29].attn1")
        self.assertEqual(carrier["expected_patch_embedding_shape"], EXPECTED_PATCH_SHAPE)
        self.assertEqual(carrier["expected_hook_output_shape"], EXPECTED_INTERNAL_SHAPE)
        self.assertEqual(carrier["expected_token_grid"], [13, 20, 32])
        self.assertEqual(carrier["token_order"], "index=((t*20)+h)*32+w_width_fastest_c_order")
        self.assertEqual(carrier["expected_branch_sequence"], EXPECTED_BRANCHES)
        self.assertEqual(carrier["expected_timestep_values"], EXPECTED_TIMESTEPS)
        self.assertEqual(carrier["target_relative_rms"], 0.03)
        self.assertEqual(carrier["target_relative_rms_absolute_tolerance"], 0.00005)
        self.assertEqual(self.config["history_lock"]["direct_v2_conclusion"], "NO_GO")
        self.assertEqual(self.config["history_lock"]["overall_method_conclusion"], "NOT_DETERMINED")

    def test_all_scientific_mutations_fail_closed(self) -> None:
        mutations: list[dict] = []
        paths = [
            ("module_path", "transformer.blocks[28].attn1"),
            ("block_index", 28),
            ("token_order", "unknown"),
            ("basis_formula_x", "learned"),
            ("target_relative_rms", 0.031),
            ("processor_or_qkv_mutation_permitted", True),
        ]
        for key, value in paths:
            changed = copy.deepcopy(self.config)
            changed["carrier"][key] = value
            mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["history_lock"]["direct_v2_conclusion"] = "PASS"
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["generation"]["seed"] += 1
        mutations.append(changed)
        changed = copy.deepcopy(self.config)
        changed["relation_thresholds"]["maximum_public_aisb_residual"] = 0.3
        mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(Exception):
                validate_internal_config(changed)

    def test_preserved_direct_v2_files_are_byte_identical(self) -> None:
        for relative, digest in V2_FILE_DIGESTS.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)

    def test_internal_basis_formula_and_width_fastest_token_order(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable in CPU contract environment")
        output = torch.zeros(EXPECTED_INTERNAL_SHAPE, dtype=torch.bfloat16)
        bx, by = build_internal_basis(torch, output)
        self.assertEqual(list(bx.shape), EXPECTED_INTERNAL_SHAPE)
        self.assertEqual(list(by.shape), EXPECTED_INTERNAL_SHAPE)
        grid_x = bx.float().reshape(1, 13, 20, 32, 1536)
        grid_y = by.float().reshape(1, 13, 20, 32, 1536)
        self.assertTrue(torch.equal(grid_x[0, 0, 0, :, 0], grid_x[0, 12, 19, :, 1535]))
        self.assertTrue(torch.equal(grid_y[0, 0, :, 0, 0], grid_y[0, 12, :, 31, 1535]))
        self.assertNotEqual(float(grid_x[0, 0, 0, 0, 0]), float(grid_x[0, 0, 0, 1, 0]))
        self.assertNotEqual(float(grid_y[0, 0, 0, 0, 0]), float(grid_y[0, 0, 1, 0, 0]))

    def test_quantized_internal_delta_is_measured_after_bfloat16_add(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch unavailable in CPU contract environment")
        torch.manual_seed(17)
        output = torch.randn(EXPECTED_INTERNAL_SHAPE, dtype=torch.float32).to(torch.bfloat16)
        modified, delta, record = construct_internal_residual(
            torch, output, self.config["carrier"]["temporal_points"], 0.03, 0.00005, 12,
        )
        measured = float((delta.float().square().mean().sqrt() / output.float().square().mean().sqrt()).item())
        self.assertTrue(torch.equal(delta, modified - output))
        self.assertLessEqual(abs(measured - 0.03), 0.00005)
        self.assertAlmostEqual(measured, float(record["effective_relative_rms"]), places=8)
        reconstructed = [[value / float(record["applied_gain"]) for value in point] for point in project_internal_q(torch, delta)]
        mse = sum((reconstructed[i][d] - self.config["carrier"]["temporal_points"][i][d]) ** 2 for i in range(13) for d in range(2)) / 26.0
        self.assertLessEqual(mse, 0.0001)

    def test_observed_cfg_sequence_rejects_repeated_reversed_and_missing_branches(self) -> None:
        correct = [{"branch_label": label, "timestep_value": timestep} for label, timestep in zip(EXPECTED_BRANCHES, EXPECTED_TIMESTEPS, strict=True)]
        self.assertTrue(observed_cfg_sequence_matches(correct, correct, correct))
        repeated = copy.deepcopy(correct)
        repeated[1]["branch_label"] = "step_0_cond"
        self.assertFalse(observed_cfg_sequence_matches(repeated, correct, correct))
        reversed_pair = copy.deepcopy(correct)
        reversed_pair[0]["branch_label"], reversed_pair[1]["branch_label"] = reversed_pair[1]["branch_label"], reversed_pair[0]["branch_label"]
        self.assertFalse(observed_cfg_sequence_matches(reversed_pair, correct, correct))
        self.assertFalse(observed_cfg_sequence_matches(correct[:-1], correct, correct))

    def test_hooks_prove_value_mapping_and_do_not_use_forbidden_carriers(self) -> None:
        source = (ROOT / "src/sc_sstw_feasibility/gpu_internal_challenger.py").read_text(encoding="utf-8")
        self.assertIn("patch_output.flatten(2).transpose(1, 2)", source)
        self.assertIn("torch.equal(expected, actual)", source)
        self.assertIn("transformer.blocks[29].attn1.register_forward_hook", source)
        self.assertIn("transformer.cache_context = observed_cache_context", source)
        self.assertIn("actual_branch not in {\"cond\", \"uncond\"}", source)
        self.assertIn("processor_object_id", source)
        self.assertIn("target_attention_processor_and_qkvout_parameters_unchanged", source)
        self.assertIn("attention_handle.remove()", source)
        self.assertIn("modified_has_distinct_storage", source)
        self.assertNotIn("set_processor", source)
        self.assertNotIn("to_q(", source)
        self.assertNotIn("to_k(", source)
        self.assertNotIn("to_v(", source)
        self.assertNotIn("scheduler.step =", source)
        self.assertNotIn("replace_result_sample", source)
        self.assertNotIn("\"cond\" if call_index % 2", source)
        self.assertNotIn("\"uncond\" if call_index % 2", source)

    def test_only_blind_single_mp4_relation_can_complete_gate(self) -> None:
        source = inspect.getsource(__import__("src.sc_sstw_feasibility.gpu_internal_challenger", fromlist=["run_internal_challenger"]).run_internal_challenger)
        self.assertIn('"single_saved_watermarked_mp4_only"', source)
        self.assertIn('"eligible_for_blind_pass": False', source)
        self.assertIn("execution_integrity_pass and blind_relation_pass", source)
        self.assertNotIn("paired_finite and blind_relation_pass", source)
        self.assertNotIn("source_index", source)
        self.assertNotIn("ground_truth", source)

    def test_failure_package_is_traceable_and_non_claiming(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            write_internal_failure_package(self.config_path, output, "c" * 40, "forced failure")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(metrics["gate_pass"])
            self.assertFalse(metrics["public_relation_propagation_bridge_claim"])
            self.assertFalse(metrics["method_claim"])
            self.assertTrue((output / "checksums.sha256").exists())
            self.assertTrue(output.with_suffix(".tar.gz").exists())

    def test_thin_notebook_calls_only_repository_cli(self) -> None:
        notebook_path = ROOT / "notebooks" / "sc_sstw_gpu_internal_output_challenger_v1.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 5)
        joined = "\n".join("".join(cell["source"]) for cell in code_cells)
        self.assertIn("run_gpu_internal_output_challenger_v1.py", joined)
        self.assertIn("gpu_internal_output_challenger_v1.json", joined)
        self.assertIn("gpu_internal_output_challenger_v1", joined)
        self.assertIn("userdata.get", joined)
        self.assertNotIn("construct_internal_residual", joined)
        self.assertNotIn("saved_video_observation", joined)
        self.assertNotIn("register_forward_hook", joined)
        self.assertNotIn("scheduler.step", joined)
        self.assertTrue(all(cell["execution_count"] is None and not cell["outputs"] for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
