from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.sc_sstw_feasibility.gpu_carrier import (  # noqa: E402
    CarrierPropagationError,
    paired_difference_observation,
    replace_result_sample,
    saved_video_observation,
    validate_config,
    write_failure_package,
)


class GpuCarrierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = ROOT / "configs" / "gpu_carrier_propagation_v1.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_contract_is_exactly_13_tokens_and_frozen_public_aisb(self) -> None:
        validate_config(self.config)
        self.assertEqual(self.config["carrier"]["expected_tensor_shape"], [1, 16, 13, 40, 64])
        self.assertEqual(len(self.config["carrier"]["temporal_points"]), 13)
        self.assertEqual(self.config["carrier"]["temporal_points"][:6], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.25, 0.35], [0.75, 0.2], [0.2, 0.8]])
        self.assertEqual([item for group in self.config["diagnostic_readouts"]["temporal_frame_groups"] for item in group], list(range(49)))

    def test_contract_forbids_scheduler_and_in_place_mutation(self) -> None:
        carrier = self.config["carrier"]
        self.assertEqual(carrier["kind"], "direct_model_output_residual")
        self.assertFalse(carrier["scheduler_mutation_permitted"])
        self.assertFalse(carrier["in_place_mutation_permitted"])
        self.assertLessEqual(carrier["target_relative_rms"], 0.03)
        joined = (ROOT / "src" / "sc_sstw_feasibility" / "gpu_carrier.py").read_text(encoding="utf-8")
        self.assertNotIn("scheduler.step =", joined)
        self.assertNotIn("scheduler._", joined)
        self.assertEqual(self.config["paired_diagnostic_thresholds"], {"maximum_public_aisb_residual": 0.25, "maximum_public_condition_number": 10.0})
        self.assertIn("paired_difference_as_blind_observation", self.config["forbidden_claims"])


    def test_replace_result_sample_returns_new_object(self) -> None:
        class Result:
            def __init__(self, sample):
                self.sample = sample

        original = Result("clean")
        replaced = replace_result_sample(original, "modified")
        self.assertIsNot(original, replaced)
        self.assertEqual(original.sample, "clean")
        self.assertEqual(replaced.sample, "modified")

    def test_key_independent_saved_video_front_end_is_two_dimensional(self) -> None:
        pattern = [int(max(0, min(255, 100.0 + 40.0 * math.cos(2.0 * math.pi * (x + 0.5) / 32)))) for x in range(32)]
        frame = [[(pattern[x], pattern[x], pattern[x]) for x in range(32)] for _y in range(24)]
        frames = [frame for _index in range(49)]
        q = saved_video_observation(frames, self.config["diagnostic_readouts"]["temporal_frame_groups"])
        self.assertEqual(len(q), 13)
        self.assertTrue(all(len(point) == 2 for point in q))
        self.assertTrue(all(math.isfinite(value) for point in q for value in point))
        self.assertGreater(q[0][0], abs(q[0][1]))

    def test_paired_readout_is_separate_and_shape_checked(self) -> None:
        frame = [[(0, 0, 0) for _x in range(8)] for _y in range(8)]
        clean = [frame for _index in range(49)]
        watermarked = [frame for _index in range(49)]
        q = paired_difference_observation(clean, watermarked, self.config["diagnostic_readouts"]["temporal_frame_groups"])
        self.assertEqual([len(q), len(q[0])], [13, 2])
        with self.assertRaises(CarrierPropagationError):
            paired_difference_observation(clean, watermarked[:-1], self.config["diagnostic_readouts"]["temporal_frame_groups"])

    def test_thin_notebook_calls_only_repository_cli(self) -> None:
        notebook = json.loads((ROOT / "notebooks" / "sc_sstw_gpu_carrier_propagation_v1.ipynb").read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 5)
        joined = "\n".join("".join(cell["source"]) for cell in code_cells)
        self.assertIn("run_gpu_carrier_propagation.py", joined)
        self.assertIn("SC_SSTW_COMMIT", joined)
        self.assertIn("userdata.get", joined)
        self.assertIn("make_archive", joined)
        self.assertNotIn("construct_residual", joined)
        self.assertNotIn("saved_video_observation", joined)
        self.assertNotIn("scheduler.step", joined)
        self.assertNotIn("print(HF_TOKEN", joined)
        self.assertTrue(all(cell["execution_count"] is None and not cell["outputs"] for cell in code_cells))


    def test_failure_package_is_traceable_and_non_claiming(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "run"
            write_failure_package(self.config_path, output, "a" * 40, "forced failure")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
            self.assertFalse(metrics["gate_pass"])
            self.assertFalse(metrics["method_claim"])
            self.assertEqual(decision["auditor_decision"], "PENDING")
            self.assertTrue(output.with_suffix(".tar.gz").exists())
            self.assertTrue((output / "checksums.sha256").exists())


if __name__ == "__main__":
    unittest.main()
