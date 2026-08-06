from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from src.sc_sstw_feasibility.gpu_shape_preflight import _result_sample, _shape


ROOT = Path(__file__).resolve().parents[1]


class GpuShapePreflightTests(unittest.TestCase):
    def test_config_is_diagnostic_only_and_exactly_frozen(self) -> None:
        config = json.loads((ROOT / "configs" / "gpu_shape_preflight_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(config["generation"]["frame_count"], 49)
        self.assertEqual(config["generation"]["inference_steps"], 8)
        self.assertEqual(config["model"]["revision"], "0fad780a534b6463e45facd96134c9f345acfa5b")
        self.assertFalse(config["hook"]["mutation_permitted"])
        self.assertIn("method_pass", config["forbidden_claims"])

    def test_runner_has_no_injection_or_mock_path(self) -> None:
        path = ROOT / "src" / "sc_sstw_feasibility" / "gpu_shape_preflight.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn("make_mock_frames", names)
        self.assertNotIn("dry_run_records", names)
        text = path.read_text(encoding="utf-8")
        self.assertIn("pipe.transformer.forward = traced_forward", text)
        self.assertIn("pipe.transformer.forward = original_forward", text)
        self.assertNotIn("scheduler.step =", text)

    def test_shape_and_result_sample_helpers(self) -> None:
        class Tensor:
            shape = (1, 16, 13, 20, 32)

        class Result:
            sample = Tensor()

        self.assertEqual(_shape(Tensor()), [1, 16, 13, 20, 32])
        self.assertIsInstance(_result_sample(Result()), Tensor)

    def test_cli_failure_writes_minimum_traceable_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed_run"
            result = subprocess.run(
                [sys.executable, str(ROOT / "experiments" / "run_gpu_shape_preflight.py"),
                 "--config", str(ROOT / "configs" / "gpu_shape_preflight_v1.json"),
                 "--output-dir", str(output), "--expected-commit", "0" * 40],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 2)
            required = {"environment.json", "git_state.json", "config.json", "command.txt",
                        "stdout.log", "stderr.log", "metrics.json", "gate_decision.json", "README.md"}
            self.assertTrue(required.issubset({item.name for item in output.iterdir()}))
            self.assertTrue((output / "artifacts").is_dir())
            self.assertTrue((output / "checksums.sha256").is_file())
            self.assertTrue(output.with_suffix(".tar.gz").is_file())
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(metrics["gate_pass"])
            self.assertFalse(metrics["method_claim"])

    def test_thin_notebook_has_no_method_implementation(self) -> None:
        notebook = json.loads((ROOT / "notebooks" / "sc_sstw_gpu_shape_preflight_v1.ipynb").read_text(encoding="utf-8"))
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code), 5)
        joined = "\n".join("".join(cell["source"]) for cell in code)
        self.assertIn("SC_SSTW_COMMIT", joined)
        self.assertIn("run_gpu_shape_preflight.py", joined)
        self.assertIn("checksums.sha256", joined)
        self.assertNotIn("transformer.forward", joined)
        self.assertNotIn("scheduler.step", joined)


if __name__ == "__main__":
    unittest.main()
