from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.gpu_observation import (  # noqa: E402
    CLAIM_BOUNDARY,
    dry_run_records,
    public_aisb_points,
    readout_q_from_rgb_frames,
    relation_prompts,
    summarize_observations,
)


class GpuObservationProbeTests(unittest.TestCase):
    def test_fixed_probe_contract_is_small(self) -> None:
        self.assertEqual(len(public_aisb_points()), 6)
        prompts = relation_prompts()
        self.assertEqual(len(prompts), 10)
        self.assertEqual([prompt["kind"] for prompt in prompts].count("repeat_floor"), 2)
        self.assertEqual([prompt["kind"] for prompt in prompts].count("public_aisb"), 6)
        self.assertEqual([prompt["kind"] for prompt in prompts].count("state_window"), 2)

    def test_dry_run_records_summarize_to_nonformal_json(self) -> None:
        records = dry_run_records()
        summary = summarize_observations(records)
        self.assertEqual(summary["claim_support_status"], CLAIM_BOUNDARY)
        self.assertFalse(summary["formal_result"])
        self.assertFalse(summary["fixed_fpr"])
        self.assertFalse(summary["observer_or_detector"])
        self.assertEqual(summary["record_count"], 10)
        self.assertTrue(summary["readout_finite"])
        self.assertIn(summary["probe_decision"], {"observable_candidate", "no_observable_signal"})

    def test_readout_responds_to_patch_location(self) -> None:
        left_frame = [[(8, 8, 8) for _x in range(32)] for _y in range(24)]
        right_frame = [[(8, 8, 8) for _x in range(32)] for _y in range(24)]
        for y in range(10, 14):
            for x in range(4, 8):
                left_frame[y][x] = (240, 240, 240)
            for x in range(24, 28):
                right_frame[y][x] = (240, 240, 240)
        left_q = readout_q_from_rgb_frames([left_frame] * 4)
        right_q = readout_q_from_rgb_frames([right_frame] * 4)
        self.assertLess(left_q[0], right_q[0])

    def test_cli_dry_run_writes_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "probe"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "experiments" / "run_gpu_observation_probe.py"),
                    "--dry-run",
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn(CLAIM_BOUNDARY, result.stdout)
            payload = json.loads((output_dir / "gpu_observation_probe_result.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["claim_support_status"], CLAIM_BOUNDARY)
            self.assertEqual(payload["summary"]["record_count"], 10)

    def test_notebook_has_only_five_code_cells(self) -> None:
        notebook = json.loads((ROOT / "notebooks" / "sc_sstw_gpu_observation_probe.ipynb").read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 5)
        joined = "\n".join("".join(cell["source"]) for cell in code_cells)
        self.assertIn("drive.mount('/content/drive')", joined)
        self.assertIn("https://github.com/RICHAAARC/SC-SSTW-Feasibility.git", joined)
        self.assertIn("run_gpu_observation_probe.py", joined)
        self.assertIn("make_archive", joined)
        self.assertIn("/content/drive/MyDrive/SSTW/diagnostic_tests/sc_sstw_gpu_observation_probe", joined)
        self.assertNotIn("fixed_fpr", joined.lower())


if __name__ == "__main__":
    unittest.main()
