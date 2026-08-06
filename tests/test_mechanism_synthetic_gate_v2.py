from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.mechanism_synthetic_gate_v2 import (
    ProtocolError,
    BurstCandidate,
    _semantic_truth_coverage,
    build_state_records,
    _channel,
    _validate_gate1a_admission,
    acquire_public_ambiguity,
)
from sc_sstw_feasibility.gate1a_preflight import run_preflight

CONFIG = ROOT / "configs" / "mechanism_feasibility_v2.json"
LOCK = ROOT / "protocols" / "gate1b_admission_v2.json"


class MechanismSyntheticGateV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        run_preflight(CONFIG, Path(self.directory.name))
        self.evidence = Path(self.directory.name) / "gate1a_input_admissibility.json"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_gate1a_admission_and_all_constructed_channels_match(self) -> None:
        admission = _validate_gate1a_admission(CONFIG, self.evidence, LOCK)
        self.assertEqual(admission["matrix_record_count"], 8)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        for index in range(8):
            matrix, bias, condition = _channel(config, index)
            self.assertEqual(len(matrix), 2)
            self.assertEqual(len(bias), 2)
            self.assertLessEqual(condition, 10.0)

    def test_tampered_gate1a_evidence_is_rejected_before_chain(self) -> None:
        payload = json.loads(self.evidence.read_text(encoding="utf-8"))
        payload["cases"][0]["A"][0][0] += 0.01
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "evidence SHA mismatch"):
                _validate_gate1a_admission(CONFIG, path, LOCK)

    def test_boundary_subwindow_with_correct_correspondence_is_covered(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        edited = build_state_records(config, config["candidate_keys"]["owner_key_id"])
        candidate = BurstCandidate(
            start_index=7, template_id="double_redundant_alpha_v1", residual=0.0,
            observed_length=11, missing_template_index=0,
        )
        covered, total, false_count = _semantic_truth_coverage([candidate], edited, config)
        self.assertEqual((covered, total, false_count), (1, 3, 0))

    def test_wrong_template_correspondence_is_not_covered(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        edited = build_state_records(config, config["candidate_keys"]["owner_key_id"])
        candidate = BurstCandidate(
            start_index=7, template_id="double_redundant_alpha_v1", residual=0.0,
            observed_length=11, missing_template_index=1,
        )
        covered, total, false_count = _semantic_truth_coverage([candidate], edited, config)
        self.assertEqual((covered, total, false_count), (0, 3, 1))

    def test_acquisition_remains_key_and_truth_independent(self) -> None:
        self.assertEqual(set(inspect.signature(acquire_public_ambiguity).parameters), {"observations", "config"})


if __name__ == "__main__":
    unittest.main()
