from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.mechanism_synthetic_gate import (
    ProtocolError,
    acquire_public_ambiguity,
    apply_declared_edit,
    build_state_records,
    load_and_validate_config,
)


CONFIG_PATH = ROOT / "configs" / "mechanism_feasibility_v1.json"


class MechanismSyntheticGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_and_validate_config(CONFIG_PATH)

    def test_acquisition_api_has_no_candidate_key_or_truth_input(self) -> None:
        parameters = set(inspect.signature(acquire_public_ambiguity).parameters)
        self.assertEqual(parameters, {"observations", "config"})

    def test_missing_required_protocol_field_is_rejected(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        del payload["synthetic_protocol"]["channel"]["matrix_shape"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises((KeyError, ProtocolError)):
                load_and_validate_config(path)

    def test_duplication_reuses_exact_observation_without_new_noise(self) -> None:
        case = self.config["synthetic_protocol"]["case_protocols"][6]
        records = build_state_records(self.config, self.config["candidate_keys"]["owner_key_id"])
        observed = [{**record, "q": [float(index), -float(index)]} for index, record in enumerate(records)]
        edited = apply_declared_edit(observed, case)
        duplicates = [record for record in edited if record.get("duplicated_observation")]
        self.assertEqual(len(duplicates), 1)
        duplicate = duplicates[0]
        original = next(
            record
            for record in observed
            if record["segment_id"] == case["segment_id"] and record["private_offset"] == case["private_offset"]
        )
        self.assertEqual(duplicate["q"], original["q"])

    def test_owner_and_wrong_keys_share_public_schedule(self) -> None:
        keys = self.config["candidate_keys"]
        owner = build_state_records(self.config, keys["owner_key_id"])
        wrong = build_state_records(self.config, keys["wrong_key_ids"][0])
        for owner_record, wrong_record in zip(owner, wrong, strict=True):
            if owner_record["kind"] == "public":
                self.assertEqual(owner_record["state"], wrong_record["state"])
            else:
                self.assertEqual(owner_record["kind"], wrong_record["kind"])


if __name__ == "__main__":
    unittest.main()
