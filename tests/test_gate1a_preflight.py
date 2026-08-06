from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

from src.sc_sstw_feasibility.gate1a_preflight import run_preflight


ROOT = Path(__file__).resolve().parents[1]


class Gate1APreflightTests(unittest.TestCase):
    def test_module_cannot_import_method_chain(self) -> None:
        path = ROOT / "src" / "sc_sstw_feasibility" / "gate1a_preflight.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(imported & {"acquisition", "calibration", "sync", "scoring", "aisb"})

    def test_preflight_records_only_admissibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_preflight(ROOT / "configs" / "mechanism_feasibility_v2.json", Path(directory))
            self.assertEqual(result["implementer_decision"], "GATE_PASS")
            self.assertEqual(result["forbidden_method_calls_executed"], [])
            self.assertEqual(len(result["cases"]), 8)
            self.assertEqual(result["edit_case_count"], 8)
            self.assertEqual(result["null_case_count"], 8)
            self.assertNotIn("owner", json.dumps(result).lower())
            self.assertNotIn("held_out", json.dumps(result).lower())
            self.assertTrue(all(all(case["checks"].values()) for case in result["cases"]))


if __name__ == "__main__":
    unittest.main()
