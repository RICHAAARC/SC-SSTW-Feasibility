import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).parents[1] / "experiments" / "run_learned_observation_l1_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("l1_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiagnosticTests(unittest.TestCase):
    def test_candidate_metrics_collects_all_eight_windows(self):
        observation = np.asarray(MODULE.TEMPORAL_POINTS, dtype=np.float64)
        candidates = MODULE.candidate_metrics(observation)
        self.assertEqual([item["start_index"] for item in candidates], list(range(8)))
        self.assertTrue(candidates[0]["accepted"])
        self.assertIsNone(candidates[0]["rejection_reason"])

    def test_drift_uses_train_only_reference(self):
        train = np.arange(52 * 30, dtype=np.float64).reshape(52, 30)
        validation = {41005: train[:13] + 2000, 41006: train[:13] - 2000}
        result = MODULE.drift(train, validation, train.mean(axis=0), train.std(axis=0))
        self.assertEqual(result["normalization_source_dataset_ids"], [41001, 41002, 41003, 41004])
        self.assertEqual(set(result["distances"]), {"41005", "41006"})


if __name__ == "__main__":
    unittest.main()
