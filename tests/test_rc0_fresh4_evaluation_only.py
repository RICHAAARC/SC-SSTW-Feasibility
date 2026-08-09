from __future__ import annotations

import inspect
from pathlib import Path

from sc_sstw_feasibility.rc0_fresh4_evaluation_only import (
    OUTPUT,
    _compare_clean_output,
    _expected_artifact_paths,
    run_evaluation_only_once,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_closed_artifact_universe_and_output() -> None:
    conditions, transforms = _expected_artifact_paths()
    assert len(conditions) == 16 and len(transforms) == 64
    assert OUTPUT == Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-fresh4-effectiveness-cpu-run1/evaluation-only-result")


def test_clean_comparison_uses_identity_only() -> None:
    common = {"observation": [[0.0, 0.0]], "presence": {"presence_pass": True}, "downstream_executed": True, "calibration": {"K2": 1}}
    identity = {"transform_id": "identity", "passed": True}
    actual = {**common, "phase": identity}
    original = {**common, "phase": {"identity": identity, "delete6_duplicate12": {"passed": False}}}
    assert _compare_clean_output(actual, original) is True
    original["phase"]["identity"] = {"transform_id": "identity", "passed": False}
    assert _compare_clean_output(actual, original) is False


def test_zero_parameter_read_only_source_surface() -> None:
    assert list(inspect.signature(run_evaluation_only_once).parameters) == ["repo_root", "argv", "cwd"]
    runner = (ROOT / "experiments/run_rc0_fresh4_evaluation_only.py").read_text(encoding="utf-8")
    module = (ROOT / "src/sc_sstw_feasibility/rc0_fresh4_evaluation_only.py").read_text(encoding="utf-8")
    assert "argparse" not in runner and "--" not in runner
    for forbidden in ("encode_mp4", "apply_carrier", "zipfile", "ZipFile", "TemporaryDirectory", "WanPipeline", "torch"):
        assert forbidden not in module
    assert module.index("_validate_all_identities(repo_root)") < module.index("decode_saved_mp4(condition_paths[key])")
    assert "ENGINEERING_EVALUATION_INVALID" in module
