from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from sc_sstw_feasibility.rc0_fresh12_method_stability_cpu import (
    BASE_MEMBERS,
    GROUP_ORDER,
    SOURCE_COMMIT,
    SOURCE_TREE,
    SOURCE_ZIP_SHA256,
    evaluate_clean_identity,
    load_frozen_fresh12,
)
from sc_sstw_feasibility.rc0_fresh4_effectiveness_cpu import TRANSFORM_ORDER, apply_frozen_frame_transform


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_exact8_input_and_exact160_semantics() -> None:
    config, phase_c = load_frozen_fresh12(ROOT)
    assert tuple(config["attempt_order"]) == GROUP_ORDER
    assert len(BASE_MEMBERS) == 8
    assert SOURCE_COMMIT == "ba226dc6e3d11f501c9fded970d2b8c68d3c6546"
    assert SOURCE_TREE == "b2829e9c3d50917429d43071c2827b5f9d65fe74"
    assert SOURCE_ZIP_SHA256 == "d6aebdfc115f6c6835fb1569a4356939dbb0d15a619bed75b6b02037c500dc92"
    assert tuple(item["id"] for item in phase_c["transforms"]) == TRANSFORM_ORDER
    frozen = config["phase_c_frozen_not_executed"]
    assert (frozen["derived_clean_count"], frozen["derived_transformed_count"], frozen["derived_total_count"]) == (32, 128, 160)
    assert frozen["corrected_semantics"] == "clean_identity_only_transformed_each_own_truth"


def test_clean_evaluation_is_identity_only() -> None:
    source = inspect.getsource(evaluate_clean_identity)
    assert 'evaluate_transformed_target(decoded_frames, "identity")' in source
    assert all(name not in source for name in ("delete6_duplicate12", "local_phase_plus1", "local_phase_minus1", "PERTURBATION_ORDER"))


def test_transform_indices_are_directly_reused_from_fresh4() -> None:
    _config, phase_c = load_frozen_fresh12(ROOT)
    frames = np.zeros((49, 320, 512, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(49, dtype=np.uint8)
    for record in phase_c["transforms"]:
        transformed = apply_frozen_frame_transform(frames, phase_c, record["id"])
        assert transformed[:, 0, 0, 0].tolist() == record["frame_source_indices"]


def test_runner_is_cpu_only_and_old_science_is_unchanged() -> None:
    runner = (ROOT / "experiments/run_rc0_fresh12_method_stability_cpu.py").read_text(encoding="utf-8")
    module = (ROOT / "src/sc_sstw_feasibility/rc0_fresh12_method_stability_cpu.py").read_text(encoding="utf-8")
    assert all(word not in runner + module for word in ("WanPipeline", "torch.cuda", "snapshot_download"))
    assert "retry_index\": 0" in module
    baseline = "8fd6313b11dc19cd2b5f2a81cb0abe4d606b33fc"
    guarded = [
        "src/sc_sstw_feasibility/rc0_pixel_chroma_fast_cpu.py",
        "src/sc_sstw_feasibility/rc0_target_only_blind_fast_cpu.py",
        "src/sc_sstw_feasibility/rc0_fresh4_effectiveness_cpu.py",
        "src/sc_sstw_feasibility/rc0_candidate_self_calibration_fast_cpu.py",
        "src/sc_sstw_feasibility/rc0_phase_recovery_fast_cpu.py",
    ]
    import subprocess
    for relative in guarded:
        current = subprocess.run(["git", "hash-object", relative], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        frozen = subprocess.run(["git", "rev-parse", f"{baseline}:{relative}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        assert current == frozen
