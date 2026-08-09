from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from src.sc_sstw_feasibility.rc0_aisb_capture_fast_cpu import (
    BOUND_ARTIFACT_SHA256,
    CANDIDATE_BUDGET_K,
    CONDITION_ORDER,
    GROUP_ORDER,
    PERTURBATION_ORDER,
    START_INDICES,
    TOP_K_PER_START,
    capture_cell,
    load_and_validate_config,
    perturb_observation,
    run_step3_once,
)
from src.sc_sstw_feasibility.rc0_causal_localization_v2 import schedule_a, schedule_b


ROOT = Path(__file__).resolve().parents[1]


def _affine_observation(schedule: tuple[tuple[float, float], ...]) -> np.ndarray:
    points = np.asarray(schedule, dtype=np.float64)
    transform = np.arange(60, dtype=np.float64).reshape(2, 30) / 37.0 + 0.2
    bias = np.linspace(-0.5, 0.5, 30, dtype=np.float64)
    return points @ transform + bias


def test_config_freezes_existing_formula_budget_and_perturbations() -> None:
    config = load_and_validate_config(ROOT / "configs/rc0_aisb_capture_fast_cpu.json")
    assert config["capture"]["invariant"] == "affine_burst_residual_v1"
    assert TOP_K_PER_START == 1
    assert CANDIDATE_BUDGET_K == 8
    assert config["capture"]["candidate_universe_size"] == 16
    assert config["capture"]["pruning"] == "at_each_start_retain_only_the_lower_residual_of_A_and_B"
    assert START_INDICES == tuple(range(8))
    assert PERTURBATION_ORDER == ("identity", "private_tail_delete6_duplicate12")
    assert config["capture"]["acceptance_threshold"] is None


def test_synthetic_affine_capture_and_fixed_tail_edit() -> None:
    for condition, schedule in (("A", schedule_a()), ("B", schedule_b())):
        observation = _affine_observation(schedule)
        edited = perturb_observation(observation, "private_tail_delete6_duplicate12")
        assert edited.shape == (13, 30)
        assert np.array_equal(edited[:6], observation[:6])
        assert np.array_equal(edited[-1], observation[12])
        for perturbation in PERTURBATION_ORDER:
            result = capture_cell(observation, condition, perturbation)
            assert result["target_captured"] is True
            assert result["target"]["start_index"] == 0
            assert result["target"]["template_id"] == condition
            assert result["start0_competition"]["own_beats_cross"] is True
            assert result["start0_competition"]["own_residual"] < result["start0_competition"]["cross_residual"]
            assert len(result["candidate_set"]) == CANDIDATE_BUDGET_K


def test_tie_rule_is_stable_A_before_B() -> None:
    tied = np.zeros((13, 30), dtype=np.float64)
    a = capture_cell(tied, "A", "identity")
    b = capture_cell(tied, "B", "identity")
    assert a["target"] is not None
    assert a["target_captured"] is False
    assert b["target_captured"] is False
    assert a["start0_competition"]["own_beats_cross"] is False
    assert all(candidate["template_id"] == "A" for candidate in a["candidate_set"])


def test_all_eight_input_hashes_are_fixed() -> None:
    assert tuple(BOUND_ARTIFACT_SHA256) == GROUP_ORDER
    assert sum(len(group) for group in BOUND_ARTIFACT_SHA256.values()) == 8
    for group in GROUP_ORDER:
        assert tuple(BOUND_ARTIFACT_SHA256[group]) == CONDITION_ORDER
        assert all(len(digest) == 64 for digest in BOUND_ARTIFACT_SHA256[group].values())


def test_runner_has_no_generation_encoding_or_free_parameters() -> None:
    source = inspect.getsource(run_step3_once)
    assert "decode_saved_mp4" in source and "extract_feature_matrix" in source
    for forbidden in ("encode_mp4", "apply_carrier", "validate_source_zip", "generation", "training"):
        assert f"{forbidden}(" not in source
    cli = (ROOT / "experiments/run_rc0_aisb_capture_fast_cpu.py").read_text(encoding="utf-8")
    assert "argparse" not in cli
    assert "add_argument" not in cli
    config = json.loads((ROOT / "configs/rc0_aisb_capture_fast_cpu.json").read_text(encoding="utf-8"))
    assert "threshold" not in json.dumps(config["input"], sort_keys=True)
    assert "candidate" not in json.dumps(config["input"], sort_keys=True)
