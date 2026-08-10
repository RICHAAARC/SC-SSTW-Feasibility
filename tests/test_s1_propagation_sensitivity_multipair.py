from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
__import__("sys").path.insert(0, str(SRC))

from sstw.s1_propagation_sensitivity_multipair import (  # noqa: E402
    BLOCKS,
    CONDITION_ORDER,
    SIGN_ORDER,
    STATUS_NO_GO,
    STATUS_READY,
    S1InstrumentationError,
    all_basis_updates,
    assert_scheduler_snapshot_unchanged,
    capture_scheduler_snapshot,
    fork_scheduler_snapshot,
    install_processor,
    load_inputs,
    propagation_exit_code,
    solve_cfg_weights,
    scheduler_state_summary,
    stage1_ranking,
    two_pair_basis,
    validate_processor_record,
)
from sstw.s1_real_dit_relation_primitive import evaluate_preregistered_statistics  # noqa: E402


CONFIG = ROOT / "configs/s1_propagation_sensitivity_multipair.json"
PLAN = ROOT / "plans/s1_propagation_sensitivity_multipair.json"
PROTOCOL = ROOT / "protocols/s1_propagation_sensitivity_multipair.md"
MODULE = ROOT / "src/sstw/s1_propagation_sensitivity_multipair.py"
RUNNER = ROOT / "experiments/run_s1_propagation_sensitivity_multipair.py"
NOTEBOOK = ROOT / "notebooks/sstw_s1_propagation_sensitivity_multipair.ipynb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relation_probe() -> dict[str, object]:
    shape = (13, 12, 2)
    probe = {condition: {} for condition in CONDITION_ORDER}
    head = np.linspace(0.6, 1.4, 12)[None, :, None]
    time = np.linspace(0.9, 1.1, 13)[:, None, None]
    for branch_index, branch in enumerate(("cond", "uncond")):
        odd_b1 = np.zeros(shape)
        odd_b2 = np.zeros(shape)
        odd_b1[..., 0] = (time * (head if branch_index else 1.0))[..., 0]
        odd_b2[..., 1] = (time[::-1] * (head[..., ::-1, :] if branch_index else 1.0))[..., 0]
        even_b1 = np.zeros(shape)
        even_b2 = np.zeros(shape)
        even_b1[..., branch_index] = 0.02 * (1.0 + np.arange(12)[None, :] / 20.0)
        even_b2[..., 1 - branch_index] = 0.015 * (1.0 + np.arange(13)[:, None] / 20.0)
        zero = np.zeros(shape)
        values = {
            "OFF_R1": zero,
            "OFF_R2": zero,
            "PLUS_B1": even_b1 + odd_b1,
            "MINUS_B1": even_b1 - odd_b1,
            "PLUS_B2": even_b2 + odd_b2,
            "MINUS_B2": even_b2 - odd_b2,
        }
        for condition in CONDITION_ORDER:
            probe[condition][branch] = {"relation": values[condition].tolist()}
    return probe


def _zero_numeric_probe() -> dict[str, object]:
    shapes = {"relation": (13, 12, 2), "block": (13, 1536), "velocity": (13, 16, 2, 2)}
    return {condition: {branch: {layer: np.zeros(shape).tolist() for layer, shape in shapes.items()} for branch in ("cond", "uncond")} for condition in CONDITION_ORDER}


def test_authority_inputs_and_frozen_budget() -> None:
    config, plan, base = load_inputs(ROOT)
    assert config["method_authority"] == {
        "commit": "0f85bd70de6f042c68560ed348722fc4fbd112e9",
        "tree": "d791cff7b5cbebb6ec690f16196bde9ee6d50bd1",
        "raw_sha256": "85c32f49897a6b7c6248d5fd6fe4c7a307b4b841dc7388551d095854b6189292",
    }
    assert _sha(ROOT / "SSTW_METHOD_AUTHORITY.md") == config["method_authority"]["raw_sha256"]
    assert config["input_result"]["run_id"] == "79f92e73ef3a6223"
    assert config["input_result"]["archive_sha256"] == "fbedc7cccdb31889621dfc6e228888062a1d22978601c273d753674777bab229"
    assert config["call_budget"]["stage1_total"] == 108
    assert config["call_budget"]["full_path_transformer_calls"] == 170
    assert config["call_budget"]["possible_terminal_transformer_calls"] == [108, 138, 146, 162, 170]
    assert plan["stage1_constructions"] == 12
    assert base["flow_support"]["lambda"] == 1.0


@pytest.mark.parametrize("axis,radii", (("B1", (2, 5)), ("B2", (2, 8))))
@pytest.mark.parametrize("sign", SIGN_ORDER)
def test_two_pair_basis_is_sparse_zero_sum_unit_l2(axis: str, radii: tuple[int, int], sign: str) -> None:
    rows = two_pair_basis(axis, sign)
    assert len(rows) == 13
    stride = 1 if axis == "B1" else 32
    for query, row in zip((336,976,1616,2256,2896,3536,4176,4816,5456,6096,6736,7376,8016), rows, strict=True):
        assert [key for key, _ in row] == [query-stride*radii[0], query+stride*radii[0], query-stride*radii[1], query+stride*radii[1]]
        coefficients = np.asarray([value for _, value in row])
        assert coefficients.sum() == pytest.approx(0.0)
        assert np.linalg.norm(coefficients) == pytest.approx(1.0)
        assert set(np.abs(coefficients)) == {0.5}
    updates = all_basis_updates(axis, sign, 1.0)
    assert len(updates) == 52


def test_cfg_generalized_eigen_solution_is_unique_normalized_and_oriented() -> None:
    result = solve_cfg_weights(_relation_probe())
    weights = np.asarray([result["weights"]["cond"], result["weights"]["uncond"]])
    assert np.linalg.norm(weights) == pytest.approx(1.0)
    assert result["generalized_eigenvalues"][1] > result["generalized_eigenvalues"][0]
    assert all(value > 0.0 for value in result["guided_axis_orientations"])
    assert np.isfinite(np.asarray(result["signal_covariance"])).all()
    assert np.isfinite(np.asarray(result["nuisance_covariance"])).all()


def test_cfg_zero_or_rank_deficient_is_invalid() -> None:
    probe = _relation_probe()
    for condition in CONDITION_ORDER:
        for branch in ("cond", "uncond"):
            probe[condition][branch]["relation"] = np.zeros((13, 12, 2)).tolist()
    with pytest.raises(S1InstrumentationError, match="rank deficient"):
        solve_cfg_weights(probe)


class _FakeTorch:
    @staticmethod
    def is_tensor(_value: object) -> bool:
        return False


class _StatefulFakeUniPC:
    order = 1

    def __init__(self, step_index: int) -> None:
        self._step_index = step_index
        self.lower_order_nums = step_index
        self.model_outputs = [np.asarray([float(index), float(step_index)]) for index in range(3)]
        self.timestep_list = [index for index in range(step_index)]
        self.last_sample = np.asarray([float(step_index)])
        self.config = {"solver_order": 3, "solver_type": "bh2"}

    def step(self, _model_output: object, timestep: int, sample: np.ndarray, *, return_dict: bool) -> tuple[np.ndarray]:
        assert return_dict is False
        this_order = 0 if timestep != self._step_index else min(3, self.lower_order_nums + 1)
        assert this_order > 0
        self.model_outputs.pop(0)
        self.model_outputs.append(np.asarray([float(timestep), float(sample.sum())]))
        self.timestep_list.append(timestep)
        self.last_sample = sample.copy()
        self.lower_order_nums = min(self.lower_order_nums + 1, 3)
        self._step_index += 1
        return (sample + float(timestep + 1),)


def test_stateful_scheduler_reuse_reproduces_failure_but_independent_forks_pass() -> None:
    torch = _FakeTorch()
    retained, summary = capture_scheduler_snapshot(
        _StatefulFakeUniPC(3), torch, expected_step_index=3, label="step3"
    )
    shared = fork_scheduler_snapshot(retained, summary, torch, label="old_shared")
    latent = np.asarray([1.0, 2.0])
    for timestep in (3, 4):
        latent = shared.step(None, timestep, latent, return_dict=False)[0]
    with pytest.raises(AssertionError):
        shared.step(None, 3, np.asarray([1.0, 2.0]), return_dict=False)

    starts = []
    finishes = []
    for condition in ("PLUS_B1", "MINUS_B1", "PLUS_B2", "MINUS_B2"):
        fork = fork_scheduler_snapshot(retained, summary, torch, label=condition)
        condition_latent = np.asarray([1.0, 2.0])
        starts.append(hashlib.sha256(condition_latent.tobytes()).hexdigest())
        for timestep in (3, 4):
            condition_latent = fork.step(None, timestep, condition_latent, return_dict=False)[0]
        finishes.append(condition_latent.copy())
        assert_scheduler_snapshot_unchanged(retained, summary, torch, label=condition)
    assert len(set(starts)) == 1
    assert all(np.array_equal(value, finishes[0]) for value in finishes)


def test_step3_fit_apply_and_step4_off_forks_do_not_mutate_snapshots() -> None:
    torch = _FakeTorch()
    step3, step3_summary = capture_scheduler_snapshot(
        _StatefulFakeUniPC(3), torch, expected_step_index=3, label="step3"
    )
    step4, step4_summary = capture_scheduler_snapshot(
        _StatefulFakeUniPC(4), torch, expected_step_index=4, label="step4"
    )
    for purpose in ("three_fit", "three_apply"):
        for condition in ("PLUS_B1", "MINUS_B1", "PLUS_B2", "MINUS_B2"):
            fork = fork_scheduler_snapshot(step3, step3_summary, torch, label=f"{purpose}:{condition}")
            sample = np.asarray([0.0])
            for timestep in (3, 4):
                sample = fork.step(None, timestep, sample, return_dict=False)[0]
            assert_scheduler_snapshot_unchanged(step3, step3_summary, torch, label=purpose)
    off = fork_scheduler_snapshot(step4, step4_summary, torch, label="off")
    off.step(None, 4, np.asarray([0.0]), return_dict=False)
    assert_scheduler_snapshot_unchanged(step4, step4_summary, torch, label="off")
    assert scheduler_state_summary(step3, torch) == step3_summary
    assert scheduler_state_summary(step4, torch) == step4_summary


def test_scheduler_snapshot_requires_history_and_preserves_full_170_budget() -> None:
    class MissingHistory:
        _step_index = 3

    with pytest.raises(S1InstrumentationError, match="history fields unavailable"):
        scheduler_state_summary(MissingHistory(), _FakeTorch())
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    budget = config["call_budget"]
    assert budget["stage1_shared_prefix"] + budget["stage1_shared_step4_off"] + budget["stage1_active"] == 108
    assert 108 + budget["stage2_single_weighted_apply"] + budget["stage2_three_off_continuation_and_terminal_repeats"] + budget["stage2_three_unweighted_fit"] + budget["stage2_three_weighted_apply"] == 170
    assert budget["possible_terminal_transformer_calls"] == [108, 138, 146, 162, 170]


def test_stateful_fake_executes_full_170_call_schedule_with_isolated_forks() -> None:
    torch = _FakeTorch()
    step3, step3_summary = capture_scheduler_snapshot(
        _StatefulFakeUniPC(3), torch, expected_step_index=3, label="step3"
    )
    step4, step4_summary = capture_scheduler_snapshot(
        _StatefulFakeUniPC(4), torch, expected_step_index=4, label="step4"
    )
    calls = 108
    calls += 4 * 2  # selected single-step weighted application

    off = fork_scheduler_snapshot(step4, step4_summary, torch, label="three_off")
    calls += 2  # cond/uncond step-4 forward
    off.step(None, 4, np.asarray([0.0]), return_dict=False)
    calls += 2 * 2  # two step-5 OFF repeats on both branches
    assert_scheduler_snapshot_unchanged(step4, step4_summary, torch, label="three_off")

    start_hashes = []
    for purpose in ("three_fit", "three_apply"):
        for condition_index, condition in enumerate(("PLUS_B1", "MINUS_B1", "PLUS_B2", "MINUS_B2")):
            fork = fork_scheduler_snapshot(step3, step3_summary, torch, label=f"{purpose}:{condition}")
            latent = np.asarray([1.0, 2.0])
            start_hashes.append(hashlib.sha256(latent.tobytes()).hexdigest())
            for timestep in (3, 4, 5):
                calls += 2  # cond/uncond real forward
                if timestep != 5:
                    # Distinct carriers may produce distinct later latents; the common
                    # starting latent and scheduler history are the frozen contract.
                    model_output = np.asarray([float(condition_index), float(timestep)])
                    latent = fork.step(model_output, timestep, latent, return_dict=False)[0]
            assert_scheduler_snapshot_unchanged(step3, step3_summary, torch, label=purpose)
    assert len(set(start_hashes)) == 1
    assert calls == 170


def test_no_signal_stage1_is_json_safe_and_unusable() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    overrides = {(branch, layer, axis): 0.0 for branch in ("cond", "uncond", "guidance") for layer in ("block", "velocity") for axis in ("B1", "B2")}
    evaluation = evaluate_preregistered_statistics(_zero_numeric_probe(), config["thresholds"], overrides)
    assert stage1_ranking(evaluation, 7, "coherent_sum", "coherent_sum") is None
    encoded = json.dumps(evaluation, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_processor_record_contract() -> None:
    valid = {"changed_logit_count": 52, "pair_sum_zero": True, "selected_bias_shape": [1,1,13,8320], "dense_full_bias": False, "grad_enabled": False, "inference_mode_enabled": True, "relation_bank": np.zeros((13,12,4)).tolist()}
    validate_processor_record(valid, active=True, captured=True)
    for name in ("grad_enabled", "pair_sum_zero"):
        broken = dict(valid)
        broken[name] = not broken[name]
        with pytest.raises(S1InstrumentationError):
            validate_processor_record(broken, active=True, captured=True)


def test_fake_wan_adapter_targets_only_requested_frozen_block() -> None:
    class Attention:
        def __init__(self) -> None:
            self.processor = object()

        def set_processor(self, value: object) -> None:
            self.processor = value

    class Block:
        def __init__(self) -> None:
            self.attn1 = Attention()

    class Transformer:
        def __init__(self) -> None:
            self.blocks = [Block() for _ in range(30)]

    transformer = Transformer()
    originals = [block.attn1.processor for block in transformer.blocks]
    processor, original = install_processor(transformer, object(), 22)
    assert original is originals[22]
    assert transformer.blocks[22].attn1.processor is processor
    assert all(block.attn1.processor is originals[index] for index, block in enumerate(transformer.blocks) if index != 22)
    with pytest.raises(S1InstrumentationError):
        install_processor(transformer, object(), 21)


def test_static_real_call_graph_and_no_proxy() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "dispatch_native_selected_attention" in source
    assert "for step in range(4)" in source
    assert "for block in BLOCKS" in source
    assert "three_steps345" in source
    assert "call_count not in config" in source
    assert "torch.inference_mode()" in source
    assert "capture_block_output" in source and "_extract_velocity_slice" in source
    assert "copy.deepcopy(snapshot)" in source
    assert "condition_scheduler.step" in source
    assert "off_scheduler.step" in source
    assert "pipe.scheduler.step(guided, timesteps[step], condition_latent" not in source
    for forbidden in ("vae.decode", "encode_video", "saved.mp4", "lambda_scan"):
        assert forbidden not in source.lower()


@pytest.mark.parametrize(("status", "expected"), ((STATUS_READY, 0), (STATUS_NO_GO, 3)))
def test_exit_codes(status: str, expected: int) -> None:
    assert propagation_exit_code(status) == expected


def test_cli_three_statuses_without_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("propagation_cli", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_propagation_once", lambda **_: {"status": STATUS_NO_GO})
    monkeypatch.setattr(__import__("sys"), "argv", [str(RUNNER), "--output", str(tmp_path / "out")])
    assert module.main() == 3
    assert json.loads(capsys.readouterr().out)["status"] == STATUS_NO_GO
    monkeypatch.setattr(module, "run_propagation_once", lambda **_: (_ for _ in ()).throw(RuntimeError("fake runtime")))
    assert module.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "INSTRUMENTATION_INSUFFICIENT"


def test_notebook_static_contract() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code[0]["source"] == ["from google.colab import drive\n", "drive.mount('/content/drive')\n"]
    source = "".join("".join(cell["source"]) for cell in code)
    assert "run_s1_propagation_sensitivity_multipair.py" in source
    assert "AUTHORIZED_REF = '0000000000000000000000000000000000000000'" in source
    assert "AUTHORIZE_EXECUTION = False" in source and "AUTHORIZE_DRIVE_IO = False" in source
    assert "CUDA and BF16 capability are required" in source
    assert "NVIDIA L4" not in source
    for cell in code:
        compile("".join(cell["source"]), "notebook-cell", "exec")
