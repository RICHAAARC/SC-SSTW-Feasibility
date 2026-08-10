from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC))

from sstw.s1_local_pair_dictionary import (  # noqa: E402
    AXIS_ORDER,
    BRANCH_ORDER,
    RADII,
    STATUS_NO_GO,
    STATUS_READY,
    S1InstrumentationError,
    PairDictionaryWanProcessor,
    dictionary_exit_code,
    install_pair_dictionary_processor,
    pair_indices,
    screen_pair_dictionary,
)


CONFIG_PATH = ROOT / "configs/s1_local_pair_dictionary.json"
PLAN_PATH = ROOT / "plans/s1_local_pair_dictionary.json"
PROTOCOL_PATH = ROOT / "protocols/s1_local_pair_dictionary.md"
MODULE_PATH = ROOT / "src/sstw/s1_local_pair_dictionary.py"
RUNNER_PATH = ROOT / "experiments/run_s1_local_pair_dictionary.py"
NOTEBOOK_PATH = ROOT / "notebooks/sstw_s1_local_pair_dictionary.ipynb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _thresholds() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["candidate_thresholds"]


def _captures(*, signal: bool = True) -> dict[str, object]:
    result: dict[str, object] = {}
    for branch in BRANCH_ORDER:
        result[branch] = {}
        for axis in AXIS_ORDER:
            result[branch][axis] = {}
            for radius in RADII:
                values = np.zeros((13, 12, 2), dtype=np.float64)
                if signal:
                    if axis == "B1_horizontal":
                        amplitudes = np.array([0.02] * 6 + [1e-10] * 6)
                    else:
                        amplitudes = np.array([1e-10] * 6 + [0.02] * 6)
                    values[..., 0] = amplitudes[None, :]
                    values[..., 1] = amplitudes[None, :]
                result[branch][axis][str(radius)] = values.tolist()
    return result


def test_frozen_inputs_and_prior_failure_registration() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert _sha(ROOT / config["base_s1"]["config_path"]) == config["base_s1"]["config_raw_sha256"]
    assert _sha(ROOT / config["base_s1"]["plan_path"]) == config["base_s1"]["plan_raw_sha256"]
    assert _sha(ROOT / "SSTW_METHOD_AUTHORITY.md") == config["base_s1"]["authority_raw_sha256"]
    assert config["input_result"] == {
        "run_id": "0d3223669982a6c3",
        "archive_sha256": "07a4c137382ce67aaf16d1775ea244287568a7c2d7d3fedcc5fa104a75a31cdb",
        "source_commit": "e8923e5de8b752ed3211af98c4120340acdea100",
        "status": "S1_NO_GO_THIS_CONSTRUCTION",
        "structural_passed": True,
    }
    assert config["frozen_construction"]["lambda"] == 1.0
    base_thresholds = json.loads(
        (ROOT / "configs/s1_real_dit_relation_primitive.json").read_text(encoding="utf-8")
    )["thresholds"]
    for name, value in config["candidate_thresholds"].items():
        assert base_thresholds[name] == value
    assert plan["exact_transformer_calls"] == 10
    assert plan["no_block_or_velocity_injection"] is True
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    for marker in ("1.20", "5.89", "0.0045", "0.032", "0.0666", "0.491", "0.024", "16", "120", "0.406", "0.413"):
        assert marker in text
    assert "S2 remains HOLD" in text


def test_dictionary_geometry_coefficients_and_budget() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["frozen_construction"]["coefficient_pair"] == pytest.approx([1 / math.sqrt(2), -1 / math.sqrt(2)])
    for axis in AXIS_ORDER:
        for radius in RADII:
            pairs = pair_indices(axis, radius)
            assert len(pairs) == 13
            for query, (left, right) in zip(config["frozen_construction"]["query_indices"], pairs, strict=True):
                stride = radius if axis == "B1_horizontal" else 32 * radius
                assert (left, right) == (query - stride, query + stride)
                assert 0 <= left < query < right < 8320
    assert config["call_budget"]["total_transformer_calls"] == 10
    assert config["call_budget"]["retry_count"] == 0


def test_unique_screen_is_deterministic_and_lexicographic() -> None:
    result = screen_pair_dictionary(_captures(), _thresholds())
    assert result["status"] == STATUS_READY
    assert result["selected_pair"] == {"horizontal_radius": 1, "vertical_radius": 1}
    assert result["eligible_order"] == {axis: list(RADII) for axis in AXIS_ORDER}
    selected = next(item for item in result["combinations"] if item["horizontal_radius"] == 1 and item["vertical_radius"] == 1)
    assert selected["passed"] is True
    for branch in BRANCH_ORDER:
        assert selected["branches"][branch]["axis_absolute_cosine"] < 0.5
        assert selected["branches"][branch]["maximum_cross_ratio"] <= 0.25


def test_no_eligible_and_zero_denominators_are_json_safe() -> None:
    result = screen_pair_dictionary(_captures(signal=False), _thresholds())
    assert result["status"] == STATUS_NO_GO
    assert result["selected_pair"] is None
    assert result["combinations"] == []
    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    assert "NaN" not in encoded and "Infinity" not in encoded
    record = result["candidates"]["B1_horizontal"]["1"]["branches"]["cond"]
    assert record["odd_to_noise_or_ulp"] == 0.0
    assert record["even_to_odd"] is None
    assert record["temporal_total_variation"] is None
    assert record["passed"] is False


@pytest.mark.parametrize("mutation", ("shape", "nan", "probability", "missing_radius", "missing_branch"))
def test_capture_contract_fails_closed(mutation: str) -> None:
    captures = _captures()
    if mutation == "shape":
        captures["cond"]["B1_horizontal"]["1"] = np.zeros((12, 12, 2)).tolist()
    elif mutation == "nan":
        captures["cond"]["B1_horizontal"]["1"][0][0][0] = float("nan")
    elif mutation == "probability":
        captures["cond"]["B1_horizontal"]["1"][0][0] = [0.8, 0.8]
    elif mutation == "missing_radius":
        captures["cond"]["B1_horizontal"].pop("8")
    else:
        captures.pop("uncond")
    with pytest.raises(S1InstrumentationError):
        screen_pair_dictionary(captures, _thresholds())


def test_one_axis_failure_cannot_be_hidden() -> None:
    captures = _captures()
    for branch in BRANCH_ORDER:
        for radius in RADII:
            captures[branch]["B2_vertical"][str(radius)] = np.zeros((13, 12, 2)).tolist()
    result = screen_pair_dictionary(captures, _thresholds())
    assert result["status"] == STATUS_NO_GO
    assert result["eligible_order"]["B1_horizontal"] == list(RADII)
    assert result["eligible_order"]["B2_vertical"] == []


def test_processor_and_runner_are_off_capture_only() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "dispatch_attention_fn(" in source
    assert "torch.softmax(logits, dim=-1)" in source
    assert "output_replaced\": False" in source
    assert "full_heads = dispatch_attention_fn" in source
    assert "target_attention.set_processor(original)" in source
    assert "for scheduler_index in range(4)" in source
    assert "for branch in BRANCH_ORDER" in source
    assert "VAE_was_run\": False" in source
    assert "MP4_was_written\": False" in source
    assert "vae.decode" not in source.lower()
    assert "decode(" not in source.lower()
    assert "encode_video" not in source
    assert "8320, 8320" not in source


def test_processor_record_defaults_to_no_replacement() -> None:
    class TorchFlags:
        @staticmethod
        def is_grad_enabled() -> bool:
            return False

        @staticmethod
        def is_inference_mode_enabled() -> bool:
            return True

    processor = PairDictionaryWanProcessor(TorchFlags)
    processor.set_context(call_index=1, scheduler_index=0, timestep=999, branch="cond", capture=False)
    assert processor.context == {
        "call_index": 1,
        "scheduler_index": 0,
        "timestep": 999,
        "branch": "cond",
        "capture": False,
    }
    with pytest.raises(S1InstrumentationError):
        processor.set_context(call_index=1, scheduler_index=0, timestep=999, branch="guidance", capture=False)


def test_fake_wan_adapter_installs_only_block14_attn1() -> None:
    class Attention:
        def __init__(self) -> None:
            self.processor = object()
            self.received = None

        def set_processor(self, value: object) -> None:
            self.received = value
            self.processor = value

    class Block:
        def __init__(self) -> None:
            self.attn1 = Attention()

    class Transformer:
        def __init__(self) -> None:
            self.blocks = [Block() for _ in range(30)]

    transformer = Transformer()
    untouched = [block.attn1.processor for block in transformer.blocks]
    processor, original = install_pair_dictionary_processor(transformer, object())
    assert transformer.blocks[14].attn1.processor is processor
    assert original is untouched[14]
    for index, block in enumerate(transformer.blocks):
        if index != 14:
            assert block.attn1.processor is untouched[index]


@pytest.mark.parametrize(("status", "expected"), ((STATUS_READY, 0), (STATUS_NO_GO, 3)))
def test_exit_code(status: str, expected: int) -> None:
    assert dictionary_exit_code(status) == expected


def test_cli_statuses_without_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("s1_dictionary_cli", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_dictionary_once", lambda **_: {"status": STATUS_NO_GO})
    monkeypatch.setattr(__import__("sys"), "argv", [str(RUNNER_PATH), "--output", str(tmp_path / "out")])
    assert module.main() == 3
    assert json.loads(capsys.readouterr().out)["status"] == STATUS_NO_GO

    def fail(**_: object) -> object:
        raise RuntimeError("synthetic connection failure")

    monkeypatch.setattr(module, "run_dictionary_once", fail)
    assert module.main() == 2
    invalid = json.loads(capsys.readouterr().out)
    assert invalid["status"] == "INSTRUMENTATION_INSUFFICIENT"
    assert invalid["message"] == "synthetic connection failure"


def test_notebook_static_contract() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells[0]["source"] == [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
    ]
    source = "".join("".join(cell["source"]) for cell in code_cells)
    assert "run_s1_local_pair_dictionary.py" in source
    assert "PAIR_DICTIONARY_READY" in source and "PAIR_DICTIONARY_NO_GO" in source
    assert "--output" in source
    assert "vae" not in source.lower()
    assert "mp4" not in source.lower()
    assert "S2" not in source
    assert "AUTHORIZED_REF = '0000000000000000000000000000000000000000'" in source
    assert "AUTHORIZE_EXECUTION = False" in source
    assert "AUTHORIZE_DRIVE_IO = False" in source
    for cell in code_cells:
        compile("".join(cell["source"]), "notebook-cell", "exec")
