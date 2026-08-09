from __future__ import annotations

import inspect
import json
from pathlib import Path
import re

from sc_sstw_feasibility.rc0_fresh_exact4_base_gpu import (
    GROUP_ORDER,
    MODEL_REVISION,
    SEEDS,
    generate_official_base_frames,
    load_and_validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact4_config_and_fresh_content_identity() -> None:
    config = load_and_validate_config(ROOT / "configs/rc0_fresh_exact4_base_gpu.json")
    assert config["attempt_order"] == list(GROUP_ORDER)
    assert tuple(item["seed"] for item in config["groups"]) == SEEDS
    assert set(SEEDS).isdisjoint({52001, 52002})
    assert len({item["content_grammar"] for item in config["groups"]}) == 4
    assert all(item["group_id"] not in {"orbital_glass", "articulated_paper"} for item in config["groups"])
    assert config["model"]["revision"] == MODEL_REVISION


class _FakeGenerator:
    def __init__(self, device: str):
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int) -> "_FakeGenerator":
        self.seed = seed
        return self


class _FakeTorch:
    Generator = _FakeGenerator


class _Result:
    frames = [[object() for _ in range(49)]]


class _FakePipe:
    def __init__(self) -> None:
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return _Result()


def test_official_pipeline_call_has_no_carrier_hook_or_manual_decode() -> None:
    config = load_and_validate_config(ROOT / "configs/rc0_fresh_exact4_base_gpu.json")
    pipe = _FakePipe()
    frames = generate_official_base_frames(pipe, _FakeTorch, config["groups"][0], config["generation"])
    assert len(frames) == 49
    assert set(pipe.kwargs) == {"prompt", "negative_prompt", "num_frames", "height", "width", "guidance_scale", "num_inference_steps", "generator"}
    assert pipe.kwargs["generator"].device == "cuda" and pipe.kwargs["generator"].seed == 53011
    source = inspect.getsource(generate_official_base_frames)
    assert "callback" not in source and "output_type" not in source and "latent" not in source


def test_phase_c_real_mp4_frame_maps_are_exact_and_nontrivial() -> None:
    config = json.loads((ROOT / "configs/rc0_fresh_exact4_base_gpu.json").read_text(encoding="utf-8"))
    phase_c = config["phase_c_frozen_not_executed"]
    transforms = {item["id"]: item for item in phase_c["transforms"]}
    assert tuple(transforms) == ("identity", "delete6_duplicate12", "local_phase_plus1", "local_phase_minus1")
    assert all(len(item["symbol_source_indices"]) == 13 and len(item["frame_source_indices"]) == 49 for item in transforms.values())
    assert transforms["delete6_duplicate12"]["symbol_source_indices"] == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 12]
    assert transforms["local_phase_plus1"]["symbol_source_indices"] == [0, 1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 11, 12]
    assert transforms["local_phase_minus1"]["symbol_source_indices"] == [0, 1, 2, 3, 4, 5, 5, 7, 8, 9, 10, 11, 12]
    assert phase_c["pixel_chroma_relative_amplitude"] == 6 / 255


def test_notebook_is_thin_exact_ref_base_only_delivery() -> None:
    notebook = json.loads((ROOT / "notebooks/sc_sstw_rc0_fresh_exact4_base_gpu.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    first = "".join(code_cells[0]["source"])
    assert first == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    source = "\n".join("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"] for cell in code_cells)
    assert source.count("drive.mount('/content/drive')") == 1
    assert source.index("drive.mount('/content/drive')") < source.index("REPOSITORY_URL =")
    assert "snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=False)" in source
    assert "run_rc0_fresh_exact4_base_gpu.py" in source
    assert "--output" in source
    for forbidden in ("--generate", "carrier residual", "target_only_observation", "evaluate_phase_cell", "OFF_R1", "schedule_a"):
        assert forbidden not in source
    ref = re.search(r"AUTHORIZED_REF = '([^']*)'", source).group(1)
    run_id = re.search(r"RUN_ID = '([^']*)'", source).group(1)
    enabled = "AUTHORIZE_EXECUTION = True" in source and "AUTHORIZE_DRIVE_IO = True" in source
    assert (ref == "" and run_id == "" and not enabled) or (re.fullmatch(r"[0-9a-f]{40}", ref) and re.fullmatch(r"[0-9a-f]{16}", run_id) and enabled)
