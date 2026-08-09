from __future__ import annotations

import inspect
import json
from pathlib import Path
import re

from sc_sstw_feasibility.rc0_fresh8_method_stability_gpu import (
    CATEGORY_ORDER,
    GROUP_ORDER,
    MODEL_REVISION,
    SEEDS,
    generate_official_base_frames,
    load_and_validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact8_identity_and_frozen_future_phase_c() -> None:
    config = load_and_validate_config(ROOT / "configs/rc0_fresh8_method_stability_gpu.json")
    fresh4 = json.loads((ROOT / "configs/rc0_fresh_exact4_base_gpu.json").read_text(encoding="utf-8"))
    assert config["attempt_order"] == list(GROUP_ORDER)
    assert tuple(item["category"] for item in config["groups"]) == CATEGORY_ORDER
    assert tuple(item["seed"] for item in config["groups"]) == SEEDS
    assert len(set(CATEGORY_ORDER)) == 8 and len(set(SEEDS)) == 8
    historical = {"orbital_glass", "articulated_paper", "liquid_mosaic", "clockwork_escapement", "steam_fins", "magnetic_filings"}
    assert set(GROUP_ORDER).isdisjoint(historical)
    assert config["model"] == fresh4["model"]
    assert config["generation"] == fresh4["generation"]
    assert config["encoding"] == fresh4["encoding"]
    phase_c = config["phase_c_frozen_not_executed"]
    assert (phase_c["derived_clean_count"], phase_c["derived_transformed_count"], phase_c["derived_total_count"]) == (32, 128, 160)
    assert phase_c["corrected_semantics"] == "clean_identity_only_transformed_each_own_truth"
    assert phase_c["success_counts"] == {"clean_off_reject": 16, "clean_positive_accept": 16, "transformed_off_reject": 64, "transformed_positive_accept": 64}


class _FakeGenerator:
    def __init__(self, device: str):
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int):
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


def test_official_base_only_call_is_fresh4_identical_and_carrier_free() -> None:
    config = load_and_validate_config(ROOT / "configs/rc0_fresh8_method_stability_gpu.json")
    pipe = _FakePipe()
    frames = generate_official_base_frames(pipe, _FakeTorch, config["groups"][0], config["generation"])
    assert len(frames) == 49
    assert set(pipe.kwargs) == {"prompt", "negative_prompt", "num_frames", "height", "width", "guidance_scale", "num_inference_steps", "generator"}
    assert pipe.kwargs["generator"].device == "cuda" and pipe.kwargs["generator"].seed == 54011
    source = inspect.getsource(generate_official_base_frames)
    assert all(word not in source for word in ("callback", "output_type", "latent"))
    assert config["generation"]["carrier"] == "none" and config["generation"]["hook"] == "none"
    assert config["model"]["revision"] == MODEL_REVISION


def test_notebook_first_cell_exact_ref_and_phase_c_absent() -> None:
    notebook = json.loads((ROOT / "notebooks/sc_sstw_rc0_fresh8_method_stability_gpu.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert "".join(code_cells[0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert source.count("drive.mount('/content/drive')") == 1
    assert source.index("drive.mount('/content/drive')") < source.index("REPOSITORY_URL =")
    assert "snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_files_only=False)" in source
    assert "run_rc0_fresh8_method_stability_gpu.py" in source
    assert "--output" in source
    for forbidden in ("apply_carrier", "target_only_observation", "evaluate_phase_cell", "OFF_R1", "schedule_a", "manual_latent"):
        assert forbidden not in source
    ref = re.search(r"AUTHORIZED_REF = '([^']*)'", source).group(1)
    run_id = re.search(r"RUN_ID = '([^']*)'", source).group(1)
    enabled = "AUTHORIZE_EXECUTION = True" in source and "AUTHORIZE_DRIVE_IO = True" in source
    assert (ref == "" and run_id == "" and not enabled) or (re.fullmatch(r"[0-9a-f]{40}", ref) and re.fullmatch(r"[0-9a-f]{16}", run_id) and enabled)


def test_old_fresh4_scientific_files_are_unchanged_from_baseline() -> None:
    baseline = "dc7b089837d50fcfb5ab916ac3739e065dfc3d50"
    guarded = [
        "configs/rc0_fresh_exact4_base_gpu.json",
        "protocols/rc0_fresh_exact4_base_gpu.md",
        "src/sc_sstw_feasibility/rc0_fresh_exact4_base_gpu.py",
        "src/sc_sstw_feasibility/rc0_fresh4_effectiveness_cpu.py",
    ]
    import subprocess
    for relative in guarded:
        current = subprocess.run(["git", "hash-object", relative], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        frozen = subprocess.run(["git", "rev-parse", f"{baseline}:{relative}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        assert current == frozen
