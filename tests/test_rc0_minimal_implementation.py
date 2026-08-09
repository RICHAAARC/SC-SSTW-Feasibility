from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any
import zipfile

import numpy as np
import pytest

from src.sc_sstw_feasibility.learned_observation import FRAME_GROUPS
from src.sc_sstw_feasibility.rc0_causal_localization_v2 import (
    STATUS_INSUFFICIENT,
    STATUS_INVALID,
    STATUS_P_FAIL,
    STATUS_P_PASS_R_FAIL,
    STATUS_P_PASS_R_PASS,
    schedule_a,
    schedule_b,
)
from src.sc_sstw_feasibility.rc0_generation import (
    GenerationReceipt,
    consume_generation_receipt,
    run_rc0_generation,
)
from src.sc_sstw_feasibility.rc0_minimal_implementation import (
    CONFIG_PATH,
    CONFIG_RAW_SHA256,
    DIAGNOSTIC_BOOTSTRAP_SCHEMA,
    DIAGNOSTIC_CLASS,
    DIAGNOSTIC_CONFIG_PATH,
    DIAGNOSTIC_DECISION_BUILD_BLIND,
    DIAGNOSTIC_DECISION_CARRIER_NOT_FEASIBLE,
    DIAGNOSTIC_DECISION_INSUFFICIENT,
    DIAGNOSTIC_FEASIBLE,
    DIAGNOSTIC_INSUFFICIENT_TO_DECIDE,
    DIAGNOSTIC_NOT_FEASIBLE,
    DIAGNOSTIC_PROTOCOL_PATH,
    EVIDENCE_CPU_HARNESS,
    EXTRACTOR_IDENTITY,
    MANIFEST_SCHEMA,
    NOTEBOOK_PATH,
    OUTPUT_SCHEMA,
    PACKAGE_STATUS_INSUFFICIENT,
    PACKAGE_STATUS_INVALID,
    PACKAGE_STATUS_P_FAIL,
    PACKAGE_STATUS_P_PASS_R_FAIL,
    PACKAGE_STATUS_P_PASS_R_PASS,
    PACKAGE_TO_PROTOCOL_OUTCOME,
    PLAN_PATH,
    PLAN_RAW_SHA256,
    PROTOCOL_PATH,
    PROTOCOL_RAW_SHA256,
    REQUIRED_SOURCE_PATHS,
    RUNNER_PATH,
    InvalidExperiment,
    Evaluation,
    canonical_json_bytes,
    evaluate_execution,
    evaluation_audit,
    preflight,
    sha256_file,
    validate_execution_record,
    validate_manifest_schema,
    validate_diagnostic_bootstrap_schema,
    validate_status_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_path() -> Any:
    """Use a native Linux filesystem required by atomic renameat2 tests."""

    with tempfile.TemporaryDirectory(prefix="rc0-tests-", dir="/tmp") as directory:
        yield Path(directory)


def _runner_module() -> Any:
    path = ROOT / RUNNER_PATH
    spec = importlib.util.spec_from_file_location("rc0_runner_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _runner_module()


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=path, check=True, capture_output=True, text=True).stdout.strip()


def _make_clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "clean-repo"
    repo.mkdir()
    for relative in REQUIRED_SOURCE_PATHS:
        source = ROOT / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(("git", "config", "user.email", "rc0-test@example.invalid"), cwd=repo, check=True)
    subprocess.run(("git", "config", "user.name", "RC0 Test"), cwd=repo, check=True)
    subprocess.run(("git", "add", "--", *REQUIRED_SOURCE_PATHS), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-q", "-m", "synthetic rc0 source fixture"), cwd=repo, check=True)
    return repo


def _manifest(repo: Path) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    source_files = {relative: sha256_file(repo / relative) for relative in REQUIRED_SOURCE_PATHS}
    attempts = [
        {"attempt_index": index, "group_id": group, "condition": condition}
        for index, (group, condition) in enumerate(
            ((group, condition) for group in ("orbital_glass", "articulated_paper") for condition in ("OFF_R1", "OFF_R2", "A", "B")),
            start=1,
        )
    ]
    return {
        "schema_version": 1,
        "manifest_schema": MANIFEST_SCHEMA,
        "implementation_schema": "sc_sstw_rc0_minimal_implementation_v1",
        "protocol_id": "sc_sstw_rc0_causal_localization_v2",
        "expected_source": {"head": head, "tree": tree, "dirty": False},
        "source_files": source_files,
        "frozen_identities": {
            "config": {"path": CONFIG_PATH, "sha256": CONFIG_RAW_SHA256},
            "plan": {"path": PLAN_PATH, "sha256": PLAN_RAW_SHA256},
            "protocol": {"path": PROTOCOL_PATH, "sha256": PROTOCOL_RAW_SHA256},
            "notebook": {"path": NOTEBOOK_PATH, "sha256": source_files[NOTEBOOK_PATH]},
            "extractor": dict(EXTRACTOR_IDENTITY),
        },
        "experiment": {
            "group_order": ["orbital_glass", "articulated_paper"],
            "condition_order": ["OFF_R1", "OFF_R2", "A", "B"],
            "attempt_order": attempts,
            "attempt_budget": 8,
            "retry_policy": "no_retry_no_replacement_no_additional_attempts",
        },
        "evidence_policy": {
            "production_entry": "same_process_runner_generate_only",
            "external_production_package_permitted": False,
            "formal_result": False,
            "stage_progression_allowed": False,
        },
        "command_schema": {
            "runner": RUNNER_PATH,
            "required_arguments": ["--generate", "--manifest", "--output", "--source-commit"],
            "forbidden_arguments": ["--execution-package", "--synthetic-fixture", "--backend", "--device", "--receipt"],
        },
        "output_schema": OUTPUT_SCHEMA,
    }


def _write_manifest(tmp_path: Path, repo: Path) -> Path:
    path = tmp_path / "synthetic-authorization.json"
    path.write_text(json.dumps(_manifest(repo), separators=(",", ":"), allow_nan=False) + "\n")
    return path


def _diagnostic_bootstrap(repo: Path) -> dict[str, Any]:
    manifest = _manifest(repo)
    return {
        "schema_version": 1,
        "bootstrap_schema": DIAGNOSTIC_BOOTSTRAP_SCHEMA,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "implementation_schema": manifest["implementation_schema"],
        "protocol_id": manifest["protocol_id"],
        "expected_source": manifest["expected_source"],
        "source_files": manifest["source_files"],
        "frozen_identities": manifest["frozen_identities"],
        "diagnostic_identities": {
            "protocol": {"path": DIAGNOSTIC_PROTOCOL_PATH, "sha256": manifest["source_files"][DIAGNOSTIC_PROTOCOL_PATH]},
            "config": {"path": DIAGNOSTIC_CONFIG_PATH, "sha256": manifest["source_files"][DIAGNOSTIC_CONFIG_PATH]},
        },
        "experiment": manifest["experiment"],
        "evidence_policy": {
            "diagnostic_only": True,
            "authorization_claimed": False,
            "gate_claimed": False,
            "production_entry": "same_process_runner_generate_only",
            "external_production_package_permitted": False,
            "formal_result": False,
            "stage_progression_allowed": False,
        },
        "command_schema": {
            "runner": RUNNER_PATH,
            "required_arguments": ["--generate", "--diagnostic-bootstrap", "--output", "--source-commit"],
            "forbidden_arguments": ["--execution-package", "--synthetic-fixture", "--backend", "--device", "--receipt"],
        },
        "output_schema": OUTPUT_SCHEMA,
    }


def _write_diagnostic_bootstrap(tmp_path: Path, repo: Path) -> Path:
    path = tmp_path / "diagnostic-bootstrap.json"
    path.write_text(json.dumps(_diagnostic_bootstrap(repo), separators=(",", ":"), allow_nan=False) + "\n")
    return path


def _args(repo: Path, manifest: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--generate",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--source-commit",
        _git(repo, "rev-parse", "HEAD"),
        *extra,
    ]


def _diagnostic_args(repo: Path, bootstrap: Path, output: Path) -> list[str]:
    return [
        "--generate",
        "--diagnostic-bootstrap",
        str(bootstrap),
        "--output",
        str(output),
        "--source-commit",
        _git(repo, "rev-parse", "HEAD"),
    ]


class CPUVideoBackend:
    device = "cuda:0"
    cpu_only_test_harness = False
    provenance_class = "production_generation"
    loaded_identity = {"device": "cuda:0", "loaded_revision": "forged"}

    def __init__(self, cache: Path, mode: str = "pass") -> None:
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.mode = mode

    def prepare_initial_latent(self, parameters: dict[str, Any]) -> np.ndarray:
        return np.asarray([[parameters["seed"], 1.0], [2.0, 3.0]], dtype=np.float32)

    def _frames(self, parameters: dict[str, Any], condition: str) -> np.ndarray:
        glass_group = "glass" in parameters["prompt"]
        base = 128
        frames = np.full((49, 320, 512, 3), base, dtype=np.uint8)
        vertical = np.cos(np.pi * (2.0 * np.arange(320) + 1.0) / (2.0 * 320))[:, None]
        horizontal = np.cos(np.pi * (2.0 * np.arange(512) + 1.0) / (2.0 * 512))[None, :]
        feature_channel = 0 if glass_group else 2
        effective = condition
        if self.mode == "p_fail" and condition in {"A", "B"}:
            effective = "OFF_R1"
        if self.mode == "r_fail" and condition in {"A", "B"}:
            fixed = np.clip(np.rint(base + 45.0 * vertical), 0, 255).astype(np.uint8)
            frames[..., feature_channel] = fixed
            return frames
        if self.mode == "noise" and condition == "OFF_R2":
            frames[..., 0] = np.uint8(base + 12)
            return frames
        if effective in {"A", "B"}:
            schedule = schedule_a() if effective == "A" else schedule_b()
            for row, frame_group in enumerate(FRAME_GROUPS):
                x, y = schedule[row]
                spatial = np.clip(np.rint(base + 45.0 * (x * vertical + y * horizontal)), 0, 255).astype(np.uint8)
                for frame_index in frame_group:
                    frames[frame_index, ..., feature_channel] = spatial
        return frames

    def _template(self, parameters: dict[str, Any], condition: str) -> Path:
        key = f"{'glass' if 'glass' in parameters['prompt'] else 'paper'}-{self.mode}-{condition}.mp4"
        path = self.cache / key
        if path.exists():
            return path
        import imageio.v3 as iio

        iio.imwrite(
            path,
            self._frames(parameters, condition),
            plugin="FFMPEG",
            fps=8,
            codec="libx264",
            pixelformat="yuv420p",
            quality=5,
            macro_block_size=16,
        )
        return path

    def generate_condition(
        self,
        parameters: dict[str, Any],
        latent: np.ndarray,
        condition: str,
        schedule: Any,
        video_path: Path,
    ) -> dict[str, Any]:
        shutil.copyfile(self._template(parameters, condition), video_path)
        primitives: list[dict[str, Any]] = []
        if condition in {"A", "B"}:
            for index in range(16):
                primitives.append(
                    {
                        "call_index": index,
                        "output_shape": [1, 4, 8],
                        "output_dtype": "float32",
                        "input_tensor_sha256": hashlib.sha256(f"input:{condition}:{index}".encode()).hexdigest(),
                        "modified_tensor_sha256": hashlib.sha256(f"modified:{condition}:{index}".encode()).hexdigest(),
                        "distinct_storage": True,
                        "effective_relative_rms": 0.03,
                    }
                )
        return {"carrier_event_primitives": primitives}


class FailingBackend(CPUVideoBackend):
    def __init__(self, cache: Path, error: BaseException) -> None:
        super().__init__(cache)
        self.error = error

    def generate_condition(self, *arguments: Any, **keywords: Any) -> dict[str, Any]:
        raise self.error


class BadVideoBackend(CPUVideoBackend):
    def __init__(self, cache: Path, defect: str) -> None:
        super().__init__(cache)
        self.defect = defect

    def generate_condition(
        self,
        parameters: dict[str, Any],
        latent: np.ndarray,
        condition: str,
        schedule: Any,
        video_path: Path,
    ) -> dict[str, Any]:
        if self.defect == "plaintext":
            video_path.write_text("not an mp4")
        else:
            import imageio.v3 as iio

            frames = self._frames(parameters, condition)
            kwargs: dict[str, Any] = {"fps": 8, "codec": "libx264", "pixelformat": "yuv420p", "macro_block_size": 16}
            if self.defect == "fps":
                kwargs["fps"] = 7
            elif self.defect == "frames":
                frames = frames[:48]
            elif self.defect == "geometry":
                frames = frames[:, :256]
            elif self.defect == "codec":
                kwargs["codec"] = "mpeg4"
            iio.imwrite(video_path, frames, plugin="FFMPEG", **kwargs)
        return {"carrier_event_primitives": []}


class MaliciousEventBackend(CPUVideoBackend):
    def __init__(self, cache: Path, defect: str) -> None:
        super().__init__(cache)
        self.defect = defect

    def generate_condition(self, *arguments: Any, **keywords: Any) -> dict[str, Any]:
        result = super().generate_condition(*arguments, **keywords)
        condition = arguments[2]
        events = result["carrier_event_primitives"]
        if self.defect == "off_event" and condition == "OFF_R1":
            result["carrier_event_primitives"] = [{"unexpected": True}]
        elif condition == "A":
            if self.defect == "count":
                events.pop()
            elif self.defect == "rms":
                events[0]["effective_relative_rms"] = 0.04
            elif self.defect == "call_index":
                events[0]["call_index"] = 2
            elif self.defect == "extra":
                events[0]["provenance_class"] = "production_generation"
            elif self.defect == "expanded_result":
                result["snapshot"] = {"provenance_class": "production_generation"}
        return result


class AliasedLatentBackend(CPUVideoBackend):
    class AliasedArray(np.ndarray):
        def copy(self, *arguments: Any, **keywords: Any) -> Any:
            return self

    def prepare_initial_latent(self, parameters: dict[str, Any]) -> np.ndarray:
        base = super().prepare_initial_latent(parameters)
        return base.view(self.AliasedArray)


def _preflight_fixture(tmp_path: Path) -> tuple[Path, Path, Any]:
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    checked = preflight(repo, manifest, declared_source_commit=_git(repo, "rev-parse", "HEAD"))
    return repo, manifest, checked


def test_manifest_exact_schema_and_injection_matrix(tmp_path: Path) -> None:
    repo = _make_clean_repo(tmp_path)
    valid = _manifest(repo)
    validate_manifest_schema(valid)
    mutations = []
    item = copy.deepcopy(valid); item["extra"] = None; mutations.append(item)
    item = copy.deepcopy(valid); item.pop("experiment"); mutations.append(item)
    item = copy.deepcopy(valid); item["schema_version"] = True; mutations.append(item)
    item = copy.deepcopy(valid); item["experiment"]["attempt_budget"] = 9; mutations.append(item)
    item = copy.deepcopy(valid); item["experiment"]["retry_policy"] = "retry_allowed"; mutations.append(item)
    item = copy.deepcopy(valid); item["experiment"]["condition_order"].append("C"); mutations.append(item)
    item = copy.deepcopy(valid); item["evidence_policy"]["formal_result"] = 0; mutations.append(item)
    item = copy.deepcopy(valid); item["frozen_identities"]["config"]["threshold"] = 0.0; mutations.append(item)
    item = copy.deepcopy(valid); item["frozen_identities"]["extractor"]["raw_sha256"] = "0" * 64; mutations.append(item)
    item = copy.deepcopy(valid); item["source_files"] = dict(reversed(list(item["source_files"].items()))); mutations.append(item)
    for mutation in mutations:
        with pytest.raises(InvalidExperiment):
            validate_manifest_schema(mutation)


def test_diagnostic_bootstrap_and_two_step_decision_mapping(tmp_path: Path) -> None:
    repo = _make_clean_repo(tmp_path)
    bootstrap = _diagnostic_bootstrap(repo)
    validate_diagnostic_bootstrap_schema(bootstrap)
    bootstrap_path = _write_diagnostic_bootstrap(tmp_path, repo)
    checked = preflight(
        repo,
        bootstrap_path,
        declared_source_commit=_git(repo, "rev-parse", "HEAD"),
        diagnostic_only=True,
    )
    record = {"evidence_mode": "production_saved_mp4", "groups": [{"attempts": []}]}
    cases = (
        (
            Evaluation(PACKAGE_STATUS_P_FAIL, STATUS_P_FAIL, "P_FAIL", (), (), True),
            DIAGNOSTIC_NOT_FEASIBLE,
            DIAGNOSTIC_INSUFFICIENT_TO_DECIDE,
            DIAGNOSTIC_DECISION_CARRIER_NOT_FEASIBLE,
        ),
        (
            Evaluation(PACKAGE_STATUS_P_PASS_R_FAIL, STATUS_P_PASS_R_FAIL, "R_FAIL", (), ({"cell_pass": False},), True),
            DIAGNOSTIC_FEASIBLE,
            DIAGNOSTIC_INSUFFICIENT_TO_DECIDE,
            DIAGNOSTIC_DECISION_INSUFFICIENT,
        ),
        (
            Evaluation(PACKAGE_STATUS_P_PASS_R_PASS, STATUS_P_PASS_R_PASS, "R_PASS", (), ({"cell_pass": True},), True),
            DIAGNOSTIC_FEASIBLE,
            DIAGNOSTIC_FEASIBLE,
            DIAGNOSTIC_DECISION_BUILD_BLIND,
        ),
        (
            Evaluation(PACKAGE_STATUS_INSUFFICIENT, STATUS_INSUFFICIENT, "NOISE", (), (), True),
            DIAGNOSTIC_INSUFFICIENT_TO_DECIDE,
            DIAGNOSTIC_INSUFFICIENT_TO_DECIDE,
            DIAGNOSTIC_DECISION_INSUFFICIENT,
        ),
    )
    for evaluation, step1, step2, route in cases:
        audit = evaluation_audit(checked, record, "0" * 64, evaluation, cpu_only_test_harness=False)
        assert audit["diagnostic_class"] == DIAGNOSTIC_CLASS
        assert audit["step1_carrier_survival"] == step1
        assert audit["step2_relation_readout"] == step2
        assert audit["route_decision"] == route
        assert audit["formal_result"] is False and audit["stage_progression_allowed"] is False
        assert "manifest_sha256" not in audit and audit["diagnostic_bootstrap_sha256"] == sha256_file(bootstrap_path)


def test_diagnostic_runner_bootstrap_reaches_generation_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _make_clean_repo(tmp_path)
    bootstrap = _write_diagnostic_bootstrap(tmp_path, repo)
    output = tmp_path / "diagnostic-output"
    code = RUNNER.main(
        _diagnostic_args(repo, bootstrap, output),
        _repo_root=repo,
        _test_backend=FailingBackend(tmp_path / "cache", RuntimeError("diagnostic smoke stop")),
    )
    assert code == 2
    response = json.loads(capsys.readouterr().out.strip())
    audit = json.loads((Path(response["actual_package_path"]) / "audit.json").read_text())
    assert audit["status"] == STATUS_INVALID
    assert audit["diagnostic_class"] == DIAGNOSTIC_CLASS
    assert audit["step1_carrier_survival"] == DIAGNOSTIC_INSUFFICIENT_TO_DECIDE
    assert audit["step2_relation_readout"] == DIAGNOSTIC_INSUFFICIENT_TO_DECIDE
    assert audit["route_decision"] == DIAGNOSTIC_DECISION_INSUFFICIENT
    assert audit["formal_result"] is False and audit["stage_progression_allowed"] is False


def test_diagnostic_cpu_smoke_exact8_saved_mp4_package(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _make_clean_repo(tmp_path)
    bootstrap = _write_diagnostic_bootstrap(tmp_path, repo)
    output = tmp_path / "diagnostic-output"
    code = RUNNER.main(
        _diagnostic_args(repo, bootstrap, output),
        _repo_root=repo,
        _test_backend=CPUVideoBackend(tmp_path / "cache"),
    )
    assert code == 0
    response = json.loads(capsys.readouterr().out.strip())
    package = Path(response["actual_package_path"])
    audit = json.loads((package / "audit.json").read_text())
    assert audit["diagnostic_class"] == DIAGNOSTIC_CLASS
    assert audit["step1_carrier_survival"] == DIAGNOSTIC_FEASIBLE
    assert audit["step2_relation_readout"] == DIAGNOSTIC_FEASIBLE
    assert audit["route_decision"] == DIAGNOSTIC_DECISION_BUILD_BLIND
    assert len(audit["attempt_log"]) == 8
    assert (package / "diagnostic_bootstrap.json").is_file()
    assert not (package / "authorization_manifest.json").exists()


def test_preflight_binds_clean_source_and_all_frozen_identities(tmp_path: Path) -> None:
    repo, manifest, checked = _preflight_fixture(tmp_path)
    assert checked.source_state.dirty is False
    assert checked.source_state.head == _git(repo, "rev-parse", "HEAD")
    assert checked.source_hashes[CONFIG_PATH] == CONFIG_RAW_SHA256
    assert checked.source_hashes[PLAN_PATH] == PLAN_RAW_SHA256
    assert sha256_file(repo / PROTOCOL_PATH) == PROTOCOL_RAW_SHA256
    assert manifest.is_file()


def test_preflight_rejects_dirty_hash_head_tree_and_fake_declaration(tmp_path: Path) -> None:
    for mutation in ("dirty", "hash", "head", "tree", "declaration"):
        case = tmp_path / mutation
        case.mkdir()
        repo = _make_clean_repo(case)
        manifest_data = _manifest(repo)
        declaration = manifest_data["expected_source"]["head"]
        if mutation == "dirty":
            (repo / "untracked").write_text("dirty")
        elif mutation == "hash":
            manifest_data["source_files"][REQUIRED_SOURCE_PATHS[0]] = "0" * 64
        elif mutation == "head":
            manifest_data["expected_source"]["head"] = "0" * 40
        elif mutation == "tree":
            manifest_data["expected_source"]["tree"] = "0" * 40
        else:
            declaration = "0" * 40
        manifest = case / "manifest.json"
        manifest.write_text(json.dumps(manifest_data, separators=(",", ":"), allow_nan=False))
        with pytest.raises(InvalidExperiment):
            preflight(repo, manifest, declared_source_commit=declaration)


@pytest.mark.parametrize("relative_path", [CONFIG_PATH, PLAN_PATH, NOTEBOOK_PATH, "src/sc_sstw_feasibility/learned_observation.py"])
def test_preflight_rejects_hidden_source_tamper(tmp_path: Path, relative_path: str) -> None:
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    subprocess.run(("git", "update-index", "--assume-unchanged", relative_path), cwd=repo, check=True)
    with (repo / relative_path).open("ab") as handle:
        handle.write(b"\n")
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=all") == ""
    with pytest.raises(InvalidExperiment):
        preflight(repo, manifest, declared_source_commit=_git(repo, "rev-parse", "HEAD"))


def test_external_production_is_rejected_before_manifest_or_execution_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "external-reject"
    forbidden_manifest = tmp_path / "must-not-read-manifest.json"
    forbidden_execution = tmp_path / "must-not-read-execution"
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path in {forbidden_manifest, forbidden_execution}:
            raise AssertionError("forbidden path was read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    code = RUNNER.main(
        [
            "--generate",
            "--manifest",
            str(forbidden_manifest),
            "--output",
            str(output),
            "--source-commit",
            "0" * 40,
            "--execution-package",
            str(forbidden_execution),
        ]
    )
    assert code == 2
    audit = json.loads((output / "audit.json").read_text())
    assert audit["reason_code"] == "EXTERNAL_PRODUCTION_PACKAGE_FORBIDDEN"
    assert audit["science_metrics_present"] is False
    assert audit["cpu_only_test_harness"] is False and audit["test_only_non_evidence"] is False


def test_cpu_harness_real_runner_decode_recompute_and_level_p_r(tmp_path: Path) -> None:
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    output = tmp_path / "output"
    backend = CPUVideoBackend(tmp_path / "video-cache")
    code = RUNNER.main(_args(repo, manifest, output), _repo_root=repo, _test_backend=backend)
    assert code == 0
    audit = json.loads((output / "audit.json").read_text())
    assert audit["status"] == PACKAGE_STATUS_P_PASS_R_PASS
    assert audit["protocol_outcome"] == STATUS_P_PASS_R_PASS
    assert audit["evidence_mode"] == EVIDENCE_CPU_HARNESS
    assert audit["cpu_only_test_harness"] is True
    assert audit["test_only_non_evidence"] is True
    assert audit["formal_result"] is False and audit["stage_progression_allowed"] is False
    assert len(audit["level_p"]) == 4 and len(audit["level_r"]) == 4
    assert len(json.loads((output / "generation/execution.json").read_text())["groups"]) == 2
    declared = RUNNER._parse_checksums((output / "checksums.sha256").read_bytes())
    assert all(sha256_file(output / name) == digest for name, digest in declared.items())


@pytest.mark.parametrize(
    ("mode", "status", "protocol_outcome", "exit_code", "level_r_count"),
    [
        ("p_fail", PACKAGE_STATUS_P_FAIL, STATUS_P_FAIL, 3, 0),
        ("r_fail", PACKAGE_STATUS_P_PASS_R_FAIL, STATUS_P_PASS_R_FAIL, 3, 4),
        ("noise", PACKAGE_STATUS_INSUFFICIENT, STATUS_INSUFFICIENT, 4, 0),
    ],
)
def test_p_short_circuit_r_fail_and_noise_sufficiency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    status: str,
    protocol_outcome: str,
    exit_code: int,
    level_r_count: int,
) -> None:
    if mode in {"p_fail", "noise"}:
        import sc_sstw_feasibility.rc0_minimal_implementation as runner_implementation

        def forbidden_level_r(*arguments: Any, **keywords: Any) -> Any:
            raise AssertionError("Level R extractor/cache path was reached before all Level P cells passed")

        monkeypatch.setattr(runner_implementation, "_extract_and_verify_cache", forbidden_level_r)
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    output = tmp_path / "output"
    code = RUNNER.main(_args(repo, manifest, output), _repo_root=repo, _test_backend=CPUVideoBackend(tmp_path / "cache", mode=mode))
    assert code == exit_code
    audit = json.loads((output / "audit.json").read_text())
    assert (audit["status"], audit["protocol_outcome"]) == (status, protocol_outcome)
    assert len(audit["level_r"]) == level_r_count


def test_receipt_type_copy_pickle_replay_and_disk_tamper(tmp_path: Path) -> None:
    repo, _manifest_path, checked = _preflight_fixture(tmp_path)
    generation = tmp_path / "generation"
    outcome = run_rc0_generation(checked, generation, [RUNNER_PATH, "--generate"], _test_backend=CPUVideoBackend(tmp_path / "cache"))
    with pytest.raises(TypeError):
        GenerationReceipt()
    forged = object.__new__(GenerationReceipt)
    for value in ({}, json.loads("{}"), object(), forged):
        with pytest.raises(Exception):
            consume_generation_receipt(value, outcome.record_path)
    with pytest.raises(Exception):
        copy.copy(outcome.receipt)
    with pytest.raises(Exception):
        copy.deepcopy(outcome.receipt)
    with pytest.raises(Exception):
        pickle.dumps(outcome.receipt)
    with pytest.raises(Exception):
        dataclasses.replace(outcome.receipt)
    with pytest.raises(Exception):
        outcome.receipt._snapshot = b"tamper"  # type: ignore[misc]
    record, artifacts, mode = validate_execution_record(outcome.record_path, checked, outcome.receipt, allow_cpu_test_harness=True)
    assert mode == EVIDENCE_CPU_HARNESS and artifacts
    with pytest.raises(InvalidExperiment):
        validate_execution_record(outcome.record_path, checked, outcome.receipt, allow_cpu_test_harness=True)
    assert record["groups"][0]["clone_identities"][0]["distinct_storage"] is True


def test_receipt_disk_tamper_rejected_before_evaluation(tmp_path: Path) -> None:
    _repo, _manifest_path, checked = _preflight_fixture(tmp_path)
    generation = tmp_path / "generation"
    outcome = run_rc0_generation(checked, generation, [RUNNER_PATH, "--generate"], _test_backend=CPUVideoBackend(tmp_path / "cache"))
    record = json.loads(outcome.record_path.read_text())
    record["groups"][0]["attempts"][0]["retry_index"] = 1
    outcome.record_path.write_text(json.dumps(record))
    with pytest.raises(InvalidExperiment, match="receipt"):
        validate_execution_record(outcome.record_path, checked, outcome.receipt, allow_cpu_test_harness=True)


def test_cpu_backend_cannot_escalate_to_production(tmp_path: Path) -> None:
    _repo, _manifest_path, checked = _preflight_fixture(tmp_path)
    outcome = run_rc0_generation(checked, tmp_path / "generation", [RUNNER_PATH, "--generate"], _test_backend=CPUVideoBackend(tmp_path / "cache"))
    assert outcome.cpu_only_test_harness is True
    with pytest.raises(InvalidExperiment, match="provenance"):
        validate_execution_record(outcome.record_path, checked, outcome.receipt, allow_cpu_test_harness=False)


@pytest.mark.parametrize("defect", ["plaintext", "fps", "frames", "geometry", "codec"])
def test_bad_saved_mp4_is_rejected_before_receipt_and_science(tmp_path: Path, defect: str) -> None:
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    output = tmp_path / "output"
    code = RUNNER.main(_args(repo, manifest, output), _repo_root=repo, _test_backend=BadVideoBackend(tmp_path / "cache", defect))
    assert code == 2
    invalid_root = next(tmp_path.glob("output.invalid.*"))
    audit = json.loads((invalid_root / "audit.json").read_text())
    assert audit["status"] == STATUS_INVALID and audit["science_metrics_present"] is False
    assert not output.exists()


@pytest.mark.parametrize("defect", ["off_event", "count", "rms", "call_index", "extra", "expanded_result"])
def test_carrier_event_and_backend_snapshot_injection_fail_closed(tmp_path: Path, defect: str) -> None:
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    output = tmp_path / "output"
    code = RUNNER.main(_args(repo, manifest, output), _repo_root=repo, _test_backend=MaliciousEventBackend(tmp_path / "cache", defect))
    assert code == 2
    audit = json.loads((next(tmp_path.glob("output.invalid.*")) / "audit.json").read_text())
    assert audit["status"] == STATUS_INVALID and audit["science_metrics_present"] is False


def test_generation_owns_four_independent_clones(tmp_path: Path) -> None:
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    output = tmp_path / "output"
    code = RUNNER.main(_args(repo, manifest, output), _repo_root=repo, _test_backend=AliasedLatentBackend(tmp_path / "cache"))
    assert code == 2
    audit = json.loads((next(tmp_path.glob("output.invalid.*")) / "audit.json").read_text())
    assert audit["status"] == STATUS_INVALID and audit["science_metrics_present"] is False


@pytest.mark.parametrize("artifact_name", ["features", "environment", "config", "integrity", "video"])
def test_receipt_bound_artifact_tamper_is_rejected(tmp_path: Path, artifact_name: str) -> None:
    _repo, _manifest_path, checked = _preflight_fixture(tmp_path)
    outcome = run_rc0_generation(checked, tmp_path / "generation", [RUNNER_PATH, "--generate"], _test_backend=CPUVideoBackend(tmp_path / "cache"))
    record = json.loads(outcome.record_path.read_text())
    relative = record["groups"][0]["conditions"][0]["artifacts"][artifact_name]["path"]
    path = outcome.record_path.parent / relative
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(InvalidExperiment):
        validate_execution_record(outcome.record_path, checked, outcome.receipt, allow_cpu_test_harness=True)


def test_symlink_artifact_is_rejected(tmp_path: Path) -> None:
    _repo, _manifest_path, checked = _preflight_fixture(tmp_path)
    outcome = run_rc0_generation(checked, tmp_path / "generation", [RUNNER_PATH, "--generate"], _test_backend=CPUVideoBackend(tmp_path / "cache"))
    record = json.loads(outcome.record_path.read_text())
    identity = record["groups"][0]["conditions"][0]["artifacts"]["stdout"]
    path = outcome.record_path.parent / identity["path"]
    backup = tmp_path / "outside.txt"
    shutil.copyfile(path, backup)
    path.unlink()
    path.symlink_to(backup)
    with pytest.raises(InvalidExperiment):
        validate_execution_record(outcome.record_path, checked, outcome.receipt, allow_cpu_test_harness=True)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("ordinary failure"),
        RuntimeError("CUDA out of memory"),
        MemoryError("oom"),
        TypeError("bad type"),
        subprocess.CalledProcessError(1, ["ffmpeg"]),
    ],
)
def test_generic_generation_exceptions_form_minimal_invalid_package(tmp_path: Path, error: BaseException) -> None:
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    output = tmp_path / "output"
    code = RUNNER.main(_args(repo, manifest, output), _repo_root=repo, _test_backend=FailingBackend(tmp_path / "cache", error))
    assert code == 2
    actual = next(tmp_path.glob("output.invalid.*"))
    audit = json.loads((actual / "audit.json").read_text())
    assert audit["status"] == PACKAGE_STATUS_INVALID
    assert audit["protocol_outcome"] == STATUS_INVALID
    assert audit["science_metrics_present"] is False
    assert audit["cpu_only_test_harness"] is True and audit["test_only_non_evidence"] is True
    assert "level_p" not in audit and "level_r" not in audit


def test_preexisting_output_and_publish_race_are_no_clobber(tmp_path: Path) -> None:
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    for kind in ("file", "empty_dir", "nonempty_dir", "symlink"):
        output = tmp_path / kind
        if kind == "file":
            output.write_bytes(b"owner")
        elif kind == "empty_dir":
            output.mkdir()
        elif kind == "nonempty_dir":
            output.mkdir(); (output / "owner").write_bytes(b"owner")
        else:
            target = tmp_path / "symlink-target"; target.mkdir(exist_ok=True); output.symlink_to(target, target_is_directory=True)
        before = output.lstat()
        code = RUNNER.main(_args(repo, manifest, output), _repo_root=repo, _test_backend=CPUVideoBackend(tmp_path / f"cache-{kind}"))
        assert code == 2 and output.lstat().st_ino == before.st_ino
        assert next(tmp_path.glob(f"{kind}.invalid.*")).is_dir()

    output = tmp_path / "race-output"

    def race(point: str, _staging: Path | None, target: Path | None) -> None:
        if point == "main_publish_race" and target is not None:
            target.mkdir()
            (target / "owner").write_bytes(b"race-owner")

    code = RUNNER.main(_args(repo, manifest, output), _repo_root=repo, _test_backend=CPUVideoBackend(tmp_path / "race-cache"), _test_fault=race)
    assert code == 2
    assert (output / "owner").read_bytes() == b"race-owner"
    assert next(tmp_path.glob("race-output.invalid.*")).is_dir()


@pytest.mark.parametrize("fault_point", ["main_audit_write", "main_checksum_self_check", "main_publish"])
def test_main_package_writer_faults_never_publish_science_package(tmp_path: Path, fault_point: str) -> None:
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    output = tmp_path / "output"

    def fail(point: str, _staging: Path | None, _target: Path | None) -> None:
        if point == fault_point:
            raise OSError(f"injected {fault_point}")

    code = RUNNER.main(
        _args(repo, manifest, output),
        _repo_root=repo,
        _test_backend=CPUVideoBackend(tmp_path / "cache"),
        _test_fault=fail,
    )
    assert code == 2 and not output.exists()
    invalid_root = next(tmp_path.glob("output.invalid.*"))
    audit = json.loads((invalid_root / "audit.json").read_text())
    assert audit["status"] == STATUS_INVALID and audit["science_metrics_present"] is False
    assert not (invalid_root / "generation").exists()


@pytest.mark.parametrize(
    "fault_point",
    ["fallback_staging_mkdir", "fallback_audit_write", "fallback_command_write", "fallback_checksum_write", "fallback_publish"],
)
def test_fallback_fault_matrix_is_nonrecursive(tmp_path: Path, capsys: pytest.CaptureFixture[str], fault_point: str) -> None:
    output = tmp_path / "fallback-matrix"

    def fail(point: str, _staging: Path | None, _target: Path | None) -> None:
        if point == fault_point:
            raise OSError(f"injected {fault_point}")

    code = RUNNER.main(
        ["--generate", "--manifest", str(tmp_path / "unread"), "--output", str(output), "--source-commit", "0" * 40, "--execution-package", "sentinel"],
        _test_fault=fail,
    )
    captured = capsys.readouterr()
    assert code == 2 and "Traceback" not in captured.err and "Traceback" not in captured.out
    response = json.loads((captured.err or captured.out).strip())
    assert response == {
        "actual_package_path": None,
        "package_written": False,
        "protocol_outcome": STATUS_INVALID,
        "reason_code": "INVALID_PACKAGE_WRITE_FAILED",
        "status": STATUS_INVALID,
    }


def test_fallback_writer_failure_is_nonrecursive_single_response(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "fallback-fail"

    def fail(point: str, _staging: Path | None, _target: Path | None) -> None:
        if point == "fallback_audit_write":
            raise OSError("injected fallback failure")

    code = RUNNER.main(
        ["--generate", "--manifest", str(tmp_path / "unread"), "--output", str(output), "--source-commit", "0" * 40, "--execution-package", "sentinel"],
        _test_fault=fail,
    )
    captured = capsys.readouterr()
    assert code == 2 and "Traceback" not in captured.err and "Traceback" not in captured.out
    response = json.loads((captured.err or captured.out).strip())
    assert response["status"] == STATUS_INVALID and response["package_written"] is False


def test_interrupts_are_not_swallowed(tmp_path: Path) -> None:
    repo = _make_clean_repo(tmp_path)
    manifest = _write_manifest(tmp_path, repo)
    for interrupt in (KeyboardInterrupt(), SystemExit(9)):
        with pytest.raises(type(interrupt)):
            RUNNER.main(_args(repo, manifest, tmp_path / f"output-{type(interrupt).__name__}"), _repo_root=repo, _test_backend=FailingBackend(tmp_path / "cache", interrupt))


def test_status_alias_mapping_is_exact() -> None:
    assert PACKAGE_TO_PROTOCOL_OUTCOME == {
        PACKAGE_STATUS_P_FAIL: STATUS_P_FAIL,
        PACKAGE_STATUS_P_PASS_R_FAIL: STATUS_P_PASS_R_FAIL,
        PACKAGE_STATUS_P_PASS_R_PASS: STATUS_P_PASS_R_PASS,
        PACKAGE_STATUS_INVALID: STATUS_INVALID,
        PACKAGE_STATUS_INSUFFICIENT: STATUS_INSUFFICIENT,
    }
    for package_status, protocol_outcome in PACKAGE_TO_PROTOCOL_OUTCOME.items():
        validate_status_mapping(package_status, protocol_outcome)
    with pytest.raises(InvalidExperiment):
        validate_status_mapping(PACKAGE_STATUS_P_FAIL, STATUS_P_PASS_R_PASS)


def test_notebook_is_thin_exact_ref_generate_only_and_failure_safe() -> None:
    raw = (ROOT / NOTEBOOK_PATH).read_text()
    notebook = json.loads(raw)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    execution_source = "".join(notebook["cells"][2]["source"])
    assert "REPOSITORY_URL = 'https://github.com/RICHAAARC/SC-SSTW-Feasibility.git'" in source
    assert "AUTHORIZED_REF = '54ef62c22489796f0467c45cc07edfc338cc322f'" in source
    assert "DRIVE_OUTPUT_ROOT = '/content/drive/MyDrive/SC-SSTW-Feasibility'" in source
    assert "AUTHORIZE_EXECUTION = True" in source
    assert "AUTHORIZE_DRIVE_IO = True" in source
    assert "'--generate'" in source
    assert "'--diagnostic-bootstrap'" in source
    assert "MANIFEST_PATH" not in source
    argv_line = next(line for line in execution_source.splitlines() if line.strip().startswith("argv ="))
    assert "--execution-package" not in argv_line
    assert "--synthetic-fixture" not in argv_line
    assert "--backend" not in argv_line and "--device" not in argv_line and "--receipt" not in argv_line
    assert "git', 'checkout', '--detach', AUTHORIZED_REF" in source
    assert "audit = json.loads" in source
    assert "diagnostic_values = {'FEASIBLE', 'NOT_FEASIBLE', 'INSUFFICIENT_TO_DECIDE'}" in execution_source
    assert "audit.get('step1_carrier_survival') not in diagnostic_values" in execution_source
    assert "'route_decision': audit['route_decision']" in execution_source
    assert "'RC0_P_FAIL': 3" in source and "'RC0_P_PASS_R_FAIL': 3" in source
    assert "finally:" in execution_source and "testzip" in execution_source
    assert "make_archive" not in source and "copytree" not in source and "exist_ok=True" not in source
    assert "_zip_tree_exclusive(ARCHIVE_ROOT, ARCHIVE)" in execution_source
    assert "_copy_file_exclusive(ARCHIVE, DRIVE_ARCHIVE)" in execution_source
    assert "_copy_file_exclusive(SIDECAR, DRIVE_SIDECAR)" in execution_source
    assert "DRIVE_SIDECAR.read_bytes() != sidecar_payload" in execution_source
    assert "zipfile.ZipFile(DRIVE_ARCHIVE)" in execution_source
    assert "actual_package_path = _validated_actual_package" in execution_source
    assert "_copy_tree_exclusive(safe_actual, ARCHIVE_ROOT/'actual-package')" in execution_source
    assert "_copy_tree_exclusive(OUTPUT" not in execution_source
    assert "'runner_started': False" in execution_source and "'audit_loaded': False" in execution_source
    assert "LOCAL_OUTPUT_ROOT must remain the frozen local root" in execution_source
    assert "output_root.mkdir(mode=0o700, parents=False, exist_ok=False)" in execution_source
    assert "_require_real_directory(output_root, expected_output_root)" in execution_source
    assert "_require_absent([WORK, LOG, OUTPUT, ARCHIVE_ROOT, ARCHIVE, SIDECAR])" in execution_source
    assert "_require_absent([DRIVE_ARCHIVE, DRIVE_SIDECAR])" in execution_source
    assert "Path('/content/drive') not in drive_root.parents" in execution_source
    assert execution_source.count("completed = subprocess.run(argv") == 1
    assert execution_source.index("try:\n") < execution_source.index("RUN_ID = hashlib.sha256(('RC0-DIAG-FAST:' + AUTHORIZED_REF)")
    assert execution_source.index("RUN_ID = hashlib.sha256(('RC0-DIAG-FAST:' + AUTHORIZED_REF)") < execution_source.index("git', 'clone'")
    assert execution_source.index("git', 'clone'") < execution_source.index("snapshot_download(")
    assert "'diagnostic_class': DIAGNOSTIC_CLASS" in execution_source
    assert "'authorization_claimed': False" in execution_source
    assert "'gate_claimed': False" in execution_source
    assert "diagnostic_bootstrap_bytes" in execution_source
    assert "DIAGNOSTIC_BOOTSTRAP = LOG/'diagnostic_bootstrap.json'" in execution_source
    assert execution_source.index("dirty = subprocess.run") < execution_source.index("diagnostic_bootstrap = {")
    assert execution_source.index("diagnostic_bootstrap = {") < execution_source.index("snapshot_download(")
    assert "model_id = frozen_config['model']['id']" in execution_source
    assert "model_revision = frozen_config['model']['revision']" in execution_source
    assert "re.fullmatch(r'[0-9a-f]{40}', model_revision)" in execution_source
    assert "snapshot_download(repo_id=model_id, revision=model_revision, local_files_only=False)" in execution_source
    assert "snapshot_path.lstat()" in execution_source
    assert "stat.S_ISLNK(snapshot_metadata.st_mode)" in execution_source
    assert "snapshot_path.resolve(strict=True)" in execution_source
    assert "snapshot_path.name != model_revision or resolved_snapshot.name != model_revision" in execution_source
    assert "downloaded model snapshot commit identity mismatch" in execution_source
    assert "token=" not in execution_source and "revision='main'" not in execution_source and "revision=\"main\"" not in execution_source
    assert execution_source.index("snapshot_path.resolve(strict=True)") < execution_source.index("completed = subprocess.run(argv")
    assert execution_source.index("drive.mount('/content/drive')") < execution_source.index("completed = subprocess.run(argv")
    assert execution_source.index("audit = json.loads") < execution_source.index("expected_exit =")
    archive_call = execution_source.index("        _zip_tree_exclusive(ARCHIVE_ROOT, ARCHIVE)")
    archive_test = execution_source.index("        with zipfile.ZipFile(ARCHIVE) as handle:", archive_call)
    archive_hash = execution_source.index("        archive_sha = hashlib.sha256(archive_bytes).hexdigest()", archive_test)
    sidecar_write = execution_source.index("        _write_bytes_exclusive(SIDECAR, sidecar_payload)", archive_hash)
    assert archive_call < archive_test < archive_hash < sidecar_write
    generation_source = (ROOT / "src/sc_sstw_feasibility/rc0_generation.py").read_text()
    assert "local_files_only=True" in generation_source
    assert "torch.cuda.is_bf16_supported()" in source
    for forbidden in ("level_p_metrics", "level_r_metrics", "2/255", "0.25", "construct_internal_residual"):
        assert forbidden not in source


def test_notebook_no_clobber_helpers_and_actual_package_boundary(tmp_path: Path) -> None:
    notebook = json.loads((ROOT / NOTEBOOK_PATH).read_text())
    execution_source = "".join(notebook["cells"][2]["source"])
    tree = ast.parse(execution_source)
    helper_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"Path": Path, "os": os, "shutil": shutil, "stat": stat, "zipfile": zipfile}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), NOTEBOOK_PATH, "exec"), namespace)

    existing = tmp_path / "existing"
    existing.write_bytes(b"sentinel")
    namespace["_require_absent"]([tmp_path / "missing"])
    with pytest.raises(RuntimeError, match="preexisting"):
        namespace["_require_absent"]([existing])

    exclusive = tmp_path / "exclusive.bin"
    namespace["_write_bytes_exclusive"](exclusive, b"first")
    with pytest.raises(FileExistsError):
        namespace["_write_bytes_exclusive"](exclusive, b"second")
    assert exclusive.read_bytes() == b"first"

    copied = tmp_path / "copied.bin"
    namespace["_copy_file_exclusive"](exclusive, copied)
    with pytest.raises(FileExistsError):
        namespace["_copy_file_exclusive"](exclusive, copied)
    assert copied.read_bytes() == b"first"

    source_tree = tmp_path / "source-tree"
    source_tree.mkdir()
    (source_tree / "regular.json").write_text("{}\n")
    copied_tree = tmp_path / "copied-tree"
    namespace["_copy_tree_exclusive"](source_tree, copied_tree)
    assert (copied_tree / "regular.json").read_text() == "{}\n"
    with pytest.raises(FileExistsError):
        namespace["_copy_tree_exclusive"](source_tree, copied_tree)
    unsafe_tree = tmp_path / "unsafe-tree"
    unsafe_tree.mkdir()
    (unsafe_tree / "escape").symlink_to(existing)
    with pytest.raises(RuntimeError, match="unsafe"):
        namespace["_copy_tree_exclusive"](unsafe_tree, tmp_path / "unsafe-copy")

    archive_root = tmp_path / "archive-root"
    archive_root.mkdir()
    (archive_root / "evidence.json").write_text("{}\n")
    archive = tmp_path / "evidence.zip"
    namespace["_zip_tree_exclusive"](archive_root, archive)
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
    with pytest.raises(FileExistsError):
        namespace["_zip_tree_exclusive"](archive_root, archive)

    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    output = output_parent / "run"
    output.mkdir()
    invalid = output_parent / "run.invalid.0001"
    invalid.mkdir()
    assert namespace["_validated_actual_package"](str(output), output) == output
    assert namespace["_validated_actual_package"](str(invalid), output) == invalid
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    with pytest.raises(RuntimeError, match="escaped"):
        namespace["_validated_actual_package"](str(escaped), output)
    linked = output_parent / "run.invalid.0002"
    linked.symlink_to(escaped, target_is_directory=True)
    with pytest.raises(RuntimeError, match="real directory"):
        namespace["_validated_actual_package"](str(linked), output)


def test_diag_fast_exact8_and_frozen_carrier_sanity() -> None:
    config = json.loads((ROOT / CONFIG_PATH).read_text())
    plan = json.loads((ROOT / PLAN_PATH).read_text())
    diagnostic = json.loads((ROOT / DIAGNOSTIC_CONFIG_PATH).read_text())
    assert plan["group_order"] == ["orbital_glass", "articulated_paper"]
    assert plan["condition_order"] == ["OFF_R1", "OFF_R2", "A", "B"]
    assert [(item["attempt_index"], item["group_id"], item["condition"]) for item in plan["attempts"]] == [
        (1, "orbital_glass", "OFF_R1"),
        (2, "orbital_glass", "OFF_R2"),
        (3, "orbital_glass", "A"),
        (4, "orbital_glass", "B"),
        (5, "articulated_paper", "OFF_R1"),
        (6, "articulated_paper", "OFF_R2"),
        (7, "articulated_paper", "A"),
        (8, "articulated_paper", "B"),
    ]
    assert plan["attempt_budget"] == 8 and plan["retry_policy"] == "no_retry_no_replacement_no_additional_attempts"
    assert config["carrier"] == {
        "kind": "dit_internal_self_attention_output_residual",
        "module_path": "transformer.blocks[29].attn1",
        "block_index": 29,
        "target_relative_rms": 0.03,
        "target_relative_rms_absolute_tolerance": 0.00005,
        "apply_to": "8_steps_each_cond_then_uncond_16_calls",
    }
    assert config["model"] == {"id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "revision": "0fad780a534b6463e45facd96134c9f345acfa5b"}
    assert config["encoding"]["codec"] == "h264" and config["encoding"]["pixel_format"] == "yuv420p"
    assert diagnostic["diagnostic_questions"] == {
        "step1_carrier_survival": [DIAGNOSTIC_FEASIBLE, DIAGNOSTIC_NOT_FEASIBLE, DIAGNOSTIC_INSUFFICIENT_TO_DECIDE],
        "step2_relation_readout": [DIAGNOSTIC_FEASIBLE, DIAGNOSTIC_NOT_FEASIBLE, DIAGNOSTIC_INSUFFICIENT_TO_DECIDE],
    }
    assert diagnostic["observations"]["vae_reencode_relation"]["available"] is False
    generation_source = (ROOT / "src/sc_sstw_feasibility/rc0_generation.py").read_text()
    assert "clones = [_clone(initial) for _condition in CONDITION_ORDER]" in generation_source
    assert "len(set(clone_pointers)) != 4" in generation_source
    assert "schedule = None if condition.startswith(\"OFF_\")" in generation_source
    assert "pipe.transformer.blocks[int(carrier[\"block_index\"])].attn1" in generation_source
    assert "retry_index\": 0" in generation_source


def test_receipt_module_has_no_ordinary_issuer_or_snapshot_constructor() -> None:
    import src.sc_sstw_feasibility.rc0_generation as generation

    names = set(dir(generation))
    assert not any(token in name.lower() for name in names for token in ("issue", "authority", "from_snapshot", "from_mapping", "binder", "factory"))
    assert "_close_receipt_authority" not in names and "_run_generation_control_flow" not in names


def test_frozen_science_and_old_g0_rc1_guards_are_unchanged() -> None:
    assert sha256_file(ROOT / CONFIG_PATH) == CONFIG_RAW_SHA256
    assert sha256_file(ROOT / PLAN_PATH) == PLAN_RAW_SHA256
    assert sha256_file(ROOT / PROTOCOL_PATH) == PROTOCOL_RAW_SHA256
    assert sha256_file(ROOT / "configs/rc1_method_validation.json") == "8accea693798e2dd2ad4451ea14df71651116b89e3d8707cfe344886a57bcf15"
    assert sha256_file(ROOT / "plans/rc1_matched_triplets.json") == "d5396107f00e68eba8f972a08996608421f02a4c205339a34b3616636e41a8c7"
    assert sha256_file(ROOT / "protocols/rc1_method_validation.md") == "ccac64d53c1a9b87bfdab8316e963ed5bdfe415e5d95fdc553e3ae8f09438b2a"
    assert sha256_file(ROOT / "configs/learned_observation_l1_v2_development.json") == "4584ed9d016ddf98c9a01408622b284b6e87724f33f1452b7cc7412ee1b88007"
