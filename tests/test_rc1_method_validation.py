from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import pickle
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from src.sc_sstw_feasibility.learned_observation import decode_saved_mp4, extract_feature_matrix
from src.sc_sstw_feasibility.rc1_gpu_generation import (
    GenerationReceipt,
    run_gpu_generation,
)

from src.sc_sstw_feasibility.learned_observation_l1_v2 import (
    ABSOLUTE_THRESHOLDS,
    CONFIG_PATH as G0_CONFIG_PATH,
    OUTPUT_SCHEMA as G0_OUTPUT_SCHEMA,
    PERMITTED_INPUT_IDS,
    PROTOCOL_ID as G0_PROTOCOL_ID,
    REQUIRED_SOURCE_PATHS as G0_REQUIRED_SOURCE_PATHS,
    SUCCESS_STATUS as G0_SUCCESS_STATUS,
    transform,
)
from src.sc_sstw_feasibility.rc1_method_validation import (
    ARTIFACT_NAMES,
    CONFIG_PATH,
    CONFIG_RAW_SHA256,
    CONDITIONS,
    EVIDENCE_CPU_HARNESS,
    EVIDENCE_PRODUCTION,
    EVIDENCE_SYNTHETIC,
    EXECUTION_SCHEMA,
    FEATURE_CACHE_SCHEMA,
    FORBIDDEN_CLAIM_TOKENS,
    GPU_LIBRARY_PATH,
    MANIFEST_SCHEMA,
    NOTEBOOK_PATH,
    OUTPUT_SCHEMA,
    PASS_CONCLUSION,
    PLAN_PATH,
    PLAN_RAW_SHA256,
    PRODUCTION_EVIDENCE_SCHEMA,
    PROTOCOL_ID,
    PROTOCOL_PATH,
    REQUIRED_SOURCE_PATHS,
    RUNNER_PATH,
    START_INDICES,
    STATUS_FAIL,
    STATUS_INVALID,
    STATUS_PASS,
    STATUS_PREREQUISITE,
    SYNTHETIC_EVIDENCE_SCHEMA,
    TEMPLATES,
    FrozenPrerequisite,
    InvalidExperiment,
    PrerequisiteNotMet,
    _inspect_and_decode_saved_mp4,
    canonical_json_bytes,
    command_artifact_payload,
    condition_config_payload,
    evaluate_execution,
    expected_matched_parameters,
    frontend_definition,
    integrity_artifact_payload,
    preflight,
    schedule_a,
    schedule_b,
    schedule_preflight,
    sha256_bytes,
    sha256_file,
    synthetic_environment_payload,
    validate_execution_record,
    validate_frozen_config_bytes,
    validate_frozen_plan_bytes,
    validate_manifest_schema,
    validate_prerequisite_package,
)
from src.sc_sstw_feasibility.rc1_prerequisite import G0_SUCCESS_PACKAGE_FILES, build_prerequisite_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]
HEX64 = "a" * 64
HEX40 = "b" * 40


def _write_json(path: Path, value: object, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2 if pretty else None, sort_keys=False, allow_nan=False).encode("utf-8") + b"\n"
    path.write_bytes(payload)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _raw_features(schedule: tuple[tuple[float, float], ...]) -> np.ndarray:
    features = np.tile(np.linspace(-0.03, 0.03, 30), (13, 1))
    features[:, 0] = np.asarray(schedule)[:, 0]
    features[:, 1] = np.asarray(schedule)[:, 1]
    return features


def _synthetic_readout() -> np.ndarray:
    features = _raw_features(schedule_a())
    design = np.column_stack((transform(features, "A1"), np.ones(13)))
    return np.linalg.lstsq(design, np.asarray(schedule_a()), rcond=None)[0]


def _make_prerequisite(path: Path, *, status: str = G0_SUCCESS_STATUS, candidate: str = "A1") -> str:
    thresholds = dict(ABSOLUTE_THRESHOLDS)
    authorization_bytes = canonical_json_bytes({"synthetic": "authorization_identity"}) + b"\n"
    config_bytes = canonical_json_bytes({"synthetic": "g0_config_identity"}) + b"\n"
    audit = {
        "protocol_id": G0_PROTOCOL_ID,
        "output_schema": G0_OUTPUT_SCHEMA,
        "valid_experiment": True,
        "status": status,
        "formal_result": False,
        "stage_progression_allowed": False,
        "selected_candidate": candidate,
        "source_state": {"head": HEX40, "tree": "c" * 40, "dirty": False},
        "source_file_sha256": {name: HEX64 for name in G0_REQUIRED_SOURCE_PATHS},
        "config_identity": {"path": G0_CONFIG_PATH, "sha256": sha256_bytes(config_bytes)},
        "manifest_identity": {"sha256": sha256_bytes(authorization_bytes)},
        "input_sha256": {str(item): "e" * 64 for item in PERMITTED_INPUT_IDS},
        "candidates": [{"candidate": candidate, "development_gate_pass": True, "derived_envelope": thresholds}],
    }
    if candidate in {"A1", "A2"}:
        artifacts = build_prerequisite_artifacts(audit, _synthetic_readout())
        (path / "audit.json").parent.mkdir(parents=True, exist_ok=True)
        (path / "audit.json").write_bytes(artifacts.audit_bytes)
        (path / "frozen_frontend.json").write_bytes(artifacts.frontend_bytes)
        (path / "readout.json").write_bytes(artifacts.readout_bytes)
    else:
        _write_json(path / "audit.json", audit)
        _write_json(path / "frozen_frontend.json", {"forged": True})
        _write_json(path / "readout.json", {"forged": True})
    (path / "authorization_manifest.json").write_bytes(authorization_bytes)
    (path / "config.json").write_bytes(config_bytes)
    (path / "command.txt").write_text(f"python {G0_REQUIRED_SOURCE_PATHS[1]} --manifest synthetic --output synthetic\n", encoding="utf-8")
    checksums = "".join(f"{sha256_file(path / name)}  {name}\n" for name in G0_SUCCESS_PACKAGE_FILES)
    (path / "checksums.sha256").write_text(checksums, encoding="utf-8")
    return sha256_file(path / "checksums.sha256")


def _make_repo(path: Path) -> tuple[str, str]:
    for relative in set(REQUIRED_SOURCE_PATHS).union(G0_REQUIRED_SOURCE_PATHS):
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    init = path / "src/sc_sstw_feasibility/__init__.py"
    init.write_text("", encoding="utf-8")
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "synthetic@example.invalid")
    _git(path, "config", "user.name", "Synthetic Fixture")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "synthetic RC1 source")
    return _git(path, "rev-parse", "HEAD"), _git(path, "rev-parse", "HEAD^{tree}")


def _run_real_g0_success(repo: Path, root: Path, head: str, tree: str) -> Path:
    inputs = root / "g0-synthetic-inputs"
    inputs.mkdir()
    base = _raw_features(schedule_a())
    rng = np.random.default_rng(17)
    input_paths: dict[int, Path] = {}
    for dataset_id in PERMITTED_INPUT_IDS:
        features = base.copy()
        if dataset_id in PERMITTED_INPUT_IDS[:4]:
            features[:, :2] += 1e-3 * rng.normal(size=(13, 2))
        path = inputs / f"synthetic-{dataset_id}.json"
        path.write_bytes(canonical_json_bytes({"fixture": "pure_synthetic", "features": features.tolist()}))
        input_paths[dataset_id] = path
    source_files = {relative: sha256_file(repo / relative) for relative in G0_REQUIRED_SOURCE_PATHS}
    manifest = {
        "schema_version": 1,
        "protocol_id": G0_PROTOCOL_ID,
        "expected_source": {"head": head, "tree": tree},
        "source_files": source_files,
        "config": {"path": G0_CONFIG_PATH, "sha256": source_files[G0_CONFIG_PATH]},
        "inputs": [{"dataset_id": dataset_id, "path": str(input_paths[dataset_id]), "sha256": sha256_file(input_paths[dataset_id])} for dataset_id in PERMITTED_INPUT_IDS],
        "command_schema": {"runner": G0_REQUIRED_SOURCE_PATHS[1], "required_arguments": ["--manifest", "--output"], "optional_arguments": ["--source-commit"]},
        "output_schema": G0_OUTPUT_SCHEMA,
    }
    manifest_path = root / "g0-synthetic-authorization.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    output = root / "g0-output"
    completed = subprocess.run(
        [sys.executable, "-B", str(repo / G0_REQUIRED_SOURCE_PATHS[1]), "--manifest", str(manifest_path), "--output", str(output), "--source-commit", head],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    audit = json.loads((output / "audit.json").read_text())
    assert audit["status"] == G0_SUCCESS_STATUS and audit["selected_candidate"] == "A1"
    assert set(path.name for path in output.iterdir()) == set(G0_SUCCESS_PACKAGE_FILES).union({"checksums.sha256"})
    return output


def _make_manifest(path: Path, repo: Path, prerequisite: Path, checksums_sha: str, *, head: str, tree: str) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "manifest_schema": MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "expected_source": {"head": head, "tree": tree},
        "source_files": {relative: sha256_file(repo / relative) for relative in REQUIRED_SOURCE_PATHS},
        "config": {"path": CONFIG_PATH, "sha256": CONFIG_RAW_SHA256},
        "plan": {"path": PLAN_PATH, "sha256": PLAN_RAW_SHA256},
        "prerequisite": {"package_path": str(prerequisite), "checksums_sha256": checksums_sha},
        "evidence_policy": {"default_mode": EVIDENCE_PRODUCTION, "synthetic_fixture_permitted": True},
        "command_schema": {
            "runner": RUNNER_PATH,
            "required_arguments": ["--manifest", "--output"],
            "optional_arguments": ["--source-commit", "--execution-package", "--generate", "--synthetic-fixture"],
        },
        "output_schema": OUTPUT_SCHEMA,
    }
    _write_json(path, manifest, pretty=True)
    return manifest


def _artifact_features(condition: str, group_index: int) -> np.ndarray:
    if condition == "A":
        return _raw_features(schedule_a())
    if condition == "B":
        return _raw_features(schedule_b())
    return np.random.default_rng(700 + group_index).normal(size=(13, 30))


def _make_execution(path: Path, frozen: object) -> Path:
    groups = []
    for group_index, plan_group in enumerate(frozen.plan["groups"]):
        parameters = expected_matched_parameters(frozen.config, plan_group)
        parameter_sha = sha256_bytes(canonical_json_bytes(parameters))
        latent_sha = hashlib.sha256(f"latent:{group_index}".encode()).hexdigest()
        conditions = []
        attempts = []
        for condition in CONDITIONS:
            condition_dir = path / plan_group["group_id"] / condition
            condition_dir.mkdir(parents=True)
            video = condition_dir / "synthetic_video.json"
            _write_json(video, {"schema_version": 1, "evidence_mode": EVIDENCE_SYNTHETIC, "artifact_kind": "synthetic_video_identity", "group_id": plan_group["group_id"], "condition": condition})
            feature_payload = {
                "schema_version": 1,
                "feature_cache_schema": FEATURE_CACHE_SCHEMA,
                "evidence_mode": EVIDENCE_SYNTHETIC,
                "source": "pure_synthetic_feature_fixture",
                "video_sha256": sha256_file(video),
                "extractor_identity": {"id": "test_only_synthetic_feature_matrix_v1"},
                "comparison": {"atol": 0.0, "rtol": 0.0},
                "features": _artifact_features(condition, group_index).tolist(),
            }
            _write_json(condition_dir / "features.json", feature_payload)
            (condition_dir / "stdout.txt").write_text("synthetic stdout\n", encoding="utf-8")
            (condition_dir / "stderr.txt").write_text("", encoding="utf-8")
            saved_video_path = str(video.relative_to(path))
            _write_json(condition_dir / "config.json", condition_config_payload(frozen, plan_group, condition, latent_sha, saved_video_path, EVIDENCE_SYNTHETIC))
            _write_json(condition_dir / "environment.json", synthetic_environment_payload(frozen.config))
            _write_json(condition_dir / "command.json", command_artifact_payload(frozen, EVIDENCE_SYNTHETIC))
            schedule = schedule_a() if condition == "A" else schedule_b()
            carrier_records = [] if condition == "OFF" else [
                {
                    "call_index": index,
                    "module_path": frozen.config["carrier"]["module_path"],
                    "schedule_sha256": sha256_bytes(canonical_json_bytes(schedule)),
                    "effective_relative_rms": frozen.config["carrier"]["target_relative_rms"],
                    "evidence_mode": EVIDENCE_SYNTHETIC,
                    "test_only_synthetic": True,
                }
                for index in range(16)
            ]
            _write_json(condition_dir / "integrity.json", integrity_artifact_payload(frozen, plan_group, condition, latent_sha, EVIDENCE_SYNTHETIC, carrier_records, None))
            files = {
                "video": video,
                "features": condition_dir / "features.json",
                "stdout": condition_dir / "stdout.txt",
                "stderr": condition_dir / "stderr.txt",
                "config": condition_dir / "config.json",
                "environment": condition_dir / "environment.json",
                "command": condition_dir / "command.json",
                "integrity": condition_dir / "integrity.json",
            }
            artifacts = {name: {"path": str(files[name].relative_to(path)), "sha256": sha256_file(files[name])} for name in ARTIFACT_NAMES}
            conditions.append({
                "condition": condition,
                "schedule_id": {"OFF": "NONE", "A": "A", "B": "B"}[condition],
                "carrier_enabled": condition != "OFF",
                "initial_latent_sha256": latent_sha,
                "matched_parameters_sha256": parameter_sha,
                "artifacts": artifacts,
            })
            attempts.append({"condition": condition, "attempt_index": 0, "outcome": "success", "matched_parameters_sha256": parameter_sha})
        groups.append({
            "group_id": plan_group["group_id"],
            "content_grammar": plan_group["content_grammar"],
            "prompt": plan_group["prompt"],
            "prompt_sha256": parameters["prompt_sha256"],
            "seed": plan_group["seed"],
            "matched_parameters": parameters,
            "conditions": conditions,
            "attempts": attempts,
        })
    record = {
        "schema_version": 1,
        "execution_schema": EXECUTION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": PLAN_RAW_SHA256,
        "prerequisite_identity": frozen.prerequisite.identity(),
        "evidence_mode": EVIDENCE_SYNTHETIC,
        "evidence_schema": SYNTHETIC_EVIDENCE_SCHEMA,
        "groups": groups,
    }
    record_path = path / "execution.json"
    _write_json(record_path, record, pretty=True)
    return record_path


def _write_runtime_mp4(path: Path, *, frame_count: int = 49, fps: int = 8, height: int = 320, width: int = 512, codec: str = "libx264") -> None:
    import imageio.v3 as iio

    frames = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    for index in range(frame_count):
        frames[index, :, :, 0] = index % 251
    iio.imwrite(path, frames, plugin="FFMPEG", fps=fps, codec=codec, pixelformat="yuv420p", quality=5, macro_block_size=16)


def _production_environment(config: dict[str, object]) -> dict[str, object]:
    scheduler = {"class": "UniPCMultistepScheduler", "config_sha256": "9" * 64, "bound_to_model_revision": config["model"]["revision"]}
    return {
        "schema_version": 1,
        "artifact_schema": "sc_sstw_rc1_execution_environment_v1",
        "evidence_mode": EVIDENCE_PRODUCTION,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "runtime": {
            "python": sys.version,
            "platform": "Linux-6.8-x86_64",
            "torch": "2.6.0+cu124",
            "diffusers": "0.35.2",
            "cuda": "12.4",
            "gpu": "NVIDIA A100-SXM4-40GB",
            "versions": {"imageio": "2.37.0", "imageio_ffmpeg": "0.6.0"},
        },
        "scheduler": scheduler,
        "sampler": dict(scheduler),
    }


class _CPUOnlyGenerationBackend:
    """Deterministic adapter that exercises production control flow without GPU."""

    def __init__(self, *, corrupt_video: bool = False, real_mp4: bool = False) -> None:
        self.corrupt_video = corrupt_video
        self.real_mp4 = real_mp4

    def load_environment(self, frozen: object) -> dict[str, object]:
        environment = _production_environment(frozen.config)
        environment["runtime"]["gpu"] = "CPU-only control-flow harness"
        environment["runtime"]["cuda"] = "CPU-harness-no-CUDA"
        return environment

    def prepare_initial_latents(self, parameters: dict[str, object]) -> np.ndarray:
        return np.random.default_rng(int(parameters["seed"])).normal(size=(1, 4, 2, 2)).astype(np.float32)

    def loaded_identity(self, frozen: object, environment: dict[str, object]) -> dict[str, object]:
        return {
            "model_id": frozen.config["model"]["id"],
            "loaded_revision": frozen.config["model"]["revision"],
            "scheduler": environment["scheduler"],
            "dtype": "torch.bfloat16",
            "device": "cpu-only-harness",
        }

    def latent_identity(self, latent: np.ndarray) -> dict[str, object]:
        envelope = canonical_json_bytes({"shape": list(latent.shape), "dtype": str(latent.dtype)}) + latent.tobytes()
        return {"sha256": sha256_bytes(envelope), "shape": list(latent.shape), "dtype": str(latent.dtype)}

    def generate_condition(self, frozen: object, parameters: dict[str, object], initial_latent: np.ndarray, condition: str, video_path: Path) -> dict[str, object]:
        identity = self.latent_identity(initial_latent)
        if self.corrupt_video and condition == "OFF":
            video_path.write_text("corrupt CPU harness MP4", encoding="utf-8")
            features = np.zeros((13, 30), dtype=np.float64).tolist()
            codec = {"container": "mp4", "codec_name": "h264", "pixel_format": "yuv420p", "width": 512, "height": 320, "fps": "8/1", "frame_count": 49, "decodable": True}
        elif self.real_mp4:
            _write_runtime_mp4(video_path)
            features = extract_feature_matrix(decode_saved_mp4(video_path))
            codec = {"container": "mp4", "codec_name": "h264", "pixel_format": "yuv420p", "width": 512, "height": 320, "fps": "8/1", "frame_count": 49, "decodable": True}
        else:
            video_path.write_bytes(b"CPU-only receipt-path sentinel")
            features = np.zeros((13, 30), dtype=np.float64).tolist()
            codec = {"container": "mp4", "codec_name": "h264", "pixel_format": "yuv420p", "width": 512, "height": 320, "fps": "8/1", "frame_count": 49, "decodable": True}
        schedule = schedule_a() if condition == "A" else schedule_b()
        schedule_sha = sha256_bytes(canonical_json_bytes(schedule))
        records = [] if condition == "OFF" else [
            {
                "call_index": index,
                "module_path": frozen.config["carrier"]["module_path"],
                "schedule_sha256": schedule_sha,
                "output_shape": [2, 21, 9600, 1536],
                "output_dtype": "torch.bfloat16",
                "input_tensor_sha256": sha256_bytes(f"input:{condition}:{index}".encode()),
                "modified_tensor_sha256": sha256_bytes(f"modified:{condition}:{index}".encode()),
                "distinct_storage": True,
                "effective_relative_rms": frozen.config["carrier"]["target_relative_rms"],
                "evidence_mode": EVIDENCE_PRODUCTION,
            }
            for index in range(16)
        ]
        return {
            "carrier_records": records,
            "features": features,
            "initial_latent_identity": identity,
            "condition_latent_identity": dict(identity),
            "codec_identity": codec,
            "video_saved_complete": True,
            "video_sha256": sha256_file(video_path),
        }


class _MasqueradingCPUBackend(_CPUOnlyGenerationBackend):
    cpu_only_test_harness = False
    provenance_class = "production_generation"
    complete_snapshot = {"completed": True, "provenance_class": "production_generation"}

    def load_environment(self, frozen: object) -> dict[str, object]:
        environment = _production_environment(frozen.config)
        environment["runtime"]["gpu"] = "malicious CUDA claim"
        return environment

    def loaded_identity(self, frozen: object, environment: dict[str, object]) -> dict[str, object]:
        return {
            "model_id": frozen.config["model"]["id"],
            "loaded_revision": frozen.config["model"]["revision"],
            "scheduler": environment["scheduler"],
            "dtype": "torch.bfloat16",
            "device": "cuda:0",
            "provenance_class": "production_generation",
        }


class _FailingCPUBackend(_CPUOnlyGenerationBackend):
    def generate_condition(self, frozen: object, parameters: dict[str, object], initial_latent: np.ndarray, condition: str, video_path: Path) -> dict[str, object]:
        raise RuntimeError("injected CPU generation failure before receipt issue")


def _make_receipted_production_execution(path: Path, frozen: object, *, corrupt_video: bool = False, real_mp4: bool = False):
    return run_gpu_generation(frozen, path, [RUNNER_PATH, "--generate"], _test_backend=_CPUOnlyGenerationBackend(corrupt_video=corrupt_video, real_mp4=real_mp4))


def _make_production_execution(path: Path, frozen: object) -> Path:
    template = path.parent / "runtime-generated-template.mp4"
    _write_runtime_mp4(template)
    cached_features = extract_feature_matrix(decode_saved_mp4(template))
    codec_identity = {"container": "mp4", "codec_name": "h264", "pixel_format": "yuv420p", "width": 512, "height": 320, "fps": "8/1", "frame_count": 49, "decodable": True}
    groups = []
    for group_index, plan_group in enumerate(frozen.plan["groups"]):
        parameters = expected_matched_parameters(frozen.config, plan_group)
        parameter_sha = sha256_bytes(canonical_json_bytes(parameters))
        latent_sha = hashlib.sha256(f"production-latent:{group_index}".encode()).hexdigest()
        conditions = []
        attempts = []
        for condition in CONDITIONS:
            condition_dir = path / plan_group["group_id"] / condition
            condition_dir.mkdir(parents=True)
            video = condition_dir / "saved.mp4"
            shutil.copy2(template, video)
            _write_json(condition_dir / "features.json", {
                "schema_version": 1,
                "feature_cache_schema": FEATURE_CACHE_SCHEMA,
                "evidence_mode": EVIDENCE_PRODUCTION,
                "source": "recomputed_from_single_saved_mp4",
                "video_sha256": sha256_file(video),
                "extractor_identity": {"id": "sc_sstw_learned_observation_saved_mp4_13x30_v1", "source_path": "src/sc_sstw_feasibility/learned_observation.py", "source_sha256": frozen.source_hashes["src/sc_sstw_feasibility/learned_observation.py"]},
                "comparison": {"atol": 1e-12, "rtol": 0.0},
                "features": cached_features,
            })
            (condition_dir / "stdout.txt").write_text("production-schema synthetic decoder integration\n", encoding="utf-8")
            (condition_dir / "stderr.txt").write_text("", encoding="utf-8")
            saved_video_path = str(video.relative_to(path))
            _write_json(condition_dir / "config.json", condition_config_payload(frozen, plan_group, condition, latent_sha, saved_video_path, EVIDENCE_PRODUCTION))
            _write_json(condition_dir / "environment.json", _production_environment(frozen.config))
            _write_json(condition_dir / "command.json", command_artifact_payload(frozen, EVIDENCE_PRODUCTION))
            schedule = schedule_a() if condition == "A" else schedule_b()
            carrier_records = [] if condition == "OFF" else [
                {
                    "call_index": index,
                    "module_path": frozen.config["carrier"]["module_path"],
                    "schedule_sha256": sha256_bytes(canonical_json_bytes(schedule)),
                    "output_shape": [2, 21, 9600, 1536],
                    "output_dtype": "torch.bfloat16",
                    "distinct_storage": True,
                    "effective_relative_rms": frozen.config["carrier"]["target_relative_rms"],
                    "evidence_mode": EVIDENCE_PRODUCTION,
                }
                for index in range(16)
            ]
            _write_json(condition_dir / "integrity.json", integrity_artifact_payload(frozen, plan_group, condition, latent_sha, EVIDENCE_PRODUCTION, carrier_records, codec_identity))
            files = {
                "video": video,
                "features": condition_dir / "features.json",
                "stdout": condition_dir / "stdout.txt",
                "stderr": condition_dir / "stderr.txt",
                "config": condition_dir / "config.json",
                "environment": condition_dir / "environment.json",
                "command": condition_dir / "command.json",
                "integrity": condition_dir / "integrity.json",
            }
            artifacts = {name: {"path": str(files[name].relative_to(path)), "sha256": sha256_file(files[name])} for name in ARTIFACT_NAMES}
            conditions.append({"condition": condition, "schedule_id": {"OFF": "NONE", "A": "A", "B": "B"}[condition], "carrier_enabled": condition != "OFF", "initial_latent_sha256": latent_sha, "matched_parameters_sha256": parameter_sha, "artifacts": artifacts})
            attempts.append({"condition": condition, "attempt_index": 0, "outcome": "success", "matched_parameters_sha256": parameter_sha})
        groups.append({"group_id": plan_group["group_id"], "content_grammar": plan_group["content_grammar"], "prompt": plan_group["prompt"], "prompt_sha256": parameters["prompt_sha256"], "seed": plan_group["seed"], "matched_parameters": parameters, "conditions": conditions, "attempts": attempts})
    record = {"schema_version": 1, "execution_schema": EXECUTION_SCHEMA, "protocol_id": PROTOCOL_ID, "plan_sha256": PLAN_RAW_SHA256, "prerequisite_identity": frozen.prerequisite.identity(), "evidence_mode": EVIDENCE_PRODUCTION, "evidence_schema": PRODUCTION_EVIDENCE_SCHEMA, "groups": groups}
    record_path = path / "execution.json"
    _write_json(record_path, record, pretty=True)
    return record_path


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    prerequisite = tmp_path / "prerequisite"
    manifest_path = tmp_path / "authorization.json"
    head, tree = _make_repo(repo)
    checksums_sha = _make_prerequisite(prerequisite)
    manifest = _make_manifest(manifest_path, repo, prerequisite, checksums_sha, head=head, tree=tree)
    frozen = preflight(repo, manifest_path, declared_source_commit=head)
    execution_path = _make_execution(tmp_path / "execution", frozen)
    return {"repo": repo, "prerequisite": prerequisite, "manifest_path": manifest_path, "manifest": manifest, "head": head, "tree": tree, "frozen": frozen, "execution_path": execution_path}


def test_real_g0_cli_exports_prerequisite_consumed_by_rc1_preflight() -> None:
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        root = Path(temporary)
        repo = root / "combined-repo"
        head, tree = _make_repo(repo)
        prerequisite = _run_real_g0_success(repo, root, head, tree)
        manifest_path = root / "rc1-authorization.json"
        _make_manifest(manifest_path, repo, prerequisite, sha256_file(prerequisite / "checksums.sha256"), head=head, tree=tree)
        frozen = preflight(repo, manifest_path, declared_source_commit=head)
        assert frozen.prerequisite.selected_candidate == "A1"
        assert frozen.prerequisite.source_head == head
        assert frozen.prerequisite.source_tree == tree
        produced_readout = json.loads((prerequisite / "readout.json").read_text())
        assert np.array_equal(frozen.prerequisite.readout, np.asarray(produced_readout["coefficients"]))


def _rewrite_record(path: Path, mutate) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record)
    _write_json(path, record, pretty=True)


def _reason(exc: pytest.ExceptionInfo[InvalidExperiment]) -> str:
    return exc.value.reason_code


def test_schedule_structure_cross_residual_and_heterogeneous_plan() -> None:
    a, b = schedule_a(), schedule_b()
    assert a[:3] == b[:3]
    assert sorted(a) == sorted(b)
    assert b[4] == a[5] and b[5] == a[4]
    assert a[6:] == b[6:]
    result = schedule_preflight()
    assert result["passed"] is True
    assert result["A_observation_by_B_template"] == pytest.approx(0.761311945935564, abs=1e-9)
    config = validate_frozen_config_bytes((REPO_ROOT / CONFIG_PATH).read_bytes())
    plan = validate_frozen_plan_bytes((REPO_ROOT / PLAN_PATH).read_bytes(), config)
    assert len({group["content_grammar"] for group in plan["groups"]}) == 2
    assert config["g0_baseline"]["independent_implementation_gate"] == "NOT_PASSED"


def test_valid_synthetic_execution_scans_every_case_without_averaging(environment: dict[str, object]) -> None:
    record, artifacts = validate_execution_record(environment["execution_path"], environment["frozen"], synthetic_fixture=True)
    groups, passed = evaluate_execution(record, environment["execution_path"], artifacts, environment["frozen"], synthetic_fixture=True)
    assert passed is True
    assert len(groups) == 2
    for group in groups:
        assert group["group_pass"] is True
        for condition in group["conditions"]:
            assert tuple(condition["templates"]) == TEMPLATES
            assert all([window["start"] for window in condition["templates"][template]] == list(START_INDICES) for template in TEMPLATES)
            assert condition["case_pass"] is True


def test_cpu_harness_saved_mp4_is_decoded_recomputed_and_enters_test_evaluator(environment: dict[str, object], tmp_path: Path) -> None:
    outcome = _make_receipted_production_execution(tmp_path / "production-execution", environment["frozen"], real_mp4=True)
    record_path = outcome.record_path
    record, artifacts = validate_execution_record(
        record_path,
        environment["frozen"],
        synthetic_fixture=False,
        generation_receipt=outcome.receipt,
        allow_cpu_test_harness=True,
    )
    groups, passed = evaluate_execution(record, record_path, artifacts, environment["frozen"], synthetic_fixture=False)
    assert passed is False
    assert len(groups) == 2
    assert record["evidence_mode"] == EVIDENCE_CPU_HARNESS


def _mutate_receipt_snapshot(receipt: GenerationReceipt, mutate) -> None:
    snapshot = json.loads(receipt._snapshot)
    mutate(snapshot)
    object.__setattr__(receipt, "_snapshot", canonical_json_bytes(snapshot))


@pytest.mark.parametrize("fake_receipt", [None, {}, {"completed": True}, SimpleNamespace(completed=True)])
def test_production_validator_rejects_missing_mapping_or_object_receipt_before_read(environment: dict[str, object], tmp_path: Path, fake_receipt: object) -> None:
    sentinel = tmp_path / "unreadable-sentinel" / "execution.json"
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(sentinel, environment["frozen"], synthetic_fixture=False, generation_receipt=fake_receipt)
    assert _reason(caught) == "GENERATION_RECEIPT_INVALID"


def test_generation_receipt_is_single_use_and_cpu_harness_requires_explicit_internal_admission(environment: dict[str, object], tmp_path: Path) -> None:
    outcome = _make_receipted_production_execution(tmp_path / "receipt-replay", environment["frozen"])
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(outcome.record_path, environment["frozen"], synthetic_fixture=False, generation_receipt=outcome.receipt)
    assert _reason(caught) == "CPU_TEST_HARNESS_FORBIDDEN"
    with pytest.raises(InvalidExperiment) as replayed:
        validate_execution_record(outcome.record_path, environment["frozen"], synthetic_fixture=False, generation_receipt=outcome.receipt, allow_cpu_test_harness=True)
    assert _reason(replayed) == "GENERATION_RECEIPT_INVALID"


def test_cpu_backend_cannot_select_production_with_cuda_flags_or_snapshot(environment: dict[str, object], tmp_path: Path) -> None:
    outcome = run_gpu_generation(
        environment["frozen"],
        tmp_path / "malicious-cpu-backend",
        [RUNNER_PATH, "--generate"],
        _test_backend=_MasqueradingCPUBackend(),
    )
    snapshot = json.loads(outcome.receipt._snapshot)
    assert outcome.cpu_only_test_harness is True
    assert snapshot["provenance_class"] == "cpu_test_harness"
    assert snapshot["cpu_only_test_harness"] is True
    assert snapshot["loaded_identity"]["device"] == "cpu"
    assert snapshot["environment"]["runtime"]["gpu"] == "cpu_control_flow_harness"
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(outcome.record_path, environment["frozen"], synthetic_fixture=False, generation_receipt=outcome.receipt)
    assert _reason(caught) == "CPU_TEST_HARNESS_FORBIDDEN"


def test_cpu_receipt_flag_and_device_tamper_has_no_resigning_path(environment: dict[str, object], tmp_path: Path) -> None:
    outcome = _make_receipted_production_execution(tmp_path / "cpu-relabel-attempt", environment["frozen"])
    _mutate_receipt_snapshot(
        outcome.receipt,
        lambda snapshot: (snapshot.update(cpu_only_test_harness=False, provenance_class="production_generation"), snapshot["loaded_identity"].update(device="cuda:0")),
    )
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(outcome.record_path, environment["frozen"], synthetic_fixture=False, generation_receipt=outcome.receipt)
    assert _reason(caught) == "GENERATION_RECEIPT_INVALID"


def test_generation_failure_occurs_before_any_receipt_is_issued(environment: dict[str, object], tmp_path: Path) -> None:
    output = tmp_path / "failed-before-receipt"
    with pytest.raises(RuntimeError, match="before receipt issue"):
        run_gpu_generation(environment["frozen"], output, [RUNNER_PATH, "--generate"], _test_backend=_FailingCPUBackend())
    assert not (output / "execution.json").exists()
    import src.sc_sstw_feasibility.rc1_gpu_generation as generation
    assert not any("active_receipt" in name.lower() or "issue" in name.lower() for name in dir(generation))


def test_generation_receipt_has_no_public_constructor_and_deserialized_copy_is_invalid(environment: dict[str, object], tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        GenerationReceipt()
    outcome = _make_receipted_production_execution(tmp_path / "receipt-deserialization", environment["frozen"])
    with pytest.raises(TypeError):
        pickle.dumps(outcome.receipt)


def test_receipt_issuer_is_not_available_by_ordinary_module_access() -> None:
    import src.sc_sstw_feasibility.rc1_gpu_generation as generation

    with pytest.raises(ImportError):
        exec("from src.sc_sstw_feasibility.rc1_gpu_generation import _issue_generation_receipt", {})
    forbidden = {"_issue_generation_receipt", "issue_generation_receipt", "from_snapshot", "from_mapping", "_make_receipt_authority", "_run_gpu_generation_control_flow"}
    assert forbidden.isdisjoint(dir(generation))
    assert all("issue" not in name.lower() and "from_snapshot" not in name.lower() for name in dir(generation))
    for name in forbidden:
        with pytest.raises(AttributeError):
            getattr(generation, name)
    assert not hasattr(generation.run_gpu_generation, "issuer")


def test_receipt_snapshot_is_locally_assembled_and_backend_cannot_supply_authority_payload() -> None:
    source = (REPO_ROOT / GPU_LIBRARY_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "_issue_generation_receipt" not in top_level_functions
    control = top_level_functions["_run_gpu_generation_control_flow"]
    control_source = ast.get_source_segment(source, control)
    assert control_source is not None
    assert "receipt_snapshot = {" in control_source
    assert "_authority_issue(record_path, receipt_snapshot, provenance_class)" in control_source
    forbidden_backend_authority_inputs = (
        "_test_backend.load_environment",
        "_test_backend.loaded_identity",
        "_test_backend.snapshot",
        "_test_backend.provenance_class",
        "_test_backend.cpu_only_test_harness",
    )
    assert not any(token in control_source for token in forbidden_backend_authority_inputs)


def test_protocol_declares_finite_process_threat_boundary_without_strong_authenticity_claims() -> None:
    protocol = (REPO_ROOT / PROTOCOL_PATH).read_text(encoding="utf-8").lower()
    required = (
        "ordinary import/getattr",
        "cpu-harness relabelling",
        "arbitrary code execution already",
        "closure/cell reflection",
        "monkeypatching",
        "direct memory modification",
        "debugger injection",
        "os process isolation",
        "independently held signing secret",
    )
    assert all(token in protocol for token in required)
    forbidden = ("unforgeable", "secure attestation", "cryptographically secure", "remote proof")
    assert not any(token in protocol for token in forbidden)


def test_receipt_construction_copy_serialization_and_field_mutation_cannot_register(environment: dict[str, object], tmp_path: Path) -> None:
    outcome = _make_receipted_production_execution(tmp_path / "receipt-copy-matrix", environment["frozen"])
    receipt = outcome.receipt
    forged = object.__new__(GenerationReceipt)
    object.__setattr__(forged, "_seal", receipt._seal)
    object.__setattr__(forged, "_snapshot", receipt._snapshot)
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(outcome.record_path, environment["frozen"], synthetic_fixture=False, generation_receipt=forged, allow_cpu_test_harness=True)
    assert _reason(caught) == "GENERATION_RECEIPT_INVALID"
    with pytest.raises((TypeError, ValueError)):
        dataclasses.replace(receipt)
    with pytest.raises(TypeError):
        copy.copy(receipt)
    with pytest.raises(TypeError):
        copy.deepcopy(receipt)
    with pytest.raises(TypeError):
        pickle.dumps(receipt)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        receipt._snapshot = b"tampered"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda value: value["preflight"].update(manifest_sha256="f" * 64), "GENERATION_RECEIPT_IDENTITY_MISMATCH"),
        (lambda value: value["preflight"].update(source_head="f" * 40), "GENERATION_RECEIPT_IDENTITY_MISMATCH"),
        (lambda value: value["preflight"].update(config_sha256="f" * 64), "GENERATION_RECEIPT_IDENTITY_MISMATCH"),
        (lambda value: value["preflight"].update(plan_sha256="f" * 64), "GENERATION_RECEIPT_IDENTITY_MISMATCH"),
        (lambda value: value["preflight"].update(prerequisite_identity={}), "GENERATION_RECEIPT_IDENTITY_MISMATCH"),
        (lambda value: value["implementation"].update(generation_module_sha256="f" * 64), "GENERATION_RECEIPT_IDENTITY_MISMATCH"),
        (lambda value: value["loaded_identity"].update(loaded_revision="f" * 40), "GENERATION_RECEIPT_RUNTIME_MISMATCH"),
        (lambda value: value["loaded_identity"].update(dtype="torch.float32"), "GENERATION_RECEIPT_RUNTIME_MISMATCH"),
        (lambda value: value["loaded_identity"].update(scheduler={}), "GENERATION_RECEIPT_RUNTIME_MISMATCH"),
        (lambda value: value["environment"]["runtime"].update(torch="0.0"), "GENERATION_RECEIPT_RUNTIME_MISMATCH"),
        (lambda value: value["environment"]["scheduler"].update(config_sha256="f" * 64), "GENERATION_RECEIPT_RUNTIME_MISMATCH"),
        (lambda value: value["groups"][0]["initial_latent_identity"].update(sha256="f" * 64), "GENERATION_RECEIPT_LATENT_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][1]["condition_latent_identity"].update(sha256="f" * 64), "GENERATION_RECEIPT_LATENT_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][0].update(carrier_hook_effect=True), "GENERATION_RECEIPT_HOOK_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][1]["carrier_records"][0].update(call_index=99), "GENERATION_RECEIPT_HOOK_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][1]["carrier_records"].pop(), "GENERATION_RECEIPT_HOOK_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][1]["carrier_records"][0].update(schedule_sha256="f" * 64), "GENERATION_RECEIPT_HOOK_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][1]["carrier_records"][0].update(output_shape=[1]), "GENERATION_RECEIPT_HOOK_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][1]["carrier_records"][0].update(output_dtype="torch.float32"), "GENERATION_RECEIPT_HOOK_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][1]["carrier_records"][0].update(effective_relative_rms=0.0), "GENERATION_RECEIPT_HOOK_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][1]["carrier_records"][0].update(input_tensor_sha256="f" * 64), "GENERATION_RECEIPT_HOOK_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][0].update(video_saved_complete=False), "GENERATION_RECEIPT_VIDEO_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][0].update(video_sha256="f" * 64), "GENERATION_RECEIPT_VIDEO_MISMATCH"),
        (lambda value: value["groups"][0]["conditions"][0].update(codec_identity={}), "GENERATION_RECEIPT_VIDEO_MISMATCH"),
    ],
)
def test_generation_receipt_to_disk_tamper_matrix_is_invalid(environment: dict[str, object], tmp_path: Path, mutation, reason: str) -> None:
    outcome = _make_receipted_production_execution(tmp_path / f"receipt-tamper-{reason}-{id(mutation)}", environment["frozen"])
    _mutate_receipt_snapshot(outcome.receipt, mutation)
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(outcome.record_path, environment["frozen"], synthetic_fixture=False, generation_receipt=outcome.receipt, allow_cpu_test_harness=True)
    assert _reason(caught) == "GENERATION_RECEIPT_INVALID"


def test_test_only_synthetic_cannot_masquerade_as_production(environment: dict[str, object]) -> None:
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(environment["execution_path"], environment["frozen"], synthetic_fixture=False)
    assert _reason(caught) == "GENERATION_RECEIPT_INVALID"


@pytest.mark.parametrize(
    "variant",
    ["plain_text", "wrong_codec", "wrong_frame_count", "wrong_fps", "wrong_dimensions"],
)
def test_saved_mp4_decoder_rejects_invalid_container_codec_and_geometry(tmp_path: Path, variant: str) -> None:
    path = tmp_path / f"{variant}.mp4"
    if variant == "plain_text":
        path.write_text("not a video", encoding="utf-8")
    else:
        _write_runtime_mp4(
            path,
            codec="mpeg4" if variant == "wrong_codec" else "libx264",
            frame_count=48 if variant == "wrong_frame_count" else 49,
            fps=7 if variant == "wrong_fps" else 8,
            height=304 if variant == "wrong_dimensions" else 320,
        )
    with pytest.raises(InvalidExperiment) as caught:
        _inspect_and_decode_saved_mp4(path)
    assert caught.value.reason_code in {"SAVED_MP4_INVALID", "SAVED_MP4_CODEC_OR_GEOMETRY_MISMATCH", "SAVED_MP4_DECODE_OR_EXTRACT_FAILURE"}


@pytest.mark.parametrize("tamper", ["stored_features", "extractor_identity", "plain_text_video"])
def test_production_feature_cache_and_video_tampering_are_invalid(environment: dict[str, object], tmp_path: Path, tamper: str) -> None:
    outcome = _make_receipted_production_execution(tmp_path / f"production-{tamper}", environment["frozen"])
    record_path = outcome.record_path
    record = json.loads(record_path.read_text())
    condition = record["groups"][0]["conditions"][0]
    feature_path = record_path.parent / condition["artifacts"]["features"]["path"]
    video_path = record_path.parent / condition["artifacts"]["video"]["path"]
    if tamper == "plain_text_video":
        video_path.write_text("plain text with an mp4 extension", encoding="utf-8")
        condition["artifacts"]["video"]["sha256"] = sha256_file(video_path)
        payload = json.loads(feature_path.read_text())
        payload["video_sha256"] = sha256_file(video_path)
        _write_json(feature_path, payload)
    else:
        payload = json.loads(feature_path.read_text())
        if tamper == "stored_features":
            payload["features"][0][0] += 0.1
        else:
            payload["extractor_identity"]["source_sha256"] = "f" * 64
        _write_json(feature_path, payload)
    condition["artifacts"]["features"]["sha256"] = sha256_file(feature_path)
    _write_json(record_path, record)
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(
            record_path,
            environment["frozen"],
            synthetic_fixture=False,
            generation_receipt=outcome.receipt,
            allow_cpu_test_harness=True,
        )
    assert _reason(caught) == "GENERATION_RECEIPT_DISK_MISMATCH"


def test_missing_prerequisite_is_not_invalid_and_precedes_generation_access(tmp_path: Path) -> None:
    with pytest.raises(PrerequisiteNotMet) as caught:
        validate_prerequisite_package(tmp_path / "absent", HEX64)
    assert caught.value.reason_code == "PREREQUISITE_PACKAGE_MISSING"


def test_hidden_g0_staging_directory_is_never_a_consumable_prerequisite(tmp_path: Path) -> None:
    staging = tmp_path / ".g0-package-staging-output-abcd"
    _make_prerequisite(staging)
    with pytest.raises(InvalidExperiment) as caught:
        validate_prerequisite_package(staging, sha256_file(staging / "checksums.sha256"))
    assert _reason(caught) == "PREREQUISITE_PACKAGE_PARTIAL"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda audit: audit.update(selected_candidate="A3"), "PREREQUISITE_CANDIDATE_INVALID"),
        (lambda audit: audit.update(formal_result=True), "PREREQUISITE_EVIDENCE_BOUNDARY_INVALID"),
        (lambda audit: audit["candidates"][0]["derived_envelope"].update(max_residual=0.251), "PREREQUISITE_THRESHOLDS_INVALID"),
    ],
)
def test_forged_candidate_boundary_or_threshold_is_invalid(tmp_path: Path, mutation, reason: str) -> None:
    package = tmp_path / "prerequisite"
    checksums_sha = _make_prerequisite(package)
    audit = json.loads((package / "audit.json").read_text(encoding="utf-8"))
    mutation(audit)
    _write_json(package / "audit.json", audit)
    checksums = "".join(f"{sha256_file(package / name)}  {name}\n" for name in G0_SUCCESS_PACKAGE_FILES)
    (package / "checksums.sha256").write_text(checksums, encoding="utf-8")
    with pytest.raises(InvalidExperiment) as caught:
        validate_prerequisite_package(package, sha256_file(package / "checksums.sha256"))
    assert _reason(caught) == reason


def test_damaged_prerequisite_hash_is_invalid(tmp_path: Path) -> None:
    package = tmp_path / "prerequisite"
    checksums_sha = _make_prerequisite(package)
    (package / "readout.json").write_bytes(b"{}\n")
    with pytest.raises(InvalidExperiment) as caught:
        validate_prerequisite_package(package, checksums_sha)
    assert _reason(caught) == "PREREQUISITE_FILE_HASH_MISMATCH"


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("fake_declaration", "SOURCE_COMMIT_DECLARATION_MISMATCH"),
        ("head", "HEAD_MISMATCH"),
        ("tree", "TREE_MISMATCH"),
        ("source_code_hash", "SOURCE_FILE_HASH_MISMATCH"),
        ("protocol_hash", "SOURCE_FILE_HASH_MISMATCH"),
        ("notebook_hash", "SOURCE_FILE_HASH_MISMATCH"),
        ("config_identity", "CONFIG_IDENTITY_MISMATCH"),
    ],
)
def test_source_manifest_mismatches_fail_closed(environment: dict[str, object], kind: str, reason: str) -> None:
    manifest = copy.deepcopy(environment["manifest"])
    declared = environment["head"]
    if kind == "fake_declaration":
        declared = "f" * 40
    elif kind == "head":
        manifest["expected_source"]["head"] = "f" * 40
    elif kind == "tree":
        manifest["expected_source"]["tree"] = "f" * 40
    elif kind.endswith("_hash"):
        changed_path = {
            "source_code_hash": "src/sc_sstw_feasibility/rc1_method_validation.py",
            "protocol_hash": "protocols/rc1_method_validation.md",
            "notebook_hash": NOTEBOOK_PATH,
        }[kind]
        manifest["source_files"][changed_path] = "f" * 64
    else:
        manifest["config"]["sha256"] = "f" * 64
    _write_json(environment["manifest_path"], manifest)
    with pytest.raises(InvalidExperiment) as caught:
        preflight(environment["repo"], environment["manifest_path"], declared_source_commit=declared)
    assert _reason(caught) == reason


def test_dirty_source_fails_before_prerequisite_access(environment: dict[str, object]) -> None:
    (environment["repo"] / "untracked.txt").write_text("dirty", encoding="utf-8")
    shutil.rmtree(environment["prerequisite"])
    with pytest.raises(InvalidExperiment) as caught:
        preflight(environment["repo"], environment["manifest_path"])
    assert _reason(caught) == "DIRTY_WORKTREE"


def test_unreadable_git_fails_closed_before_prerequisite_access(tmp_path: Path, environment: dict[str, object]) -> None:
    nongit = tmp_path / "not-a-repository"
    nongit.mkdir()
    shutil.rmtree(environment["prerequisite"])
    with pytest.raises(InvalidExperiment) as caught:
        preflight(nongit, environment["manifest_path"])
    assert _reason(caught) == "GIT_STATE_UNREADABLE"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda r: r["groups"][0]["matched_parameters"].update(guidance_scale=6.0), "TRIPLET_PARAMETER_MISMATCH"),
        (lambda r: r["groups"][0]["conditions"][1].update(initial_latent_sha256="f" * 64), "TRIPLET_LATENT_MISMATCH"),
        (lambda r: r["groups"][0]["conditions"].pop(), "TRIPLET_CONDITION_SET_MISMATCH"),
        (lambda r: r["groups"][0]["conditions"].insert(2, copy.deepcopy(r["groups"][0]["conditions"][1])), "TRIPLET_CONDITION_SET_MISMATCH"),
        (lambda r: r["groups"][0]["conditions"][1].update(schedule_id="B"), "TRIPLET_CONDITION_LABEL_MISMATCH"),
        (lambda r: r["groups"][0].update(prompt="changed"), "TRIPLET_PROMPT_OR_SEED_MISMATCH"),
    ],
)
def test_triplet_integrity_mismatches_fail_before_artifact_reads(environment: dict[str, object], mutation, reason: str) -> None:
    mutation_record = environment["execution_path"]
    _rewrite_record(mutation_record, mutation)
    shutil.rmtree(mutation_record.parent / "orbital_glass")
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(mutation_record, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == reason


def test_forbidden_formal_id_path_is_rejected_before_read(environment: dict[str, object], tmp_path: Path) -> None:
    disguised = tmp_path / "disguised-41008-input.json"
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(disguised, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == "FORBIDDEN_FORMAL_PATH"
    with pytest.raises(InvalidExperiment) as caught:
        preflight(environment["repo"], tmp_path / "fake41007manifest.json")
    assert _reason(caught) == "FORBIDDEN_FORMAL_PATH"


def test_artifact_path_escape_is_rejected_before_external_read(environment: dict[str, object], tmp_path: Path) -> None:
    record_path = environment["execution_path"]
    external = tmp_path / "external-secret.bin"
    external.write_bytes(b"must not be opened")
    record = json.loads(record_path.read_text())
    record["groups"][0]["conditions"][0]["artifacts"]["video"] = {"path": str(external), "sha256": sha256_file(external)}
    _write_json(record_path, record)
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == "EXECUTION_ARTIFACT_PATH_INVALID"


def test_artifact_replacement_and_retry_parameter_change_are_invalid(environment: dict[str, object]) -> None:
    record_path = environment["execution_path"]
    record = json.loads(record_path.read_text())
    artifact = record["groups"][0]["conditions"][0]["artifacts"]["video"]
    (record_path.parent / artifact["path"]).write_bytes(b"replacement")
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == "TRIPLET_ARTIFACT_HASH_MISMATCH"

    artifact["sha256"] = sha256_file(record_path.parent / artifact["path"])
    record["groups"][0]["attempts"][0]["matched_parameters_sha256"] = "f" * 64
    _write_json(record_path, record)
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == "TRIPLET_ATTEMPT_LOG_INVALID"


@pytest.mark.parametrize(
    ("artifact_name", "mutate", "reason"),
    [
        ("config", lambda value: value["matched_parameters"].update(guidance_scale=9.0), "EXECUTION_CONFIG_INVALID"),
        ("environment", lambda value: value.update(model_revision=""), "EXECUTION_ENVIRONMENT_INVALID"),
        ("command", lambda value: value.update(runner="forged_runner.py"), "EXECUTION_COMMAND_INVALID"),
        ("integrity", lambda value: value.update(seed=999), "EXECUTION_INTEGRITY_INVALID"),
    ],
)
def test_semantic_artifact_fields_are_validated(environment: dict[str, object], artifact_name: str, mutate, reason: str) -> None:
    record_path = environment["execution_path"]
    record = json.loads(record_path.read_text())
    condition = record["groups"][0]["conditions"][0]
    artifact_path = record_path.parent / condition["artifacts"][artifact_name]["path"]
    payload = json.loads(artifact_path.read_text())
    mutate(payload)
    _write_json(artifact_path, payload)
    condition["artifacts"][artifact_name]["sha256"] = sha256_file(artifact_path)
    _write_json(record_path, record)
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == reason


@pytest.mark.parametrize("condition", ["OFF", "A", "B"])
def test_carrier_record_semantics_are_fail_closed(environment: dict[str, object], condition: str) -> None:
    record_path = environment["execution_path"]
    record = json.loads(record_path.read_text())
    condition_record = next(item for item in record["groups"][0]["conditions"] if item["condition"] == condition)
    integrity_path = record_path.parent / condition_record["artifacts"]["integrity"]["path"]
    payload = json.loads(integrity_path.read_text())
    if condition == "OFF":
        payload["carrier_records"] = [{"forged": True}]
    else:
        payload["carrier_records"][0]["schedule_sha256"] = "f" * 64
    _write_json(integrity_path, payload)
    condition_record["artifacts"]["integrity"]["sha256"] = sha256_file(integrity_path)
    _write_json(record_path, record)
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) in {"EXECUTION_INTEGRITY_INVALID", "EXECUTION_CARRIER_RECORDS_INVALID"}


@pytest.mark.parametrize("bad_condition", ["A", "B", "OFF"])
def test_cross_template_wrong_window_or_off_false_positive_fails_entire_screen(environment: dict[str, object], bad_condition: str) -> None:
    record_path = environment["execution_path"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    target = next(item for item in record["groups"][0]["conditions"] if item["condition"] == bad_condition)
    feature_path = record_path.parent / target["artifacts"]["features"]["path"]
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    if bad_condition == "A":
        payload["features"] = _raw_features(schedule_b()).tolist()
    elif bad_condition == "B":
        payload["features"] = _raw_features(schedule_a()).tolist()
    else:
        payload["features"] = _raw_features(schedule_a()).tolist()
    _write_json(feature_path, payload)
    target["artifacts"]["features"]["sha256"] = sha256_file(feature_path)
    _write_json(record_path, record, pretty=True)
    validated, artifacts = validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    groups, passed = evaluate_execution(validated, record_path, artifacts, environment["frozen"], synthetic_fixture=True)
    assert passed is False
    assert groups[0]["group_pass"] is False
    assert groups[1]["group_pass"] is True


def test_fixed_time_wrong_window_pseudocorrelation_is_rejected(environment: dict[str, object]) -> None:
    record_path = environment["execution_path"]
    record = json.loads(record_path.read_text())
    target = next(item for item in record["groups"][0]["conditions"] if item["condition"] == "A")
    feature_path = record_path.parent / target["artifacts"]["features"]["path"]
    payload = json.loads(feature_path.read_text())
    shifted = np.roll(np.asarray(schedule_a()), 1, axis=0)
    payload["features"] = _raw_features(tuple(map(tuple, shifted))).tolist()
    _write_json(feature_path, payload)
    target["artifacts"]["features"]["sha256"] = sha256_file(feature_path)
    _write_json(record_path, record)
    validated, artifacts = validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    groups, passed = evaluate_execution(validated, record_path, artifacts, environment["frozen"], synthetic_fixture=True)
    assert passed is False
    a_case = next(item for item in groups[0]["conditions"] if item["condition"] == "A")
    assert a_case["templates"]["A"][1]["accepted"] is True
    assert a_case["case_pass"] is False


def test_manifest_forbids_candidate_schedule_and_threshold_injection(environment: dict[str, object]) -> None:
    for unexpected in ("selected_candidate", "schedule_override", "threshold_override", "extra_input"):
        manifest = copy.deepcopy(environment["manifest"])
        manifest[unexpected] = "forbidden"
        with pytest.raises(InvalidExperiment) as caught:
            validate_manifest_schema(manifest)
        assert _reason(caught) == "MANIFEST_SCHEMA_MISMATCH"


def test_production_only_manifest_rejects_synthetic_cli_mode(environment: dict[str, object], tmp_path: Path) -> None:
    manifest = copy.deepcopy(environment["manifest"])
    manifest["evidence_policy"]["synthetic_fixture_permitted"] = False
    _write_json(environment["manifest_path"], manifest)
    output = tmp_path / "production-only-rejection"
    completed = subprocess.run(
        [sys.executable, "-B", str(environment["repo"] / RUNNER_PATH), "--manifest", str(environment["manifest_path"]), "--output", str(output), "--source-commit", environment["head"], "--execution-package", str(environment["execution_path"]), "--synthetic-fixture"],
        cwd=environment["repo"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    audit = json.loads((output / "audit.json").read_text())
    assert audit["status"] == STATUS_INVALID and audit["reason_code"] == "EXECUTION_MODE_MISMATCH"


def test_cli_real_path_emits_pass_fail_invalid_and_prerequisite_packages(environment: dict[str, object], tmp_path: Path) -> None:
    runner = environment["repo"] / RUNNER_PATH
    base = [sys.executable, "-B", str(runner), "--manifest", str(environment["manifest_path"]), "--source-commit", environment["head"]]
    pass_output = tmp_path / "cli-pass"
    completed = subprocess.run([*base, "--output", str(pass_output), "--execution-package", str(environment["execution_path"]), "--synthetic-fixture"], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((pass_output / "audit.json").read_text())["status"] == STATUS_PASS

    missing_output = tmp_path / "cli-prerequisite"
    completed = subprocess.run([*base, "--output", str(missing_output)], text=True, capture_output=True)
    missing = json.loads((missing_output / "audit.json").read_text())
    assert completed.returncode == 2 and missing["status"] == STATUS_PREREQUISITE
    assert missing["generation_inputs_accessed"] is False and missing["science_metrics_present"] is False

    invalid_output = tmp_path / "cli-invalid"
    completed = subprocess.run([*base, "--output", str(invalid_output), "--execution-package", str(environment["execution_path"]), "--synthetic-fixture", "--source-commit", "f" * 40], text=True, capture_output=True)
    invalid = json.loads((invalid_output / "audit.json").read_text())
    assert completed.returncode == 2 and invalid["status"] == STATUS_INVALID
    assert invalid["science_metrics_present"] is False and "groups" not in invalid

    record = json.loads(environment["execution_path"].read_text())
    target = record["groups"][0]["conditions"][0]
    feature_path = environment["execution_path"].parent / target["artifacts"]["features"]["path"]
    payload = json.loads(feature_path.read_text())
    payload["features"] = _raw_features(schedule_a()).tolist()
    _write_json(feature_path, payload)
    target["artifacts"]["features"]["sha256"] = sha256_file(feature_path)
    _write_json(environment["execution_path"], record)
    fail_output = tmp_path / "cli-fail"
    completed = subprocess.run([*base, "--output", str(fail_output), "--execution-package", str(environment["execution_path"]), "--synthetic-fixture"], text=True, capture_output=True)
    failed = json.loads((fail_output / "audit.json").read_text())
    assert completed.returncode == 3 and failed["status"] == STATUS_FAIL

    passing = json.loads((pass_output / "audit.json").read_text())
    assert passing["test_only_synthetic"] is True
    assert passing["generation_trust_boundary"] == "external_test_only_synthetic_fixture"
    for audit in (passing, missing, invalid, failed):
        assert audit["formal_result"] is False and audit["stage_progression_allowed"] is False
    assert passing["conclusion"] == PASS_CONCLUSION
    assert not any(token in json.dumps(passing).lower() for token in FORBIDDEN_CLAIM_TOKENS)


def test_cpu_only_backend_traverses_real_runner_generation_receipt_decode_and_evaluator(environment: dict[str, object], tmp_path: Path) -> None:
    runner = environment["repo"] / RUNNER_PATH
    module = runpy.run_path(str(runner), run_name="rc1_cpu_only_control_flow")
    output = tmp_path / "cpu-harness-runner-output"
    result = module["main"](
        ["--manifest", str(environment["manifest_path"]), "--output", str(output), "--source-commit", environment["head"], "--generate"],
        _test_generation_backend=_CPUOnlyGenerationBackend(real_mp4=True),
    )
    audit = json.loads((output / "audit.json").read_text())
    assert result in {0, 3}
    assert audit["status"] in {STATUS_PASS, STATUS_FAIL}
    assert audit["cpu_only_test_harness"] is True
    assert audit["evidence_mode"] == EVIDENCE_CPU_HARNESS
    assert audit["generation_trust_boundary"] == "cpu_only_control_flow_harness"
    assert audit["formal_result"] is False and audit["stage_progression_allowed"] is False
    assert len(audit["groups"]) == 2
    assert all(len(group["conditions"]) == 3 for group in audit["groups"])


def test_cpu_only_backend_decoder_failure_is_packaged_as_invalid(environment: dict[str, object], tmp_path: Path) -> None:
    runner = environment["repo"] / RUNNER_PATH
    module = runpy.run_path(str(runner), run_name="rc1_cpu_only_decoder_failure")
    output = tmp_path / "cpu-harness-decoder-invalid"
    result = module["main"](
        ["--manifest", str(environment["manifest_path"]), "--output", str(output), "--source-commit", environment["head"], "--generate"],
        _test_generation_backend=_CPUOnlyGenerationBackend(corrupt_video=True),
    )
    audit = json.loads((output / "audit.json").read_text())
    assert result == 2
    assert audit["status"] == STATUS_INVALID
    assert audit["reason_code"] in {"SAVED_MP4_INVALID", "SAVED_MP4_DECODE_OR_EXTRACT_FAILURE"}
    assert audit["science_metrics_present"] is False
    assert "groups" not in audit and "conclusion" not in audit


@pytest.mark.parametrize(
    ("target", "exception_source", "reason", "exception_type"),
    [
        ("validate_execution_record", "RuntimeError('injected ordinary runtime')", "UNEXPECTED_RUNTIME_FAILURE", "RuntimeError"),
        ("evaluate_execution", "RuntimeError('CUDA out of memory')", "UNEXPECTED_MEMORY_FAILURE", "RuntimeError"),
        ("evaluate_execution", "MemoryError('injected')", "UNEXPECTED_MEMORY_FAILURE", "MemoryError"),
        ("validate_execution_record", "TypeError('injected')", "UNEXPECTED_TYPE_FAILURE", "TypeError"),
        ("evaluate_execution", "subprocess.CalledProcessError(1, ['ffprobe'])", "UNEXPECTED_SUBPROCESS_FAILURE", "CalledProcessError"),
    ],
)
def test_real_cli_packages_ordinary_exception_matrix(environment: dict[str, object], tmp_path: Path, target: str, exception_source: str, reason: str, exception_type: str) -> None:
    output = tmp_path / f"exception-{reason}-{exception_type}"
    runner = environment["repo"] / RUNNER_PATH
    script = (
        "import runpy,sys,subprocess\n"
        f"m=runpy.run_path({str(runner)!r}, run_name='rc1_injected_cli')\n"
        f"def boom(*args, **kwargs): raise {exception_source}\n"
        f"m['main'].__globals__[{target!r}]=boom\n"
        f"sys.argv=[{str(runner)!r},'--manifest',{str(environment['manifest_path'])!r},'--output',{str(output)!r},'--source-commit',{str(environment['head'])!r},'--execution-package',{str(environment['execution_path'])!r},'--synthetic-fixture']\n"
        "raise SystemExit(m['main']())\n"
    )
    completed = subprocess.run([sys.executable, "-B", "-c", script], cwd=environment["repo"], text=True, capture_output=True)
    assert completed.returncode == 2, completed.stderr
    audit = json.loads((output / "audit.json").read_text())
    assert audit["status"] == STATUS_INVALID
    assert audit["reason_code"] == reason
    assert audit["science_metrics_present"] is False
    assert "groups" not in audit and "conclusion" not in audit
    assert audit["integrity_context"]["exception_type"] == exception_type
    assert audit["integrity_context"]["failure_phase"] in {"execution_package_validation", "saved_mp4_decode_and_evaluation"}


def test_cli_does_not_overwrite_existing_output_and_uses_invalid_fallback(environment: dict[str, object], tmp_path: Path) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()
    marker = output / "user-owned.txt"
    marker.write_text("preserve", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-B", str(environment["repo"] / RUNNER_PATH), "--manifest", str(environment["manifest_path"]), "--output", str(output), "--source-commit", environment["head"]],
        cwd=environment["repo"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    assert marker.read_text(encoding="utf-8") == "preserve"
    report = json.loads(completed.stdout)
    audit = json.loads(Path(report["audit_path"]).read_text())
    assert audit["status"] == STATUS_INVALID
    assert audit["reason_code"] == "OUTPUT_DIRECTORY_UNAVAILABLE"
    assert audit["science_metrics_present"] is False


@pytest.mark.parametrize("use_unreadable_sentinel", [False, True])
def test_external_production_package_is_rejected_before_any_package_read(environment: dict[str, object], tmp_path: Path, use_unreadable_sentinel: bool) -> None:
    record_path = _make_production_execution(tmp_path / "decoder-cli-execution", environment["frozen"])
    supplied_path = tmp_path / "must-not-be-opened" / "execution.json" if use_unreadable_sentinel else record_path
    output = tmp_path / "decoder-cli-output"
    completed = subprocess.run(
        [sys.executable, "-B", str(environment["repo"] / RUNNER_PATH), "--manifest", str(environment["manifest_path"]), "--output", str(output), "--source-commit", environment["head"], "--execution-package", str(supplied_path)],
        cwd=environment["repo"],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2, completed.stderr
    audit = json.loads((output / "audit.json").read_text())
    assert audit["status"] == STATUS_INVALID
    assert audit["reason_code"] == "EXTERNAL_PRODUCTION_PACKAGE_FORBIDDEN"
    assert audit["science_metrics_present"] is False
    assert "groups" not in audit and "conclusion" not in audit
    assert audit["integrity_context"]["failure_phase"] == "execution_entry"


def test_status_vocabulary_claim_ceiling_and_notebook_static_structure() -> None:
    assert {STATUS_PASS, STATUS_FAIL, STATUS_INVALID, STATUS_PREREQUISITE} == {
        "RC1_VALID_PASS", "RC1_VALID_FAIL", "INVALID_EXPERIMENT", "PREREQUISITE_NOT_MET"
    }
    assert "method" not in PASS_CONCLUSION.lower()
    notebook = json.loads((REPO_ROOT / NOTEBOOK_PATH).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = (
        "SC_SSTW_RC1_EXACT_REF", "git', 'checkout', '--detach'", "drive.mount", "torch.cuda.is_available",
        "--generate", "synthetic_fixture_permitted': False", "try:", "finally:", "make_archive", "shutil.copy2", "sha256(drive_copy)", "testzip()", "verification.json", "packaged = True",
    )
    assert all(token in source for token in required)
    assert "--execution-package" not in source
    assert "checkout exact ref" not in source.lower()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), NOTEBOOK_PATH, "exec")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text())
    assert tuple(config["forbidden_claims"]) == FORBIDDEN_CLAIM_TOKENS


@pytest.mark.parametrize(
    ("status", "returncode", "valid"),
    [
        (STATUS_PASS, 0, True),
        (STATUS_FAIL, 3, True),
        (STATUS_PREREQUISITE, 2, False),
        (STATUS_INVALID, 2, False),
    ],
)
def test_notebook_runner_status_helper_accepts_all_four_consistent_states(tmp_path: Path, status: str, returncode: int, valid: bool) -> None:
    notebook = json.loads((REPO_ROOT / NOTEBOOK_PATH).read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    parsed = ast.parse(source)
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "load_runner_audit")
    namespace = {"json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), NOTEBOOK_PATH, "exec"), namespace)
    audit = {
        "schema_version": 1,
        "output_schema": OUTPUT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "valid_experiment": valid,
        "reason_code": "SYNTHETIC_NOTEBOOK_HELPER_TEST",
        "formal_result": False,
        "stage_progression_allowed": False,
    }
    audit_path = tmp_path / "audit.json"
    _write_json(audit_path, audit)
    assert namespace["load_runner_audit"](SimpleNamespace(returncode=returncode), audit_path) == audit


@pytest.mark.parametrize("case", ["missing", "malformed", "exit_mismatch", "status_mismatch", "boundary_mismatch"])
def test_notebook_runner_status_helper_rejects_exceptional_or_inconsistent_results(tmp_path: Path, case: str) -> None:
    notebook = json.loads((REPO_ROOT / NOTEBOOK_PATH).read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][1]["source"])
    parsed = ast.parse(source)
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "load_runner_audit")
    namespace = {"json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), NOTEBOOK_PATH, "exec"), namespace)
    path = tmp_path / "audit.json"
    returncode = 0
    if case == "malformed":
        path.write_text("{", encoding="utf-8")
    elif case != "missing":
        audit = {"schema_version": 1, "output_schema": OUTPUT_SCHEMA, "protocol_id": PROTOCOL_ID, "status": STATUS_PASS, "valid_experiment": True, "reason_code": "TEST", "formal_result": False, "stage_progression_allowed": False}
        if case == "exit_mismatch":
            returncode = 3
        elif case == "status_mismatch":
            audit["status"] = "NOT_A_STATE"
        else:
            audit["formal_result"] = True
        _write_json(path, audit)
    with pytest.raises(RuntimeError):
        namespace["load_runner_audit"](SimpleNamespace(returncode=returncode), path)
