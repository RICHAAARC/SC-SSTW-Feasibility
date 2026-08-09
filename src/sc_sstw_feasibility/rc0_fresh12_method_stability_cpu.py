"""One-shot Phase-C method-stability diagnostic over eight frozen fresh bases."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from .learned_observation import decode_saved_mp4
from .rc0_fresh4_effectiveness_cpu import (
    CONDITION_ORDER,
    TRANSFORM_ORDER,
    _derive_condition_frames,
    _git,
    apply_frozen_frame_transform,
    audit_target,
    canonical_json_bytes,
    evaluate_transformed_target,
    load_frozen_phase_c,
    probe_and_decode_mp4_bytes,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from .rc0_pixel_chroma_fast_cpu import encode_mp4, probe_mp4


SCHEMA = "sc_sstw_rc0_fresh12_method_stability_cpu_v1"
DIAGNOSTIC_CLASS = "DIAGNOSTIC_ONLY"
GROUP_ORDER = (
    "rigid_blocks",
    "elastic_membrane",
    "laminar_plumes",
    "granular_avalanche",
    "matte_balloon",
    "engraved_drum",
    "pendulum_bars",
    "soap_bubbles",
)
SOURCE_ZIP = Path("/mnt/g/我的云端硬盘/SC-SSTW-Feasibility/rc0-fresh8-method-stability/rc0-fresh8-f37e12c0e6a325fe.zip")
SOURCE_ZIP_SIZE = 683048
SOURCE_ZIP_SHA256 = "d6aebdfc115f6c6835fb1569a4356939dbb0d15a619bed75b6b02037c500dc92"
SOURCE_RUN_ID = "f37e12c0e6a325fe"
SOURCE_COMMIT = "ba226dc6e3d11f501c9fded970d2b8c68d3c6546"
SOURCE_TREE = "b2829e9c3d50917429d43071c2827b5f9d65fe74"
FROZEN_CONFIG_SHA256 = "7909a8942a3d683fea1c51d8862f0b550dd8ef437fd82a569fb0a13016c8eeaf"
FROZEN_PROTOCOL_SHA256 = "47875f24437d140bc057dfe22643a5fe90e878b29d5b93def2c74688422de2b0"
FROZEN_PLAN_SHA256 = "d5f84bcffb834e94f20b62f3c1f97e4dc8bf4230dd4bc7265b5e2581864cab55"
BASE_MEMBERS = {
    "rigid_blocks": ("output/bases/rigid_blocks/base.mp4", 26389, "6ed216da790bb48931394aa3679cb3dc56e4f5bb456bdf023ce9f80698e38397"),
    "elastic_membrane": ("output/bases/elastic_membrane/base.mp4", 31152, "66042344fd14d7845ee438598aa3f305c263f1b8d49a76597396b4201e7edd0c"),
    "laminar_plumes": ("output/bases/laminar_plumes/base.mp4", 32262, "bd5f8f3dce169d5b27cc18d119b18a9e2db3c510d5b2c4a07d283c7d6e489eef"),
    "granular_avalanche": ("output/bases/granular_avalanche/base.mp4", 490554, "4c6381206bd82d648ac96ecee9a433d1d1448fa4afc8e20c0e34373e9a402630"),
    "matte_balloon": ("output/bases/matte_balloon/base.mp4", 20985, "5083433244e0a058c5c112a1f9244b061ecef3b44db1c54528f4da5dd91db23c"),
    "engraved_drum": ("output/bases/engraved_drum/base.mp4", 41038, "ef732c5f1ad83a2860ddcfd89585a00062395468674425c9d1ae9e0009631384"),
    "pendulum_bars": ("output/bases/pendulum_bars/base.mp4", 8808, "42657d75c2b41c65c9e6e0502c5e30cd50dc9113de4b5e5ecdb0f2946319d0e2"),
    "soap_bubbles": ("output/bases/soap_bubbles/base.mp4", 25217, "fcd7a191adf35a1ce150c9d4b8a8e544179fce0b30b55c61ed9ed4050774ce0b"),
}
RUN_ROOT = Path("/home/richar/projects/SC-SSTW-Feasibility-diagnostic-runs/rc0-fresh12-method-stability-run1")


class Fresh12StabilityError(RuntimeError):
    pass


def load_frozen_fresh12(repo_root: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    config_path = repo_root / "configs/rc0_fresh8_method_stability_gpu.json"
    protocol_path = repo_root / "protocols/rc0_fresh8_method_stability_gpu.md"
    plan_path = repo_root / "plans/rc0_fresh8_method_stability_gpu.json"
    if (
        sha256_file(config_path), sha256_file(protocol_path), sha256_file(plan_path)
    ) != (FROZEN_CONFIG_SHA256, FROZEN_PROTOCOL_SHA256, FROZEN_PLAN_SHA256):
        raise Fresh12StabilityError("frozen fresh12 objects changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("attempt_order") != list(GROUP_ORDER) or config.get("attempt_budget") != 8:
        raise Fresh12StabilityError("frozen exact8 identity changed")
    if config.get("phase_c_frozen_not_executed", {}).get("derived_total_count") != 160:
        raise Fresh12StabilityError("frozen exact160 budget changed")
    if config["phase_c_frozen_not_executed"].get("corrected_semantics") != "clean_identity_only_transformed_each_own_truth":
        raise Fresh12StabilityError("corrected evaluation semantics changed")
    _fresh4_config, phase_c = load_frozen_phase_c(repo_root)
    return config, phase_c


def read_bound_bases(repo_root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if SOURCE_ZIP.is_symlink() or not SOURCE_ZIP.is_file() or SOURCE_ZIP.stat().st_size != SOURCE_ZIP_SIZE:
        raise Fresh12StabilityError("fresh8 ZIP path or size mismatch")
    if sha256_file(SOURCE_ZIP) != SOURCE_ZIP_SHA256:
        raise Fresh12StabilityError("fresh8 ZIP SHA mismatch")
    config, _phase_c = load_frozen_fresh12(repo_root)
    frames: dict[str, np.ndarray] = {}
    identities: dict[str, Any] = {}
    with zipfile.ZipFile(SOURCE_ZIP, "r") as archive:
        if archive.testzip() is not None:
            raise Fresh12StabilityError("fresh8 ZIP CRC failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise Fresh12StabilityError("fresh8 ZIP contains duplicate names")
        audit = json.loads(archive.read("output/audit.json"))
        embedded_config = json.loads(archive.read("output/config.json"))
        state = json.loads(archive.read("notebook_state.json"))
        if canonical_json_bytes(embedded_config) != canonical_json_bytes(config):
            raise Fresh12StabilityError("embedded config changed")
        if (
            audit.get("status"), audit.get("source"), audit.get("config_sha256"), audit.get("protocol_sha256"),
            audit.get("plan_sha256"), audit.get("attempts_started"), audit.get("attempts_completed"), audit.get("retry_count"),
            audit.get("carrier_was_used"), audit.get("detector_was_run"), audit.get("phase_c_was_run"),
        ) != (
            "BASE_EXACT8_READY", {"head": SOURCE_COMMIT, "tree": SOURCE_TREE, "dirty": False},
            FROZEN_CONFIG_SHA256, FROZEN_PROTOCOL_SHA256, FROZEN_PLAN_SHA256,
            list(GROUP_ORDER), list(GROUP_ORDER), 0, False, False, False,
        ):
            raise Fresh12StabilityError("embedded base audit identity changed")
        if state.get("run_id") != SOURCE_RUN_ID or state.get("authorized_ref") != SOURCE_COMMIT or state.get("runner_return_code") != 0:
            raise Fresh12StabilityError("notebook run identity changed")
        checksums = archive.read("output/checksums.sha256").decode("utf-8").splitlines()
        checksum_map = {relative: digest for digest, relative in (line.split("  ", 1) for line in checksums)}
        observed_base_names = {name for name in names if name.startswith("output/bases/") and name.endswith("/base.mp4")}
        if observed_base_names != {record[0] for record in BASE_MEMBERS.values()}:
            raise Fresh12StabilityError("fresh8 base member universe changed")
        for group in GROUP_ORDER:
            member, expected_size, expected_sha = BASE_MEMBERS[group]
            info = archive.getinfo(member)
            mode = info.external_attr >> 16
            data = archive.read(member)
            if info.is_dir() or stat.S_ISLNK(mode) or len(data) != expected_size or sha256_bytes(data) != expected_sha:
                raise Fresh12StabilityError(f"base identity mismatch for {group}")
            relative = member.removeprefix("output/")
            if checksum_map.get(relative) != expected_sha or audit.get("artifacts", {}).get(group, {}).get("sha256") != expected_sha:
                raise Fresh12StabilityError(f"base checksum binding mismatch for {group}")
            stream, decoded = probe_and_decode_mp4_bytes(data)
            frames[group] = decoded
            identities[group] = {"member": member, "size": expected_size, "sha256": expected_sha, "stream": stream,
                                 "decoded_array_sha256": sha256_bytes(np.ascontiguousarray(decoded).tobytes(order="C"))}
    return frames, {"zip": {"absolute_path": str(SOURCE_ZIP), "size": SOURCE_ZIP_SIZE, "sha256": SOURCE_ZIP_SHA256},
                    "run_id": SOURCE_RUN_ID, "source_commit": SOURCE_COMMIT, "source_tree": SOURCE_TREE, "bases": identities}


def evaluate_clean_identity(decoded_frames: np.ndarray) -> dict[str, Any]:
    """Clean artifacts are evaluated only against identity truth."""

    return evaluate_transformed_target(decoded_frames, "identity")


def run_fresh12_phase_c_once(*, repo_root: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if RUN_ROOT.exists() or RUN_ROOT.is_symlink():
        raise Fresh12StabilityError("fresh12 run root already exists")
    if not RUN_ROOT.parent.is_dir() or RUN_ROOT.parent.is_symlink():
        raise Fresh12StabilityError("fresh12 run parent unavailable")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise Fresh12StabilityError("fresh12 Phase C requires a clean checkout")
    _config, phase_c = load_frozen_fresh12(repo_root)
    RUN_ROOT.mkdir(mode=0o755)
    started_at = utc_now()
    condition_attempts: list[dict[str, Any]] = []
    transform_attempts: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    condition_outputs: dict[str, Any] = {}
    transform_outputs: dict[str, Any] = {}
    condition_truth: dict[str, Any] = {}
    transform_truth: dict[str, Any] = {}
    first_failure: dict[str, Any] | None = None
    try:
        bases, input_identity = read_bound_bases(repo_root)
        for group in GROUP_ORDER:
            artifacts[group] = {"conditions": {}, "transforms": {}}
            base = bases[group]
            for condition in CONDITION_ORDER:
                key = f"{group}:{condition}"
                condition_attempts.append({"index": len(condition_attempts) + 1, "group": group, "condition": condition,
                                           "retry_index": 0, "started": True, "completed": False})
                encoded_frames, carrier_identity = _derive_condition_frames(base, condition)
                condition_path = RUN_ROOT / "conditions" / group / condition / "saved.mp4"
                encode_mp4(encoded_frames, condition_path)
                condition_stream = probe_mp4(condition_path)
                condition_decoded = decode_saved_mp4(condition_path)
                condition_attempts[-1]["completed"] = True
                condition_attempts[-1]["sha256"] = sha256_file(condition_path)
                clean_output = evaluate_clean_identity(condition_decoded)
                condition_outputs[key] = clean_output
                condition_truth[key], stage = audit_target(clean_output, condition, transformed=True)
                if first_failure is None and stage is not None:
                    first_failure = {"kind": "condition", "group": group, "condition": condition, "truth": "identity", "stage": stage}
                artifacts[group]["conditions"][condition] = {
                    "path": str(condition_path.relative_to(RUN_ROOT)), "sha256": sha256_file(condition_path),
                    "size": condition_path.stat().st_size, "stream": condition_stream, "carrier_identity": carrier_identity,
                    "truth": "identity",
                }
                artifacts[group]["transforms"][condition] = {}
                for transform_id in TRANSFORM_ORDER:
                    transform_attempts.append({"index": len(transform_attempts) + 1, "group": group, "condition": condition,
                                               "transform": transform_id, "retry_index": 0, "started": True, "completed": False})
                    transformed_frames = apply_frozen_frame_transform(condition_decoded, phase_c, transform_id)
                    transformed_path = RUN_ROOT / "transforms" / group / condition / transform_id / "saved.mp4"
                    encode_mp4(transformed_frames, transformed_path)
                    transformed_stream = probe_mp4(transformed_path)
                    transformed_decoded = decode_saved_mp4(transformed_path)
                    transform_attempts[-1]["completed"] = True
                    transform_attempts[-1]["sha256"] = sha256_file(transformed_path)
                    transformed_output = evaluate_transformed_target(transformed_decoded, transform_id)
                    transformed_key = f"{group}:{condition}:{transform_id}"
                    transform_outputs[transformed_key] = transformed_output
                    transform_truth[transformed_key], stage = audit_target(transformed_output, condition, transformed=True)
                    if first_failure is None and stage is not None:
                        first_failure = {"kind": "transform", "group": group, "condition": condition,
                                         "transform": transform_id, "stage": stage}
                    artifacts[group]["transforms"][condition][transform_id] = {
                        "path": str(transformed_path.relative_to(RUN_ROOT)), "sha256": sha256_file(transformed_path),
                        "size": transformed_path.stat().st_size, "stream": transformed_stream, "truth": transform_id,
                        "frame_source_indices": next(item["frame_source_indices"] for item in phase_c["transforms"] if item["id"] == transform_id),
                    }
        clean_off_pass = sum(condition_truth[f"{group}:{condition}"]["passed"] for group in GROUP_ORDER for condition in ("OFF_R1", "OFF_R2"))
        clean_positive_pass = sum(condition_truth[f"{group}:{condition}"]["passed"] for group in GROUP_ORDER for condition in ("A", "B"))
        transformed_off_pass = sum(transform_truth[f"{group}:{condition}:{transform}"]["passed"] for group in GROUP_ORDER for condition in ("OFF_R1", "OFF_R2") for transform in TRANSFORM_ORDER)
        transformed_positive_pass = sum(transform_truth[f"{group}:{condition}:{transform}"]["passed"] for group in GROUP_ORDER for condition in ("A", "B") for transform in TRANSFORM_ORDER)
        feasible = (clean_off_pass, clean_positive_pass, transformed_off_pass, transformed_positive_pass) == (16, 16, 64, 64)
        status = "EFFECTIVE_ON_FRESH_12_CONTENTS" if feasible else "NOT_EFFECTIVE_ON_FRESH_12_CONTENTS"
        phase_c_result = "FEASIBLE" if feasible else "NOT_FEASIBLE"
        route = "METHOD_STABILITY_FRESH12_CLOSED" if feasible else "METHOD_STABILITY_FRESH12_NOT_CLOSED"
        result = {
            "schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status, "phase_c": phase_c_result,
            "route": route, "first_failure": first_failure, "input_identity": input_identity,
            "source_code": {"head": _git(repo_root, "rev-parse", "HEAD"), "tree": _git(repo_root, "rev-parse", "HEAD^{tree}"), "dirty": False},
            "config_sha256": FROZEN_CONFIG_SHA256, "protocol_sha256": FROZEN_PROTOCOL_SHA256, "plan_sha256": FROZEN_PLAN_SHA256,
            "condition_attempts_started": len(condition_attempts), "condition_attempts_completed": sum(item["completed"] for item in condition_attempts),
            "transform_attempts_started": len(transform_attempts), "transform_attempts_completed": sum(item["completed"] for item in transform_attempts),
            "retry_count": 0, "condition_attempts": condition_attempts, "transform_attempts": transform_attempts,
            "artifacts": artifacts, "condition_outputs": condition_outputs, "transform_outputs": transform_outputs,
            "condition_truth_after_outputs_frozen": condition_truth, "transform_truth_after_outputs_frozen": transform_truth,
            "condition_count": len(condition_truth), "transform_cell_count": len(transform_truth),
            "clean_off_pass": clean_off_pass, "clean_positive_identity_pass": clean_positive_pass,
            "transformed_off_pass": transformed_off_pass, "transformed_positive_pass": transformed_positive_pass,
            "corrected_semantics": "clean_identity_only_transformed_each_own_truth",
            "generation_was_run": False, "GPU_was_run": False, "formal_result": False, "stage_progression_allowed": False,
            "started_at": started_at, "ended_at": utc_now(),
        }
        (RUN_ROOT / "audit.json").write_bytes(canonical_json_bytes(result) + b"\n")
        (RUN_ROOT / "result.json").write_bytes(canonical_json_bytes(result) + b"\n")
        response = {"status": status, "diagnostic_class": DIAGNOSTIC_CLASS, "phase_c": phase_c_result, "route": route,
                    "first_failure": first_failure, "actual_run_root": str(RUN_ROOT), "condition_count": len(condition_truth),
                    "transform_cell_count": len(transform_truth), "clean_off_pass": clean_off_pass,
                    "clean_positive_identity_pass": clean_positive_pass, "transformed_off_pass": transformed_off_pass,
                    "transformed_positive_pass": transformed_positive_pass, "retry_count": 0}
        (RUN_ROOT / "command.json").write_bytes(canonical_json_bytes({"argv": list(argv), "cwd": str(cwd.resolve()), "exit_code": 0 if feasible else 3}) + b"\n")
        checksum_files = sorted(path for path in RUN_ROOT.rglob("*") if path.is_file() and path.name != "checksums.sha256")
        (RUN_ROOT / "checksums.sha256").write_text(
            "\n".join(f"{sha256_file(path)}  {path.relative_to(RUN_ROOT).as_posix()}" for path in checksum_files) + "\n",
            encoding="utf-8",
        )
        return response
    except BaseException as exc:
        failure = {"schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": "DIAGNOSTIC_INSUFFICIENT",
                   "phase_c": "INSUFFICIENT_TO_DECIDE", "route": "DIAGNOSTIC_INSUFFICIENT", "reason": type(exc).__name__,
                   "message": str(exc), "condition_attempts": condition_attempts, "transform_attempts": transform_attempts,
                   "science_outputs_present": bool(condition_outputs or transform_outputs), "formal_result": False,
                   "stage_progression_allowed": False, "ended_at": utc_now()}
        (RUN_ROOT / "audit.json").write_bytes(canonical_json_bytes(failure) + b"\n")
        raise
