"""Exact-eight carrier-free Wan base generation for the fresh12 diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import platform
from typing import Any, Sequence

from .learned_observation import decode_saved_mp4, encode_saved_mp4
from .rc0_fresh4_effectiveness_cpu import sha256_file
from .rc0_fresh_exact4_base_gpu import (
    MODEL_ID,
    MODEL_REVISION,
    _git,
    _load_runtime,
    canonical_json_bytes,
    generate_official_base_frames,
    sha256_bytes,
    utc_now,
)
from .rc0_pixel_chroma_fast_cpu import probe_mp4


SCHEMA = "sc_sstw_rc0_fresh8_method_stability_gpu_v1"
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
CATEGORY_ORDER = ("rigid", "nonrigid", "fluid", "particles", "low_texture", "high_texture", "periodic", "aperiodic")
SEEDS = tuple(range(54011, 54019))


class Fresh8DiagnosticError(RuntimeError):
    pass


def load_and_validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != SCHEMA or config.get("diagnostic_class") != DIAGNOSTIC_CLASS:
        raise Fresh8DiagnosticError("fresh8 config identity mismatch")
    if config.get("attempt_order") != list(GROUP_ORDER) or config.get("attempt_budget") != 8:
        raise Fresh8DiagnosticError("exact8 attempt identity changed")
    if config.get("retry_policy") != "no_retry_no_replacement_no_additional_sample":
        raise Fresh8DiagnosticError("retry policy changed")
    groups = config.get("groups")
    if not isinstance(groups, list) or len(groups) != 8:
        raise Fresh8DiagnosticError("exact8 groups missing")
    if tuple(item.get("group_id") for item in groups) != GROUP_ORDER:
        raise Fresh8DiagnosticError("exact8 group order changed")
    if tuple(item.get("category") for item in groups) != CATEGORY_ORDER:
        raise Fresh8DiagnosticError("category coverage changed")
    if tuple(item.get("seed") for item in groups) != SEEDS:
        raise Fresh8DiagnosticError("seed identity changed")
    if len({item.get("content_grammar") for item in groups}) != 8:
        raise Fresh8DiagnosticError("content grammars must remain distinct")
    for item in groups:
        if sha256_bytes(item["prompt"].encode("utf-8")) != item.get("prompt_sha256"):
            raise Fresh8DiagnosticError("prompt digest changed")
    if config.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise Fresh8DiagnosticError("model identity changed")
    generation = config.get("generation", {})
    observed = (
        generation.get("pipeline"), generation.get("decode_path"), generation.get("scheduler"), generation.get("sampler"),
        generation.get("inference_steps"), generation.get("guidance_scale"), generation.get("height"), generation.get("width"),
        generation.get("frame_count"), generation.get("fps"), generation.get("dtype"), generation.get("carrier"), generation.get("hook"),
    )
    expected = (
        "diffusers.WanPipeline", "official_pipeline_default_output_no_manual_latent_decode",
        "WanPipeline_frozen_default_for_revision", "WanPipeline_frozen_default_for_revision",
        8, 5.0, 320, 512, 49, 8, "torch.bfloat16", "none", "none",
    )
    if observed != expected:
        raise Fresh8DiagnosticError("fresh4 generation settings drifted")
    if generation.get("negative_prompt") != "text, watermark, logo, camera motion, cuts, multiple scenes, flicker":
        raise Fresh8DiagnosticError("fresh4 negative prompt drifted")
    encoding = config.get("encoding")
    if encoding != {
        "call": "diffusers.utils.export_to_video", "container": "mp4", "codec": "h264",
        "pixel_format": "yuv420p", "quality": 5.0, "bitrate": None, "macro_block_size": 16,
    }:
        raise Fresh8DiagnosticError("fresh4 encoding settings drifted")
    phase_c = config.get("phase_c_frozen_not_executed", {})
    if (
        phase_c.get("new_base_count"), phase_c.get("derived_clean_count"),
        phase_c.get("derived_transformed_count"), phase_c.get("derived_total_count"),
        phase_c.get("corrected_semantics"), phase_c.get("success_counts"), phase_c.get("execution"),
    ) != (
        8, 32, 128, 160, "clean_identity_only_transformed_each_own_truth",
        {"clean_off_reject": 16, "clean_positive_accept": 16, "transformed_off_reject": 64, "transformed_positive_accept": 64},
        "future_CPU_only_after_GPU_base_delivery",
    ):
        raise Fresh8DiagnosticError("frozen future Phase C statement changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise Fresh8DiagnosticError("diagnostic boundary changed")
    return config


def run_fresh8_base_once(*, repo_root: Path, output: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise Fresh8DiagnosticError("output target must not exist")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise Fresh8DiagnosticError("output parent unavailable")
    config_path = repo_root / "configs/rc0_fresh8_method_stability_gpu.json"
    protocol_path = repo_root / "protocols/rc0_fresh8_method_stability_gpu.md"
    plan_path = repo_root / "plans/rc0_fresh8_method_stability_gpu.json"
    config = load_and_validate_config(config_path)
    head = _git(repo_root, "rev-parse", "HEAD")
    tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise Fresh8DiagnosticError("fresh8 generation requires a clean exact checkout")
    output.mkdir(mode=0o755)
    bases_root = output / "bases"
    bases_root.mkdir()
    started_at = utc_now()
    attempts_started: list[str] = []
    attempts_completed: list[str] = []
    artifacts: dict[str, Any] = {}
    try:
        pipe, torch, runtime = _load_runtime()
        for item in config["groups"]:
            group = item["group_id"]
            attempts_started.append(group)
            group_root = bases_root / group
            group_root.mkdir()
            frames = generate_official_base_frames(pipe, torch, item, config["generation"])
            video = group_root / "base.mp4"
            encode_saved_mp4(frames, video)
            stream = probe_mp4(video)
            decoded = decode_saved_mp4(video)
            if list(decoded.shape) != [49, 320, 512, 3]:
                raise Fresh8DiagnosticError("saved base decode shape changed")
            artifacts[group] = {
                "path": str(video.relative_to(output)), "sha256": sha256_file(video), "size": video.stat().st_size,
                "stream": stream, "prompt_sha256": item["prompt_sha256"], "seed": item["seed"],
                "category": item["category"], "carrier": False, "condition": "BASE",
            }
            attempts_completed.append(group)
    except Exception as exc:
        audit = {
            "schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": "DIAGNOSTIC_INSUFFICIENT",
            "attempt_budget": 8, "attempts_started": attempts_started, "attempts_completed": attempts_completed,
            "retry_count": 0, "failure": {"type": type(exc).__name__, "message": str(exc)},
            "formal_result": False, "stage_progression_allowed": False, "ended_at": utc_now(),
        }
        (output / "audit.json").write_bytes(canonical_json_bytes(audit) + b"\n")
        raise
    audit = {
        "schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": "BASE_EXACT8_READY",
        "source": {"head": head, "tree": tree, "dirty": False},
        "config_sha256": sha256_file(config_path), "protocol_sha256": sha256_file(protocol_path),
        "plan_sha256": sha256_file(plan_path), "model": config["model"], "runtime": runtime,
        "attempt_budget": 8, "attempts_started": attempts_started, "attempts_completed": attempts_completed,
        "retry_count": 0, "artifacts": artifacts, "carrier_was_used": False, "manual_decode_was_used": False,
        "detector_was_run": False, "phase_c_was_run": False,
        "formal_result": False, "stage_progression_allowed": False,
        "started_at": started_at, "ended_at": utc_now(),
    }
    (output / "config.json").write_bytes(canonical_json_bytes(config) + b"\n")
    (output / "audit.json").write_bytes(canonical_json_bytes(audit) + b"\n")
    (output / "command.json").write_bytes(canonical_json_bytes({"argv": list(argv), "cwd": str(cwd.resolve()), "exit_code": 0}) + b"\n")
    checksum_files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "checksums.sha256").write_text(
        "\n".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in checksum_files) + "\n",
        encoding="utf-8",
    )
    return {"status": "BASE_EXACT8_READY", "diagnostic_class": DIAGNOSTIC_CLASS, "output": str(output),
            "attempts_started": 8, "attempts_completed": 8, "retry_count": 0}
