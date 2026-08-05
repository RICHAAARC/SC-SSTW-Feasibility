#!/usr/bin/env python3
"""Run the minimal SC-SSTW GPU/saved-video observation probe.

Local use defaults to ``--dry-run`` for CLI/schema validation without GPU. The
real path is intended for a user-run Colab GPU session and produces only
diagnostic JSON, not detection evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.gpu_observation import (  # noqa: E402
    CLAIM_BOUNDARY,
    DEFAULT_MODEL_ID,
    ObservationProbeConfig,
    dry_run_records,
    readout_q_from_rgb_frames,
    relation_prompts,
    summarize_observations,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/gpu_observation_probe", help="directory for JSON and optional videos")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed", type=int, default=1275)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--frame-count", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="do not import torch/diffusers; use deterministic mock videos")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ObservationProbeConfig(
        frame_count=args.frame_count,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seed=args.seed,
        model_id=args.model_id,
        dry_run=args.dry_run,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        records = dry_run_records() if args.dry_run else _real_gpu_records(config, output_dir)
        summary = summarize_observations(records)
        result = {
            "result_kind": "sc_sstw_gpu_observation_probe_v1",
            "claim_support_status": CLAIM_BOUNDARY,
            "config": config.__dict__,
            "records": records,
            "summary": summary,
        }
        write_json(output_dir / "gpu_observation_probe_result.json", result)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    except BaseException as exc:
        failure = {
            "result_kind": "sc_sstw_gpu_observation_probe_v1",
            "claim_support_status": CLAIM_BOUNDARY,
            "formal_result": False,
            "paper_claim": False,
            "probe_decision": "runtime_failure",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
        }
        write_json(output_dir / "gpu_observation_probe_failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
        return 1


def _real_gpu_records(config: ObservationProbeConfig, output_dir: Path) -> list[dict[str, Any]]:
    torch, pipeline_cls, export_to_video, imageio = _load_real_dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real GPU observation probe")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = pipeline_cls.from_pretrained(config.model_id, torch_dtype=dtype)
    pipe = pipe.to("cuda")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    records: list[dict[str, Any]] = []
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(relation_prompts()):
        generator = torch.Generator(device="cuda").manual_seed(config.seed + index)
        result = pipe(
            prompt=prompt["prompt"],
            negative_prompt="text, watermark, logo, camera shake",
            num_frames=config.frame_count,
            height=config.height,
            width=config.width,
            guidance_scale=5.0,
            num_inference_steps=8,
            generator=generator,
        )
        frames = _extract_frames(result)
        video_path = video_dir / f"{index:02d}_{prompt['probe_id']}.mp4"
        export_to_video(frames, str(video_path), fps=config.fps)
        saved_frames = imageio.imread(video_path)
        q = readout_q_from_rgb_frames(saved_frames)
        records.append(
            {
                "probe_id": prompt["probe_id"],
                "kind": prompt["kind"],
                "target_q": list(prompt["target_q"]),
                "q": list(q),
                "video_path": str(video_path),
            }
        )
    return records


def _load_real_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import ftfy  # noqa: F401
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
        import imageio.v3 as imageio
    except Exception as exc:  # pragma: no cover - exercised in Colab, not local CI
        raise RuntimeError("real mode requires torch, diffusers, ftfy, imageio, and imageio-ffmpeg") from exc
    return torch, WanPipeline, export_to_video, imageio


def _extract_frames(result: Any) -> list[Any]:
    frames = getattr(result, "frames", None)
    if frames is None and isinstance(result, dict):
        frames = result.get("frames")
    if frames is None:
        raise RuntimeError("pipeline result did not expose frames")
    if len(frames) > 0 and isinstance(frames[0], list):
        return list(frames[0])
    return list(frames)


if __name__ == "__main__":
    raise SystemExit(main())
