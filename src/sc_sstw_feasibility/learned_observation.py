"""Auditable low-capacity observation front-end for one saved MP4.

This module does not run Wan.  It freezes and implements the CPU-visible part
of the learned-observation contract: a fixed analytic feature map, a shared
530-parameter MLP, public-only relation training, key-independent AISB
acquisition, and calibration that can consume only a closed/read-back
ambiguity artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .aisb import BurstTemplate, best_non_overlapping_sequence, scan_burst_candidates
from .calibration import calibrate_from_pilot_pairs, equalize_observations
from .linalg import condition_number_2d_columns, matmul, transpose


class LearnedObservationError(RuntimeError):
    pass


PROTOCOL_ID = "sc_sstw_learned_observation_frontend"
CONFIG_CANONICAL_SHA256 = "43d24e6794badc4b5551387f61ff3086fdf397efb017afdae4ff2ac4ac1d77ad"
ACQUISITION_PROTOCOL_SHA256 = "d71fb9e125363b1834ec1da34280b73e49be2678f4a149ae68f9ca9f3f7f7369"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
TRAIN_IDS = (41001, 41002, 41003, 41004)
VALIDATION_IDS = (41005, 41006)
HELD_OUT_IDS = (41007, 41008)
NULL_IDS = tuple(range(41009, 41017))
L1_IDS = TRAIN_IDS + VALIDATION_IDS
L2_IDS = HELD_OUT_IDS + NULL_IDS
CALIBRATION_INDICES = (0, 1, 2, 3)
PER_VIDEO_HELD_OUT_INDICES = (4, 5)
FRAME_GROUPS = (
    (0,),
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
    (21, 22, 23, 24),
    (25, 26, 27, 28),
    (29, 30, 31, 32),
    (33, 34, 35, 36),
    (37, 38, 39, 40),
    (41, 42, 43, 44),
    (45, 46, 47, 48),
)
TEMPORAL_POINTS = (
    (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.25, 0.35),
    (0.75, 0.2), (0.2, 0.8), (-0.9238795325, 0.3826834324),
    (0.3826834324, 0.9238795325), (-0.3826834324, -0.9238795325),
    (0.9238795325, -0.3826834324), (-0.7071067812, 0.7071067812),
    (0.7071067812, 0.7071067812), (0.0, -1.0),
)
DCT_BASES = ((1, 0), (0, 1), (2, 0), (0, 2), (1, 1))
FORBIDDEN_MP4_SHA256 = frozenset({
    "98c9d096b24c22caac2ecaa9f25443cee2a5608dc4ce076439d3b6e8517a53d7",
    "cb118963716cb5c47078d498aed7dde8fee0d266d7b304727f7abb561478bef8",
})
FORBIDDEN_LOCAL_ROOTS = (
    Path("/tmp/sc_sstw_audit_20260807T022258Z/20260807T022258Z_cfff6893"),
    Path("/tmp/sc_sstw_internal_challenger_20260807T022258Z.tar.gz"),
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _finite_matrix(value: Any, rows: int, columns: int) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == rows
        and all(
            isinstance(row, Sequence)
            and not isinstance(row, (str, bytes))
            and len(row) == columns
            and all(math.isfinite(float(item)) for item in row)
            for row in value
        )
    )


def validate_learned_observation_config(config: dict[str, Any]) -> None:
    if sha256_bytes(canonical_json_bytes(config)) != CONFIG_CANONICAL_SHA256:
        raise LearnedObservationError("complete learned-observation config digest changed")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise LearnedObservationError("unexpected learned-observation protocol")
    if config.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise LearnedObservationError("model identity or revision changed")
    carrier = config["carrier"]
    if carrier["kind"] != "dit_internal_self_attention_output_residual":
        raise LearnedObservationError("internal attn1 output is the only admitted carrier")
    if carrier["module_path"] != "transformer.blocks[29].attn1" or carrier["block_index"] != 29:
        raise LearnedObservationError("carrier block changed")
    if carrier["target_relative_rms"] != 0.03 or carrier["target_relative_rms_absolute_tolerance"] != 0.00005:
        raise LearnedObservationError("carrier energy changed")
    if tuple(tuple(float(x) for x in point) for point in carrier["temporal_points"]) != TEMPORAL_POINTS:
        raise LearnedObservationError("temporal q schedule changed")
    if tuple(carrier["public_calibration_indices"]) != CALIBRATION_INDICES:
        raise LearnedObservationError("public calibration split changed")
    if tuple(carrier["per_video_calibration_held_out_indices"]) != PER_VIDEO_HELD_OUT_INDICES:
        raise LearnedObservationError("per-video calibration held-out split changed")
    generation = config["generation"]
    expected_generation = {"fps": 8, "frame_count": 49, "height": 320, "width": 512, "inference_steps": 8, "guidance_scale": 5.0}
    if any(generation.get(key) != value for key, value in expected_generation.items()):
        raise LearnedObservationError("generation geometry changed")
    if tuple(tuple(group) for group in generation["temporal_frame_groups"]) != FRAME_GROUPS:
        raise LearnedObservationError("49-to-13 frame mapping changed")
    if [item for group in FRAME_GROUPS for item in group] != list(range(49)):
        raise LearnedObservationError("frame groups do not partition the saved video")
    video_io = config["video_io"]
    if video_io["decoder"] != {
        "call": "imageio.v3.imiter", "plugin": "FFMPEG", "fallback_permitted": False,
        "required_dtype": "uint8", "required_shape": [49, 320, 512, 3],
    }:
        raise LearnedObservationError("decoder contract changed")
    encoder = video_io["encoder"]
    expected_encoder = {
        "call": "diffusers.utils.export_to_video", "fps": 8, "quality": 5.0,
        "bitrate": None, "macro_block_size": 16, "opencv_fallback_permitted": False,
        "required_codec_name": "h264", "required_pixel_format": "yuv420p",
    }
    if encoder != expected_encoder:
        raise LearnedObservationError("encoder contract changed")
    features = config["features"]
    if features["input"] != "single_saved_mp4_only" or features["feature_dimension"] != 30:
        raise LearnedObservationError("feature input or dimension changed")
    if tuple(tuple(pair) for pair in features["dct_basis_order"]) != DCT_BASES:
        raise LearnedObservationError("DCT basis changed")
    extractor = config["extractor"]
    if extractor["parameter_count"] != 530 or extractor["position_encoding"] or extractor["cross_window_connections"]:
        raise LearnedObservationError("extractor capacity or clock path changed")
    if extractor["forward_input"] != "standardized_feature_matrix_13_by_30_only":
        raise LearnedObservationError("extractor forward boundary changed")
    training = config["training"]
    if tuple(training["full_batch_video_order"]) != TRAIN_IDS or training["optimizer_step_count"] != 2000:
        raise LearnedObservationError("training order or step count changed")
    if training["checkpoint_selection"] != "step_2000_only" or training["early_stopping"]:
        raise LearnedObservationError("checkpoint selection changed")
    if tuple(training["inner_affine_fit"]["fit_indices"]) != CALIBRATION_INDICES:
        raise LearnedObservationError("training affine fit uses non-calibration points")
    if tuple(training["outer_relation_loss"]["indices"]) != PER_VIDEO_HELD_OUT_INDICES:
        raise LearnedObservationError("training outer relation indices changed")
    items = config["dataset"]["items"]
    ids = tuple(int(item["dataset_id"]) for item in items)
    if ids != tuple(range(41001, 41017)) or any(int(item["seed"]) != int(item["dataset_id"]) for item in items):
        raise LearnedObservationError("dataset IDs, seeds, or order changed")
    split_ids = {
        split: tuple(int(item["dataset_id"]) for item in items if item["split"] == split)
        for split in ("train", "validation", "held_out", "null")
    }
    if split_ids != {"train": TRAIN_IDS, "validation": VALIDATION_IDS, "held_out": HELD_OUT_IDS, "null": NULL_IDS}:
        raise LearnedObservationError("dataset split changed")
    protocol = config["acquisition"]["protocol"]
    actual_protocol_sha = sha256_bytes(canonical_json_bytes(protocol))
    if config["acquisition"]["protocol_sha256"] != ACQUISITION_PROTOCOL_SHA256 or actual_protocol_sha != ACQUISITION_PROTOCOL_SHA256:
        raise LearnedObservationError("AISB acquisition protocol digest changed")
    thresholds = config["gate_thresholds"]
    expected_thresholds = {
        "validation_required_pass_count": 2, "validation_case_count": 2,
        "held_out_required_pass_count": 2, "held_out_case_count": 2,
        "null_required_false_acquisition_count": 0, "null_case_count": 8,
        "maximum_aisb_residual": 0.25,
        "minimum_global_centered_output_second_singular_value": 0.1,
        "minimum_fitted_affine_second_singular_value": 0.05,
        "maximum_fitted_affine_condition_number": 10.0,
        "maximum_public_calibration_held_out_mse": 0.02,
    }
    if thresholds != expected_thresholds:
        raise LearnedObservationError("Gate thresholds changed")
    if config["software"] != {"diffusers": "0.35.2", "imageio": "2.37.0", "imageio_ffmpeg": "0.6.0"}:
        raise LearnedObservationError("video dependency versions changed")


def assert_stage_dataset_access(stage: str, dataset_ids: Sequence[int], *, l1_gate_pass: bool = False) -> None:
    ids = tuple(int(value) for value in dataset_ids)
    if stage == "gpu_train_validation":
        if ids != L1_IDS:
            raise LearnedObservationError("L1 requires the exact frozen ID sequence")
    elif stage == "gpu_public_held_out_and_null":
        if not l1_gate_pass:
            raise LearnedObservationError("L2 access requires an audited L1 pass")
        if ids != L2_IDS:
            raise LearnedObservationError("L2 requires the exact frozen ID sequence")
    else:
        raise LearnedObservationError("unknown dataset access stage")


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _valid_git_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def validate_new_dataset_artifact(
    path: Path,
    manifest_path: Path,
    config: dict[str, Any],
    *,
    expected_repository_commit: str,
    expected_manifest_sha256: str,
) -> None:
    validate_learned_observation_config(config)
    resolved = path.resolve()
    if any(resolved == root or root in resolved.parents for root in FORBIDDEN_LOCAL_ROOTS):
        raise LearnedObservationError("quarantined failed-run path is forbidden")
    if not path.is_file():
        raise LearnedObservationError("dataset artifact is not a file")
    if sha256_file(path) in FORBIDDEN_MP4_SHA256:
        raise LearnedObservationError("quarantined failed-run MP4 bytes are forbidden")
    manifest_resolved = manifest_path.resolve()
    if any(manifest_resolved == root or root in manifest_resolved.parents for root in FORBIDDEN_LOCAL_ROOTS):
        raise LearnedObservationError("quarantined manifest path is forbidden")
    if not manifest_path.is_file() or not _valid_sha256(expected_manifest_sha256):
        raise LearnedObservationError("frozen generation manifest evidence is missing")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise LearnedObservationError("generation manifest differs from its externally frozen digest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {
        "protocol_id", "config_sha256", "repository_commit", "dataset_id", "prompt",
        "seed", "carrier", "artifact_sha256", "derived_from", "generation_call",
        "injection_records_sha256",
    }
    if set(manifest) != expected_keys:
        raise LearnedObservationError("generation manifest schema changed")
    if manifest["protocol_id"] != PROTOCOL_ID or manifest["config_sha256"] != CONFIG_CANONICAL_SHA256:
        raise LearnedObservationError("generation manifest config binding changed")
    if not _valid_git_commit(expected_repository_commit) or manifest["repository_commit"] != expected_repository_commit:
        raise LearnedObservationError("generation manifest repository commit mismatch")
    if manifest["generation_call"] != "repository_formal_cli" or manifest["derived_from"] is not None:
        raise LearnedObservationError("old, derived, or non-formal generation is forbidden")
    dataset_id = int(manifest.get("dataset_id", -1))
    item = next((item for item in config["dataset"]["items"] if int(item["dataset_id"]) == dataset_id), None)
    if item is None:
        raise LearnedObservationError("dataset ID is not frozen")
    for key in ("prompt", "seed", "carrier"):
        if manifest.get(key) != item[key]:
            raise LearnedObservationError(f"generation manifest {key} mismatch")
    if manifest.get("artifact_sha256") != sha256_file(path):
        raise LearnedObservationError("dataset artifact digest mismatch")
    injection_sha = manifest["injection_records_sha256"]
    if item["carrier"] == "watermarked" and not _valid_sha256(injection_sha):
        raise LearnedObservationError("watermarked generation lacks injection-record binding")
    if item["carrier"] == "clean" and injection_sha is not None:
        raise LearnedObservationError("clean null must not claim injection records")


def decode_saved_mp4(mp4_path: Path) -> Any:
    """Decode one MP4 through the sole admitted imageio FFMPEG path."""

    try:
        import imageio
        import imageio.v3 as iio
        import imageio_ffmpeg
        import numpy as np
    except Exception as exc:
        raise LearnedObservationError("locked imageio decoder dependencies are unavailable") from exc
    versions = (imageio.__version__, imageio_ffmpeg.__version__)
    if versions != ("2.37.0", "0.6.0"):
        raise LearnedObservationError(f"decoder dependency mismatch: {versions}")
    frames = np.stack(list(iio.imiter(mp4_path, plugin="FFMPEG")), axis=0)
    if str(frames.dtype) != "uint8" or list(frames.shape) != [49, 320, 512, 3]:
        raise LearnedObservationError(f"decoded MP4 shape/dtype mismatch: {frames.shape} {frames.dtype}")
    return frames


def encode_saved_mp4(frames: Any, mp4_path: Path) -> Path:
    """Encode through the exact cfff689 diffusers/imageio call, with no fallback."""

    try:
        import diffusers
        import imageio
        import imageio_ffmpeg
        from diffusers.utils import export_to_video
    except Exception as exc:
        raise LearnedObservationError("locked diffusers/imageio encoder dependencies are unavailable") from exc
    versions = (diffusers.__version__, imageio.__version__, imageio_ffmpeg.__version__)
    if versions != ("0.35.2", "2.37.0", "0.6.0"):
        raise LearnedObservationError(f"encoder dependency mismatch: {versions}")
    result = export_to_video(frames, str(mp4_path), fps=8, quality=5.0, bitrate=None, macro_block_size=16)
    if Path(result) != mp4_path or not mp4_path.is_file():
        raise LearnedObservationError("encoder did not create the requested MP4")
    return mp4_path


def _dct_scale(index: int, length: int) -> float:
    return 1.0 / math.sqrt(length) if index == 0 else math.sqrt(2.0 / length)


def extract_feature_matrix(frames: Any) -> list[list[float]]:
    """Return the frozen 13 x 30 analytic feature matrix."""

    try:
        import numpy as np
    except Exception as exc:
        raise LearnedObservationError("numpy is required for saved-MP4 feature extraction") from exc
    array = np.asarray(frames)
    if str(array.dtype) != "uint8" or list(array.shape) != [49, 320, 512, 3]:
        raise LearnedObservationError(f"feature input shape/dtype mismatch: {array.shape} {array.dtype}")
    rgb = array.astype(np.float64) / 255.0
    channels = np.stack(
        [
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2],
            -0.168736 * rgb[..., 0] - 0.331264 * rgb[..., 1] + 0.5 * rgb[..., 2],
            0.5 * rgb[..., 0] - 0.418688 * rgb[..., 1] - 0.081312 * rgb[..., 2],
        ],
        axis=-1,
    )
    height, width = 320, 512
    yy = np.arange(height, dtype=np.float64)
    xx = np.arange(width, dtype=np.float64)
    bases = []
    for u, v in DCT_BASES:
        y_basis = _dct_scale(u, height) * np.cos(math.pi * (2.0 * yy + 1.0) * u / (2.0 * height))
        x_basis = _dct_scale(v, width) * np.cos(math.pi * (2.0 * xx + 1.0) * v / (2.0 * width))
        bases.append(y_basis[:, None] * x_basis[None, :])
    coefficients = np.empty((49, 3, 5), dtype=np.float64)
    for frame_index in range(49):
        for channel_index in range(3):
            for basis_index, basis in enumerate(bases):
                coefficients[frame_index, channel_index, basis_index] = float(np.sum(channels[frame_index, :, :, channel_index] * basis))
    result: list[list[float]] = []
    for group in FRAME_GROUPS:
        group_values = coefficients[list(group)]
        row: list[float] = []
        for channel_index in range(3):
            for basis_index in range(5):
                values = group_values[:, channel_index, basis_index]
                mean = float(values.mean())
                if len(group) == 1:
                    slope = 0.0
                else:
                    positions = np.arange(len(group), dtype=np.float64)
                    centered_positions = positions - positions.mean()
                    slope = float(np.sum(centered_positions * (values - mean)) / np.sum(centered_positions**2))
                row.extend((mean, slope))
        result.append(row)
    if not _finite_matrix(result, 13, 30):
        raise LearnedObservationError("analytic feature matrix is not finite 13 x 30")
    return result


def fit_train_only_normalizer(features_by_dataset_id: Mapping[int, Sequence[Sequence[float]]]) -> tuple[list[float], list[float]]:
    if tuple(features_by_dataset_id) != TRAIN_IDS:
        raise LearnedObservationError("normalizer must see exactly train IDs 41001 through 41004 in order")
    rows: list[list[float]] = []
    for dataset_id in TRAIN_IDS:
        matrix = features_by_dataset_id[dataset_id]
        if not _finite_matrix(matrix, 13, 30):
            raise LearnedObservationError(f"train feature matrix {dataset_id} is not finite 13 x 30")
        rows.extend([[float(value) for value in row] for row in matrix])
    means = [sum(row[column] for row in rows) / 52.0 for column in range(30)]
    stds = [
        max(math.sqrt(sum((row[column] - means[column]) ** 2 for row in rows) / 52.0), 1e-6)
        for column in range(30)
    ]
    return means, stds


@dataclass(frozen=True)
class FrozenObservationFrontend:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    first_weight: tuple[tuple[float, ...], ...]
    first_bias: tuple[float, ...]
    second_weight: tuple[tuple[float, ...], ...]
    second_bias: tuple[float, ...]
    artifact_sha256: str

    def forward(self, standardized_feature_matrix: Sequence[Sequence[float]]) -> list[list[float]]:
        if not _finite_matrix(standardized_feature_matrix, 13, 30):
            raise LearnedObservationError("forward input must be finite 13 x 30 features")
        output: list[list[float]] = []
        for row in standardized_feature_matrix:
            hidden = [
                math.tanh(sum(float(row[index]) * weights[index] for index in range(30)) + bias)
                for weights, bias in zip(self.first_weight, self.first_bias, strict=True)
            ]
            value = [
                sum(hidden[index] * weights[index] for index in range(16)) + bias
                for weights, bias in zip(self.second_weight, self.second_bias, strict=True)
            ]
            output.append(value)
        if not _finite_matrix(output, 13, 2):
            raise LearnedObservationError("forward output is not finite 13 x 2")
        return output

    def observe_saved_mp4(self, mp4_path: Path) -> list[list[float]]:
        features = extract_feature_matrix(decode_saved_mp4(mp4_path))
        standardized = [
            [(float(row[column]) - self.mean[column]) / self.std[column] for column in range(30)]
            for row in features
        ]
        return self.forward(standardized)


def load_frozen_frontend(
    path: Path,
    *,
    expected_weights_sha256: str,
    expected_config_sha256: str,
) -> FrozenObservationFrontend:
    raw = path.read_bytes()
    if not _valid_sha256(expected_weights_sha256) or sha256_bytes(raw) != expected_weights_sha256:
        raise LearnedObservationError("weights differ from the externally frozen digest")
    if expected_config_sha256 != CONFIG_CANONICAL_SHA256:
        raise LearnedObservationError("weights caller supplied the wrong config digest")
    payload = json.loads(raw.decode("utf-8"))
    expected_payload_keys = {"artifact_kind", "config_sha256", "train_artifact_sha256_by_dataset_id", "normalizer_fit_dataset_ids", "optimizer_steps", "final_training_loss", "normalizer", "model"}
    if set(payload) != expected_payload_keys or payload.get("artifact_kind") != "public_relation_observation_frontend_step_2000":
        raise LearnedObservationError("weights artifact schema or kind changed")
    if payload.get("config_sha256") != expected_config_sha256:
        raise LearnedObservationError("weights artifact is bound to a different config")
    train_digests = payload.get("train_artifact_sha256_by_dataset_id")
    if not isinstance(train_digests, dict) or tuple(train_digests) != tuple(map(str, TRAIN_IDS)) or not all(_valid_sha256(value) for value in train_digests.values()):
        raise LearnedObservationError("weights lack exact ordered train-artifact digests")
    if payload.get("optimizer_steps") != 2000 or payload.get("normalizer_fit_dataset_ids") != list(TRAIN_IDS):
        raise LearnedObservationError("weights were not frozen from the exact training stage")
    model = payload["model"]
    if set(payload["normalizer"]) != {"mean", "std"} or set(model) != {"first_weight", "first_bias", "second_weight", "second_bias"}:
        raise LearnedObservationError("weights model or normalizer schema changed")
    mean, std = payload["normalizer"]["mean"], payload["normalizer"]["std"]
    if len(mean) != 30 or len(std) != 30 or any(not math.isfinite(float(value)) for value in mean) or any(not math.isfinite(float(value)) or float(value) < 1e-6 for value in std):
        raise LearnedObservationError("invalid frozen normalizer")
    if len(model["first_weight"]) != 16 or any(len(row) != 30 for row in model["first_weight"]) or len(model["first_bias"]) != 16:
        raise LearnedObservationError("invalid first layer")
    if len(model["second_weight"]) != 2 or any(len(row) != 16 for row in model["second_weight"]) or len(model["second_bias"]) != 2:
        raise LearnedObservationError("invalid second layer")
    numeric_values = [*model["first_bias"], *model["second_bias"], *(value for row in model["first_weight"] for value in row), *(value for row in model["second_weight"] for value in row)]
    if not math.isfinite(float(payload["final_training_loss"])) or any(not math.isfinite(float(value)) for value in numeric_values):
        raise LearnedObservationError("weights artifact contains non-finite values")
    return FrozenObservationFrontend(
        mean=tuple(map(float, mean)), std=tuple(map(float, std)),
        first_weight=tuple(tuple(map(float, row)) for row in model["first_weight"]),
        first_bias=tuple(map(float, model["first_bias"])),
        second_weight=tuple(tuple(map(float, row)) for row in model["second_weight"]),
        second_bias=tuple(map(float, model["second_bias"])),
        artifact_sha256=sha256_bytes(raw),
    )


def train_public_relation_frontend(
    features_by_dataset_id: Mapping[int, Sequence[Sequence[float]]],
    train_artifact_sha256_by_dataset_id: Mapping[int, str],
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Fit exactly one final public relation model; no checkpoint selection."""

    validate_learned_observation_config(config)
    means, stds = fit_train_only_normalizer(features_by_dataset_id)
    if tuple(train_artifact_sha256_by_dataset_id) != TRAIN_IDS or not all(_valid_sha256(value) for value in train_artifact_sha256_by_dataset_id.values()):
        raise LearnedObservationError("training requires exact ordered train-artifact digests")
    try:
        import torch
    except Exception as exc:
        raise LearnedObservationError("torch is required for the frozen CPU training routine") from exc
    torch.manual_seed(20260807)
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Sequential(torch.nn.Linear(30, 16, bias=True), torch.nn.Tanh(), torch.nn.Linear(16, 2, bias=True)).double()
    torch.nn.init.xavier_uniform_(model[0].weight, gain=1.0)
    torch.nn.init.zeros_(model[0].bias)
    torch.nn.init.xavier_uniform_(model[2].weight, gain=1.0)
    torch.nn.init.zeros_(model[2].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    feature_tensors = []
    for dataset_id in TRAIN_IDS:
        standardized = [[(float(row[column]) - means[column]) / stds[column] for column in range(30)] for row in features_by_dataset_id[dataset_id]]
        feature_tensors.append(torch.tensor(standardized, dtype=torch.float64))
    q = torch.tensor(TEMPORAL_POINTS, dtype=torch.float64)
    calibration_indices = torch.tensor(CALIBRATION_INDICES, dtype=torch.int64)
    held_out_indices = torch.tensor(PER_VIDEO_HELD_OUT_INDICES, dtype=torch.int64)
    ridge = torch.diag(torch.tensor([1e-8, 1e-8, 0.0], dtype=torch.float64))
    optimizer_steps = 0
    final_loss = math.inf
    for _step in range(2000):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for features in feature_tensors:
            y = model(features)
            z = torch.cat((q, torch.ones((13, 1), dtype=torch.float64)), dim=1)
            z_c, y_c = z.index_select(0, calibration_indices), y.index_select(0, calibration_indices)
            b = torch.linalg.solve(z_c.T @ z_c + ridge, z_c.T @ y_c)
            y_h, z_h = y.index_select(0, held_out_indices), z.index_select(0, held_out_indices)
            relation = torch.sum((y_h - z_h @ b) ** 2) / (torch.sum((y_h - y_h.mean(dim=0, keepdim=True)) ** 2) + 1e-6)
            singular = torch.linalg.svdvals((y - y.mean(dim=0, keepdim=True)) / math.sqrt(13.0))
            affine_singular = torch.linalg.svdvals(b[:2, :].T)
            rank_loss = torch.relu(torch.tensor(0.10, dtype=torch.float64) - singular[1]) ** 2
            scale_loss = torch.relu(torch.tensor(0.05, dtype=torch.float64) - affine_singular[1]) ** 2
            condition_loss = torch.relu(affine_singular[0] / torch.clamp(affine_singular[1], min=1e-8) - 10.0) ** 2
            losses.append(relation + 10.0 * rank_loss + 10.0 * scale_loss + 0.1 * condition_loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer_steps += 1
        final_loss = float(loss.detach().item())
    if optimizer_steps != 2000:
        raise LearnedObservationError("training did not execute exactly 2000 full-batch updates")
    payload = {
        "artifact_kind": "public_relation_observation_frontend_step_2000",
        "config_sha256": CONFIG_CANONICAL_SHA256,
        "train_artifact_sha256_by_dataset_id": {str(dataset_id): train_artifact_sha256_by_dataset_id[dataset_id] for dataset_id in TRAIN_IDS},
        "normalizer_fit_dataset_ids": list(TRAIN_IDS),
        "optimizer_steps": optimizer_steps,
        "final_training_loss": final_loss,
        "normalizer": {"mean": means, "std": stds},
        "model": {
            "first_weight": model[0].weight.detach().tolist(), "first_bias": model[0].bias.detach().tolist(),
            "second_weight": model[2].weight.detach().tolist(), "second_bias": model[2].bias.detach().tolist(),
        },
    }
    with output_path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload) + b"\n")
        handle.flush()
    return {"weights_sha256": sha256_file(output_path), "optimizer_steps": optimizer_steps, "final_training_loss": final_loss}


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    return {
        "start_index": int(candidate.start_index), "template_id": candidate.template_id,
        "residual": float(candidate.residual), "observed_length": int(candidate.observed_length),
        "missing_template_index": candidate.missing_template_index,
    }


def acquire_and_freeze_ambiguity(
    observation: Sequence[Sequence[float]],
    artifact_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Acquire without truth/key, then write-close-readback-verify the artifact."""

    validate_learned_observation_config(config)
    if not _finite_matrix(observation, 13, 2):
        raise LearnedObservationError("AISB observation must be finite 13 x 2")
    template = BurstTemplate("burst_alpha", tuple(TEMPORAL_POINTS[:6]))
    candidates = scan_burst_candidates(
        [list(map(float, row)) for row in observation], (template,), top_k_per_start=1,
        allow_single_deletion=False, allow_double_deletion=False,
    )
    accepted = best_non_overlapping_sequence(candidates, burst_length=6, residual_threshold=0.25, maximize_count=False)
    payload = {
        "protocol_sha256": ACQUISITION_PROTOCOL_SHA256,
        "observation_sha256": sha256_bytes(canonical_json_bytes([[float(value) for value in row] for row in observation])),
        "accepted_candidates": [_candidate_payload(candidate) for candidate in accepted],
    }
    payload_sha = sha256_bytes(canonical_json_bytes(payload))
    envelope = {"payload": payload, "payload_sha256": payload_sha}
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
    artifact_sha256 = sha256_file(artifact_path)
    readback = read_frozen_ambiguity(artifact_path, expected_artifact_sha256=artifact_sha256)
    if readback != envelope:
        raise LearnedObservationError("closed ambiguity artifact readback changed")
    return {"artifact_sha256": artifact_sha256, "envelope": readback}


def read_frozen_ambiguity(artifact_path: Path, *, expected_artifact_sha256: str) -> dict[str, Any]:
    if not _valid_sha256(expected_artifact_sha256) or sha256_file(artifact_path) != expected_artifact_sha256:
        raise LearnedObservationError("ambiguity artifact differs from its externally frozen digest")
    envelope = json.loads(artifact_path.read_text(encoding="utf-8"))
    if set(envelope) != {"payload", "payload_sha256"}:
        raise LearnedObservationError("ambiguity artifact envelope changed")
    if sha256_bytes(canonical_json_bytes(envelope["payload"])) != envelope["payload_sha256"]:
        raise LearnedObservationError("ambiguity artifact digest mismatch")
    if envelope["payload"].get("protocol_sha256") != ACQUISITION_PROTOCOL_SHA256:
        raise LearnedObservationError("ambiguity artifact protocol mismatch")
    return envelope


def _second_singular_value(matrix: Sequence[Sequence[float]]) -> float:
    ata = matmul(transpose([[float(value) for value in row] for row in matrix]), [[float(value) for value in row] for row in matrix])
    trace = ata[0][0] + ata[1][1]
    determinant = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0]
    discriminant = max(0.0, trace * trace - 4.0 * determinant)
    lambda_min = max(0.0, 0.5 * (trace - math.sqrt(discriminant)))
    return math.sqrt(lambda_min)


def calibrate_from_frozen_ambiguity(
    observation: Sequence[Sequence[float]],
    artifact_path: Path,
    expected_artifact_sha256: str,
    config: dict[str, Any],
) -> dict[str, float]:
    """Calibrate from C only, consuming only a verified read-back artifact."""

    validate_learned_observation_config(config)
    if not _finite_matrix(observation, 13, 2):
        raise LearnedObservationError("calibration observation must be finite 13 x 2")
    envelope = read_frozen_ambiguity(artifact_path, expected_artifact_sha256=expected_artifact_sha256)
    expected_observation_sha = sha256_bytes(canonical_json_bytes([[float(value) for value in row] for row in observation]))
    if envelope["payload"]["observation_sha256"] != expected_observation_sha:
        raise LearnedObservationError("calibration observation differs from frozen acquisition evidence")
    accepted = envelope["payload"]["accepted_candidates"]
    if len(accepted) != 1:
        raise LearnedObservationError("calibration requires one frozen accepted candidate")
    start = int(accepted[0]["start_index"])
    pairs = [(TEMPORAL_POINTS[index], list(map(float, observation[start + index]))) for index in CALIBRATION_INDICES]
    calibration = calibrate_from_pilot_pairs(pairs)
    held_observations = [list(map(float, observation[start + index])) for index in PER_VIDEO_HELD_OUT_INDICES]
    equalized = equalize_observations(held_observations, calibration)
    mse = sum(
        (equalized[offset][dimension] - TEMPORAL_POINTS[index][dimension]) ** 2
        for offset, index in enumerate(PER_VIDEO_HELD_OUT_INDICES)
        for dimension in range(2)
    ) / 4.0
    affine_second = _second_singular_value(calibration.matrix)
    centered_mean = [sum(float(row[d]) for row in observation) / 13.0 for d in range(2)]
    centered = [[(float(row[d]) - centered_mean[d]) / math.sqrt(13.0) for d in range(2)] for row in observation]
    return {
        "public_calibration_held_out_mse": mse,
        "fitted_affine_condition_number": condition_number_2d_columns(calibration.matrix),
        "fitted_affine_second_singular_value": affine_second,
        "global_centered_output_second_singular_value": _second_singular_value(centered),
        "accepted_aisb_residual": float(accepted[0]["residual"]),
    }


def audit_truth_success_after_freeze(artifact_path: Path, expected_artifact_sha256: str, config: dict[str, Any]) -> bool:
    validate_learned_observation_config(config)
    accepted = read_frozen_ambiguity(artifact_path, expected_artifact_sha256=expected_artifact_sha256)["payload"]["accepted_candidates"]
    if len(accepted) != 1:
        return False
    candidate = accepted[0]
    expected = config["acquisition"]["protocol"]["truth_success"]["exact_candidate"]
    return all(candidate.get(key) == value for key, value in expected.items())


def watermarked_gate_checks(
    observation: Sequence[Sequence[float]],
    artifact_path: Path,
    config: dict[str, Any],
) -> dict[str, bool | float]:
    """Run acquisition first; truth is consulted only after freeze and calibration."""

    frozen = acquire_and_freeze_ambiguity(observation, artifact_path, config)
    expected_artifact_sha256 = frozen["artifact_sha256"]
    metrics = calibrate_from_frozen_ambiguity(observation, artifact_path, expected_artifact_sha256, config)
    truth_success = audit_truth_success_after_freeze(artifact_path, expected_artifact_sha256, config)
    thresholds = config["gate_thresholds"]
    checks = {
        "finite_13_by_2": _finite_matrix(observation, 13, 2),
        "aisb_residual_at_most_0_25": metrics["accepted_aisb_residual"] <= thresholds["maximum_aisb_residual"],
        "truth_acquisition_success": truth_success,
        "global_centered_output_second_singular_value_at_least_0_10": metrics["global_centered_output_second_singular_value"] >= thresholds["minimum_global_centered_output_second_singular_value"],
        "fitted_affine_second_singular_value_at_least_0_05": metrics["fitted_affine_second_singular_value"] >= thresholds["minimum_fitted_affine_second_singular_value"],
        "fitted_affine_condition_number_at_most_10": metrics["fitted_affine_condition_number"] <= thresholds["maximum_fitted_affine_condition_number"],
        "public_calibration_held_out_mse_at_most_0_02": metrics["public_calibration_held_out_mse"] <= thresholds["maximum_public_calibration_held_out_mse"],
    }
    return {**metrics, **checks, "case_pass": all(checks.values())}


def null_has_false_acquisition(observation: Sequence[Sequence[float]], artifact_path: Path, config: dict[str, Any]) -> bool:
    frozen = acquire_and_freeze_ambiguity(observation, artifact_path, config)
    return bool(frozen["envelope"]["payload"]["accepted_candidates"])


def static_contract_report(config: dict[str, Any]) -> dict[str, Any]:
    validate_learned_observation_config(config)
    forward_parameters = tuple(inspect.signature(FrozenObservationFrontend.forward).parameters)
    observe_parameters = tuple(inspect.signature(FrozenObservationFrontend.observe_saved_mp4).parameters)
    checks = {
        "complete_config_sha256_exact": sha256_bytes(canonical_json_bytes(config)) == CONFIG_CANONICAL_SHA256,
        "acquisition_protocol_sha256_exact": sha256_bytes(canonical_json_bytes(config["acquisition"]["protocol"])) == ACQUISITION_PROTOCOL_SHA256,
        "forward_signature_features_only": forward_parameters == ("self", "standardized_feature_matrix"),
        "observe_signature_single_mp4_only": observe_parameters == ("self", "mp4_path"),
        "normalizer_train_ids_exact": tuple(config["training"]["full_batch_video_order"]) == TRAIN_IDS,
        "l1_and_l2_dataset_ids_disjoint": not (set(L1_IDS) & set(L2_IDS)),
        "calibration_and_per_video_held_out_disjoint": not (set(CALIBRATION_INDICES) & set(PER_VIDEO_HELD_OUT_INDICES)),
        "decoder_unique_no_fallback": config["video_io"]["decoder"]["call"] == "imageio.v3.imiter" and config["video_io"]["decoder"]["plugin"] == "FFMPEG" and not config["video_io"]["decoder"]["fallback_permitted"],
        "optimizer_steps_exact": config["training"]["optimizer_step_count"] == 2000,
        "old_mp4_hashes_quarantined": set(config["quarantine"]["forbidden_mp4_sha256"]) == set(FORBIDDEN_MP4_SHA256),
    }
    return {"gate": "cpu_static_implementation", "checks": checks, "gate_pass": all(checks.values()), "gpu_admission": False}
