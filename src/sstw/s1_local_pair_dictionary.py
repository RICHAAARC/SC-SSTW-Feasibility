"""Fast S1 local single-pair dictionary screen on real Wan OFF Q/K."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .relation_injector import PAIR_COEFFICIENTS, TARGET_QUERY_INDICES, _apply_rotary
from .s1_real_dit_relation_primitive import (
    DIAGNOSTIC_CLASS,
    S1InstrumentationError,
    _cosine,
    _git,
    _json_safe_ratio,
    _load_runtime,
    _rms,
    canonical_json_bytes,
    load_frozen_inputs,
    sha256_file,
    sha256_parameters,
    sha256_tensor,
    transformer_forward_inference,
)


SCHEMA = "sstw.s1.local_pair_dictionary.v1"
PLAN_SCHEMA = "sstw.s1.local_pair_dictionary.plan.v1"
AXIS_ORDER = ("B1_horizontal", "B2_vertical")
BRANCH_ORDER = ("cond", "uncond")
RADII = tuple(range(1, 9))
STATUS_READY = "PAIR_DICTIONARY_READY"
STATUS_NO_GO = "PAIR_DICTIONARY_NO_GO"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def pair_indices(axis: str, radius: int) -> tuple[tuple[int, int], ...]:
    if axis not in AXIS_ORDER or radius not in RADII:
        raise S1InstrumentationError("pair axis or radius is outside the frozen dictionary")
    stride = 1 if axis == "B1_horizontal" else 32
    pairs = tuple((query - stride * radius, query + stride * radius) for query in TARGET_QUERY_INDICES)
    for query, pair in zip(TARGET_QUERY_INDICES, pairs, strict=True):
        if not (0 <= pair[0] < 8320 and 0 <= pair[1] < 8320 and pair[0] < query < pair[1]):
            raise S1InstrumentationError("dictionary pair is outside the Wan token grid")
    return pairs


def _capture_pair_probabilities(probabilities: Any, torch: Any) -> dict[str, dict[str, list[Any]]]:
    if tuple(probabilities.shape) != (1, 12, 13, 8320):
        raise S1InstrumentationError("manual OFF probabilities must be exact [1,12,13,8320]")
    result: dict[str, dict[str, list[Any]]] = {}
    for axis in AXIS_ORDER:
        result[axis] = {}
        for radius in RADII:
            rows = []
            for row_offset, pair in enumerate(pair_indices(axis, radius)):
                rows.append(torch.stack((probabilities[0, :, row_offset, pair[0]], probabilities[0, :, row_offset, pair[1]]), dim=-1))
            values = torch.stack(rows, dim=0)
            if tuple(values.shape) != (13, 12, 2) or not bool(torch.isfinite(values).all().item()):
                raise S1InstrumentationError("captured pair probabilities are invalid")
            result[axis][str(radius)] = values.detach().float().cpu().tolist()
    return result


class PairDictionaryWanProcessor:
    """Exact Wan attention output plus optional OFF local-probability capture."""

    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.context: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def set_context(self, *, call_index: int, scheduler_index: int, timestep: int, branch: str, capture: bool) -> None:
        if branch not in BRANCH_ORDER:
            raise S1InstrumentationError("dictionary branch changed")
        self.context = {
            "call_index": int(call_index),
            "scheduler_index": int(scheduler_index),
            "timestep": int(timestep),
            "branch": branch,
            "capture": bool(capture),
        }

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        rotary_emb: Any = None,
    ) -> Any:
        if self.context is None:
            raise S1InstrumentationError("dictionary processor context was not set")
        if encoder_hidden_states is not None or attention_mask is not None or rotary_emb is None:
            raise S1InstrumentationError("dictionary processor only supports frozen Wan self-attention")
        if getattr(attn, "fused_projections", False):
            raise S1InstrumentationError("fused Wan projections are outside the dictionary path")
        try:
            from diffusers.models.attention_dispatch import dispatch_attention_fn
        except Exception as exc:  # pragma: no cover - production dependency boundary
            raise S1InstrumentationError("diffusers attention dispatcher unavailable") from exc

        torch = self.torch
        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        query = _apply_rotary(torch, query, rotary_emb)
        key = _apply_rotary(torch, key, rotary_emb)
        if tuple(query.shape) != (1, 8320, 12, 128) or tuple(key.shape) != tuple(query.shape) or tuple(value.shape) != tuple(query.shape):
            raise S1InstrumentationError("dictionary QKV identity is not [1,8320,12,128]")

        full_heads = dispatch_attention_fn(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None
        )
        record = dict(self.context)
        record.update(
            {
                "qkv_shape": [1, 8320, 12, 128],
                "qkv_dtype": str(query.dtype),
                "grad_enabled": bool(torch.is_grad_enabled()),
                "inference_mode_enabled": bool(torch.is_inference_mode_enabled()),
                "output_replaced": False,
                "pair_probabilities": None,
            }
        )
        if self.context["capture"]:
            query_index = torch.tensor(TARGET_QUERY_INDICES, device=query.device, dtype=torch.long)
            target_query = query.index_select(1, query_index).float()
            logits = torch.einsum("bqhd,bkhd->bhqk", target_query, key.float()) / math.sqrt(128.0)
            probabilities = torch.softmax(logits, dim=-1)
            if not bool(torch.isfinite(probabilities).all().item()):
                raise S1InstrumentationError("manual OFF probabilities are non-finite")
            record["pair_probabilities"] = _capture_pair_probabilities(probabilities, torch)

        output = full_heads.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        self.records.append(record)
        self.context = None
        return output


def install_pair_dictionary_processor(transformer: Any, torch_module: Any) -> tuple[PairDictionaryWanProcessor, Any]:
    if not hasattr(transformer, "blocks") or len(transformer.blocks) != 30:
        raise S1InstrumentationError("frozen Wan block topology unavailable")
    attention = transformer.blocks[14].attn1
    original = attention.processor
    processor = PairDictionaryWanProcessor(torch_module)
    attention.set_processor(processor)
    return processor, original


def _validate_capture(captures: Mapping[str, Any]) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    if set(captures) != set(BRANCH_ORDER):
        raise S1InstrumentationError("capture branch set changed")
    normalized: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for branch in BRANCH_ORDER:
        if set(captures[branch]) != set(AXIS_ORDER):
            raise S1InstrumentationError("capture axis set changed")
        normalized[branch] = {}
        for axis in AXIS_ORDER:
            if {int(value) for value in captures[branch][axis]} != set(RADII):
                raise S1InstrumentationError("capture radius set changed")
            normalized[branch][axis] = {}
            for radius in RADII:
                value = np.asarray(captures[branch][axis][str(radius)], dtype=np.float64)
                if value.shape != (13, 12, 2) or not np.isfinite(value).all():
                    raise S1InstrumentationError("pair capture must be finite [13,12,2]")
                if np.any(value < 0.0) or np.any(value > 1.0) or np.any(value.sum(axis=-1) > 1.0 + 1e-6):
                    raise S1InstrumentationError("pair capture is not a probability pair")
                normalized[branch][axis][radius] = value
    return normalized


def contrast_under_pair_bias(
    captures: Mapping[str, Mapping[int, np.ndarray]],
    *,
    inject_axis: str,
    inject_radius: int,
    observe_axis: str,
    observe_radius: int,
    signed_lambda: float,
) -> np.ndarray:
    inject = np.asarray(captures[inject_axis][inject_radius], dtype=np.float64)
    observe = np.asarray(captures[observe_axis][observe_radius], dtype=np.float64)
    if inject.shape != (13, 12, 2) or observe.shape != (13, 12, 2) or not math.isfinite(signed_lambda):
        raise S1InstrumentationError("analytic pair-bias inputs are invalid")
    delta = float(signed_lambda) / math.sqrt(2.0)
    positive = math.exp(delta)
    negative = math.exp(-delta)
    denominator = 1.0 + inject[..., 0] * (positive - 1.0) + inject[..., 1] * (negative - 1.0)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise S1InstrumentationError("analytic pair-bias normalization is invalid")
    adjusted = observe.copy()
    inject_pairs = pair_indices(inject_axis, inject_radius)
    observe_pairs = pair_indices(observe_axis, observe_radius)
    for row, (inject_pair, observe_pair) in enumerate(zip(inject_pairs, observe_pairs, strict=True)):
        for side, token in enumerate(observe_pair):
            if token == inject_pair[0]:
                adjusted[row, :, side] *= positive
            elif token == inject_pair[1]:
                adjusted[row, :, side] *= negative
    adjusted /= denominator[..., None]
    contrast = adjusted[..., 0] - adjusted[..., 1]
    if contrast.shape != (13, 12) or not np.isfinite(contrast).all():
        raise S1InstrumentationError("analytic pair contrast is invalid")
    return contrast


def _axis_branch_metrics(capture: Mapping[str, Mapping[int, np.ndarray]], axis: str, radius: int) -> dict[str, Any]:
    baseline = capture[axis][radius][..., 0] - capture[axis][radius][..., 1]
    plus = contrast_under_pair_bias(
        capture, inject_axis=axis, inject_radius=radius, observe_axis=axis, observe_radius=radius, signed_lambda=1.0
    )
    minus = contrast_under_pair_bias(
        capture, inject_axis=axis, inject_radius=radius, observe_axis=axis, observe_radius=radius, signed_lambda=-1.0
    )
    odd = (plus - minus) / 2.0
    even = (plus + minus) / 2.0 - baseline
    odd_rms = _rms(odd)
    even_rms = _rms(even)
    ulp_floor = max(float(np.max(np.abs(baseline))) * (2.0**-7), 2.0**-133)
    separation, separation_status = _json_safe_ratio(odd_rms, ulp_floor)
    even_ratio, even_status = _json_safe_ratio(even_rms, odd_rms)
    temporal_numerator = _rms(np.diff(odd, axis=0))
    temporal_tv, temporal_status = _json_safe_ratio(temporal_numerator, odd_rms)
    return {
        "odd": odd,
        "odd_rms": odd_rms,
        "even_rms": even_rms,
        "ulp_floor": ulp_floor,
        "odd_to_noise_or_ulp": separation,
        "odd_to_noise_or_ulp_status": separation_status,
        "even_to_odd": even_ratio,
        "even_to_odd_status": even_status,
        "temporal_total_variation": temporal_tv,
        "temporal_total_variation_numerator": temporal_numerator,
        "temporal_total_variation_status": temporal_status,
    }


def screen_pair_dictionary(captures: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _validate_capture(captures)
    candidates: dict[str, dict[str, Any]] = {axis: {} for axis in AXIS_ORDER}
    eligible: dict[str, list[int]] = {axis: [] for axis in AXIS_ORDER}
    internal_metrics: dict[tuple[str, int, str], dict[str, Any]] = {}
    for axis in AXIS_ORDER:
        for radius in RADII:
            branches: dict[str, Any] = {}
            for branch in BRANCH_ORDER:
                metrics = _axis_branch_metrics(normalized[branch], axis, radius)
                internal_metrics[(axis, radius, branch)] = metrics
                passed = (
                    metrics["odd_to_noise_or_ulp"] is not None
                    and metrics["even_to_odd"] is not None
                    and metrics["temporal_total_variation"] is not None
                    and metrics["odd_to_noise_or_ulp"] >= float(thresholds["odd_to_noise_or_ulp_minimum"])
                    and metrics["even_to_odd"] < float(thresholds["even_to_odd_strict_maximum"])
                    and metrics["temporal_total_variation"] < float(thresholds["temporal_total_variation_strict_maximum"])
                )
                branches[branch] = {key: value for key, value in metrics.items() if key != "odd"}
                branches[branch]["passed"] = bool(passed)
            defined = all(branches[branch]["passed"] for branch in BRANCH_ORDER)
            worst_separation = min(float(branches[branch]["odd_to_noise_or_ulp"]) for branch in BRANCH_ORDER) if defined else None
            worst_even = max(float(branches[branch]["even_to_odd"]) for branch in BRANCH_ORDER) if defined else None
            worst_tv = max(float(branches[branch]["temporal_total_variation"]) for branch in BRANCH_ORDER) if defined else None
            record = {
                "axis": axis,
                "radius": radius,
                "branches": branches,
                "eligible": bool(defined),
                "ranking": {
                    "worst_branch_odd_to_noise_or_ulp": worst_separation,
                    "worst_branch_even_to_odd": worst_even,
                    "worst_branch_temporal_total_variation": worst_tv,
                },
            }
            candidates[axis][str(radius)] = record
            if defined:
                eligible[axis].append(radius)

    for axis in AXIS_ORDER:
        eligible[axis].sort(
            key=lambda radius: (
                -float(candidates[axis][str(radius)]["ranking"]["worst_branch_odd_to_noise_or_ulp"]),
                float(candidates[axis][str(radius)]["ranking"]["worst_branch_even_to_odd"]),
                float(candidates[axis][str(radius)]["ranking"]["worst_branch_temporal_total_variation"]),
                radius,
            )
        )

    combinations: list[dict[str, Any]] = []
    for horizontal_radius in eligible["B1_horizontal"]:
        for vertical_radius in eligible["B2_vertical"]:
            branch_records: dict[str, Any] = {}
            passed = True
            for branch in BRANCH_ORDER:
                horizontal_odd = internal_metrics[("B1_horizontal", horizontal_radius, branch)]["odd"]
                vertical_odd = internal_metrics[("B2_vertical", vertical_radius, branch)]["odd"]
                h_into_v = (
                    contrast_under_pair_bias(
                        normalized[branch], inject_axis="B1_horizontal", inject_radius=horizontal_radius,
                        observe_axis="B2_vertical", observe_radius=vertical_radius, signed_lambda=1.0,
                    )
                    - contrast_under_pair_bias(
                        normalized[branch], inject_axis="B1_horizontal", inject_radius=horizontal_radius,
                        observe_axis="B2_vertical", observe_radius=vertical_radius, signed_lambda=-1.0,
                    )
                ) / 2.0
                v_into_h = (
                    contrast_under_pair_bias(
                        normalized[branch], inject_axis="B2_vertical", inject_radius=vertical_radius,
                        observe_axis="B1_horizontal", observe_radius=horizontal_radius, signed_lambda=1.0,
                    )
                    - contrast_under_pair_bias(
                        normalized[branch], inject_axis="B2_vertical", inject_radius=vertical_radius,
                        observe_axis="B1_horizontal", observe_radius=horizontal_radius, signed_lambda=-1.0,
                    )
                ) / 2.0
                horizontal_response = np.stack((horizontal_odd, h_into_v), axis=-1)
                vertical_response = np.stack((v_into_h, vertical_odd), axis=-1)
                axis_cosine = abs(_cosine(horizontal_response, vertical_response))
                gain_ratio, gain_status = _json_safe_ratio(_rms(horizontal_response), _rms(vertical_response))
                j11, j21 = float(np.mean(horizontal_odd)), float(np.mean(h_into_v))
                j12, j22 = float(np.mean(v_into_h)), float(np.mean(vertical_odd))
                cross_1, _ = _json_safe_ratio(abs(j21), max(abs(j11), 1e-30))
                cross_2, _ = _json_safe_ratio(abs(j12), max(abs(j22), 1e-30))
                cross = None if cross_1 is None or cross_2 is None else max(cross_1, cross_2)
                branch_pass = (
                    gain_ratio is not None
                    and cross is not None
                    and axis_cosine < float(thresholds["axis_absolute_cosine_strict_maximum"])
                    and float(thresholds["axis_gain_ratio_inclusive"][0]) <= gain_ratio <= float(thresholds["axis_gain_ratio_inclusive"][1])
                    and j11 > 0.0
                    and j22 > 0.0
                    and cross <= float(thresholds["relation_jacobian_cross_ratio_inclusive_maximum"])
                )
                branch_records[branch] = {
                    "axis_absolute_cosine": axis_cosine,
                    "axis_gain_ratio": gain_ratio,
                    "axis_gain_ratio_status": gain_status,
                    "jacobian_columns_B1_B2": [[j11, j12], [j21, j22]],
                    "maximum_cross_ratio": cross,
                    "passed": bool(branch_pass),
                }
                passed = passed and bool(branch_pass)
            axis_records = [
                candidates["B1_horizontal"][str(horizontal_radius)]["ranking"],
                candidates["B2_vertical"][str(vertical_radius)]["ranking"],
            ]
            combinations.append(
                {
                    "horizontal_radius": horizontal_radius,
                    "vertical_radius": vertical_radius,
                    "branches": branch_records,
                    "passed": bool(passed),
                    "ranking": {
                        "worst_axis_and_branch_odd_to_noise_or_ulp": min(float(item["worst_branch_odd_to_noise_or_ulp"]) for item in axis_records),
                        "worst_axis_and_branch_even_to_odd": max(float(item["worst_branch_even_to_odd"]) for item in axis_records),
                        "worst_axis_and_branch_temporal_total_variation": max(float(item["worst_branch_temporal_total_variation"]) for item in axis_records),
                    },
                }
            )
    passing = [record for record in combinations if record["passed"]]
    passing.sort(
        key=lambda record: (
            -record["ranking"]["worst_axis_and_branch_odd_to_noise_or_ulp"],
            record["ranking"]["worst_axis_and_branch_even_to_odd"],
            record["ranking"]["worst_axis_and_branch_temporal_total_variation"],
            record["horizontal_radius"],
            record["vertical_radius"],
        )
    )
    selected = None if not passing else {
        "horizontal_radius": passing[0]["horizontal_radius"],
        "vertical_radius": passing[0]["vertical_radius"],
    }
    return {
        "status": STATUS_READY if selected is not None else STATUS_NO_GO,
        "candidates": candidates,
        "eligible_order": {axis: eligible[axis] for axis in AXIS_ORDER},
        "combinations": combinations,
        "selected_pair": selected,
    }


def load_dictionary_inputs(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = repo_root / "configs/s1_local_pair_dictionary.json"
    plan_path = repo_root / "plans/s1_local_pair_dictionary.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    base_config, base_plan = load_frozen_inputs(repo_root)
    if config.get("schema") != SCHEMA or plan.get("schema") != PLAN_SCHEMA:
        raise S1InstrumentationError("dictionary config or plan identity changed")
    base = config.get("base_s1", {})
    if (
        sha256_file(repo_root / base["config_path"]) != base["config_raw_sha256"]
        or sha256_file(repo_root / base["plan_path"]) != base["plan_raw_sha256"]
        or sha256_file(repo_root / "SSTW_METHOD_AUTHORITY.md") != base["authority_raw_sha256"]
    ):
        raise S1InstrumentationError("base S1 identity changed")
    construction = config.get("frozen_construction", {})
    if (
        construction.get("block_index"), construction.get("scheduler_index"), construction.get("expected_timestep"),
        construction.get("lambda"), tuple(construction.get("query_indices", [])), tuple(construction.get("branches", [])),
    ) != (14, 4, 749, 1.0, TARGET_QUERY_INDICES, BRANCH_ORDER):
        raise S1InstrumentationError("dictionary construction changed")
    if plan.get("exact_transformer_calls") != 10 or plan.get("radius_order") != list(RADII):
        raise S1InstrumentationError("dictionary execution plan changed")
    if config.get("formal_result") is not False or config.get("stage_progression_allowed") is not False:
        raise S1InstrumentationError("dictionary diagnostic boundary changed")
    return config, plan, base_config, base_plan


def dictionary_exit_code(status: str) -> int:
    if status == STATUS_READY:
        return 0
    if status == STATUS_NO_GO:
        return 3
    raise S1InstrumentationError("unknown dictionary status")


def run_dictionary_once(*, repo_root: Path, output: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink() or not output.parent.is_dir() or output.parent.is_symlink():
        raise S1InstrumentationError("output must be absent under an existing regular parent")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise S1InstrumentationError("dictionary screen requires a clean exact checkout")
    config, plan, base_config, _base_plan = load_dictionary_inputs(repo_root)
    output.mkdir(mode=0o755)
    started_at = utc_now()
    pipe, torch, runtime = _load_runtime(base_config)
    transformer = pipe.transformer
    target_attention = transformer.blocks[14].attn1
    parameter_sha_before = sha256_parameters(target_attention)
    processor, original = install_pair_dictionary_processor(transformer, torch)
    call_records: list[dict[str, Any]] = []
    device = pipe._execution_device

    with torch.inference_mode():
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=base_config["generation"]["prompt"],
            negative_prompt=base_config["generation"]["negative_prompt"],
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(transformer.dtype).detach()
        negative_embeds = negative_embeds.to(transformer.dtype).detach()
        pipe.scheduler.set_timesteps(base_config["generation"]["inference_steps"], device=device)
        timesteps = pipe.scheduler.timesteps
        actual_timesteps = [int(value.item()) for value in timesteps]
        if len(actual_timesteps) != 8 or actual_timesteps[4] != 749:
            raise S1InstrumentationError(f"scheduler identity mismatch: {actual_timesteps}")
        generator = torch.Generator(device="cuda").manual_seed(base_config["generation"]["seed"])
        latents = pipe.prepare_latents(
            1, transformer.config.in_channels, 320, 512, 49, torch.float32, device, generator, None
        ).detach()

    def one_call(latent_input: Any, timestep: Any, scheduler_index: int, branch: str, capture: bool) -> Any:
        call_index = len(call_records) + 1
        processor.set_context(
            call_index=call_index,
            scheduler_index=scheduler_index,
            timestep=int(timestep.item()),
            branch=branch,
            capture=capture,
        )
        before = len(processor.records)
        embeddings = prompt_embeds if branch == "cond" else negative_embeds
        velocity = transformer_forward_inference(
            torch,
            transformer,
            branch=branch,
            hidden_states=latent_input.clone().to(transformer.dtype),
            timestep=timestep.expand(1),
            encoder_hidden_states=embeddings,
        )
        if len(processor.records) != before + 1:
            raise S1InstrumentationError("dictionary processor call count mismatch")
        record = processor.records[-1]
        if record["grad_enabled"] is not False or record["inference_mode_enabled"] is not True or record["output_replaced"] is not False:
            raise S1InstrumentationError("dictionary processor escaped inference-only OFF semantics")
        call_records.append(
            {
                "call_index": call_index,
                "scheduler_index": scheduler_index,
                "timestep": int(timestep.item()),
                "branch": branch,
                "capture": capture,
            }
        )
        return velocity

    for scheduler_index in range(4):
        timestep = timesteps[scheduler_index]
        cond_velocity = one_call(latents, timestep, scheduler_index, "cond", False)
        uncond_velocity = one_call(latents, timestep, scheduler_index, "uncond", False)
        with torch.inference_mode():
            guided = uncond_velocity + base_config["generation"]["guidance_scale"] * (cond_velocity - uncond_velocity)
            latents = pipe.scheduler.step(guided, timestep, latents, return_dict=False)[0].detach()
        del cond_velocity, uncond_velocity, guided

    with torch.inference_mode():
        capture_latent = latents.detach().clone()
    capture_latent_sha256 = sha256_tensor(capture_latent)
    timestep = timesteps[4]
    for branch in BRANCH_ORDER:
        velocity = one_call(capture_latent, timestep, 4, branch, True)
        del velocity

    target_attention.set_processor(original)
    if sha256_parameters(target_attention) != parameter_sha_before:
        raise S1InstrumentationError("target attention parameters changed")
    if len(call_records) != 10 or len(processor.records) != 10 or sum(item["capture"] for item in call_records) != 2:
        raise S1InstrumentationError("dictionary exact10 call budget mismatch")
    capture_records = [record for record in processor.records if record["capture"]]
    captures = {record["branch"]: record["pair_probabilities"] for record in capture_records}
    screen = screen_pair_dictionary(captures, config["candidate_thresholds"])
    status = screen["status"]
    stats = {
        "schema": SCHEMA,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "capture_latent_sha256": capture_latent_sha256,
        "scheduler_timesteps": actual_timesteps,
        "call_records": call_records,
        "screen": screen,
    }
    audit = {
        "schema": SCHEMA,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "status": status,
        "source": {"head": _git(repo_root, "rev-parse", "HEAD"), "tree": _git(repo_root, "rev-parse", "HEAD^{tree}"), "dirty": False},
        "runtime": runtime,
        "transformer_calls": 10,
        "scheduler_steps": 4,
        "block_or_velocity_injection": False,
        "VAE_was_run": False,
        "MP4_was_written": False,
        "S2_was_run": False,
        "formal_result": False,
        "stage_progression_allowed": False,
        "started_at": started_at,
        "ended_at": utc_now(),
    }
    payloads = {
        "stats.json": stats,
        "audit.json": audit,
        "config.json": config,
        "plan.json": plan,
        "command.json": {"argv": list(argv), "cwd": str(cwd.resolve()), "exit_code": dictionary_exit_code(status)},
    }
    for name, payload in payloads.items():
        (output / name).write_bytes(canonical_json_bytes(payload) + b"\n")
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    return {
        "status": status,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "output": str(output),
        "transformer_calls": 10,
        "selected_pair": screen["selected_pair"],
        "formal_result": False,
        "stage_progression_allowed": False,
    }
