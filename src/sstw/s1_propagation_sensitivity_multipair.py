"""Finite real-Wan propagation screen for sparse two-pair relation bases."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .relation_injector import TARGET_QUERY_INDICES, _apply_rotary, dispatch_native_selected_attention, replace_selected_rows
from .s1_real_dit_relation_primitive import (
    BRANCH_ORDER,
    CONDITION_ORDER,
    DIAGNOSTIC_CLASS,
    S1InstrumentationError,
    _extract_velocity_slice,
    _git,
    _load_runtime,
    capture_block_output,
    canonical_json_bytes,
    evaluate_preregistered_statistics,
    odd_even_noise,
    sha256_file,
    sha256_parameters,
    sha256_tensor,
    transformer_forward_inference,
)


SCHEMA = "sstw.s1.propagation_sensitivity_multipair.v1"
PLAN_SCHEMA = "sstw.s1.propagation_sensitivity_multipair.plan.v1"
STATUS_READY = "PROPAGATION_CONSTRUCTION_READY"
STATUS_NO_GO = "PROPAGATION_CONSTRUCTION_NO_GO"
BLOCKS = (7, 14, 22)
SIGN_ORDER = ("coherent_sum", "contrast_difference")
AXES = ("B1", "B2")
GUIDANCE = 5.0
FLOW_SUPPORTS = {
    "single_step4": ((4,), (1.0,)),
    "three_steps345": ((3, 4, 5), (1.0 / math.sqrt(3.0),) * 3),
}
RELATION_BANK_INDEX = {
    ("B1", "coherent_sum"): 0,
    ("B1", "contrast_difference"): 1,
    ("B2", "coherent_sum"): 2,
    ("B2", "contrast_difference"): 3,
}
SCHEDULER_HISTORY_FIELDS = (
    "_step_index",
    "order",
    "lower_order_nums",
    "model_outputs",
    "timestep_list",
    "last_sample",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _scheduler_state_value(value: Any, torch: Any) -> Any:
    """Return a stable, JSON-safe representation of scheduler state."""
    if torch.is_tensor(value):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": sha256_tensor(value),
        }
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _scheduler_state_value(item, torch) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_scheduler_state_value(item, torch) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def scheduler_state_summary(scheduler: Any, torch: Any) -> dict[str, Any]:
    """Fingerprint the complete instance state and the UniPC history fields."""
    missing = [name for name in ("_step_index", "lower_order_nums", "model_outputs") if not hasattr(scheduler, name)]
    if missing:
        raise S1InstrumentationError(f"scheduler history fields unavailable: {missing}")
    instance_state = {
        str(name): _scheduler_state_value(value, torch)
        for name, value in sorted(vars(scheduler).items())
    }
    digest = hashlib.sha256(canonical_json_bytes(instance_state)).hexdigest()
    history = {
        name: _scheduler_state_value(getattr(scheduler, name), torch)
        for name in SCHEDULER_HISTORY_FIELDS
        if hasattr(scheduler, name)
    }
    return {
        "state_sha256": digest,
        "state_keys": sorted(instance_state),
        "history": history,
    }


def capture_scheduler_snapshot(
    scheduler: Any,
    torch: Any,
    *,
    expected_step_index: int,
    label: str,
) -> tuple[Any, dict[str, Any]]:
    """Deep-copy a stateful scheduler without reconstructing it from config."""
    actual_step_index = getattr(scheduler, "_step_index", None)
    if actual_step_index != expected_step_index:
        raise S1InstrumentationError(
            f"{label} scheduler step index mismatch: expected {expected_step_index}, got {actual_step_index}"
        )
    source_summary = scheduler_state_summary(scheduler, torch)
    try:
        snapshot = copy.deepcopy(scheduler)
    except Exception as exc:
        raise S1InstrumentationError(f"{label} scheduler deep snapshot failed") from exc
    snapshot_summary = scheduler_state_summary(snapshot, torch)
    if snapshot_summary != source_summary:
        raise S1InstrumentationError(f"{label} scheduler deep snapshot lost history")
    return snapshot, snapshot_summary


def fork_scheduler_snapshot(snapshot: Any, expected: Mapping[str, Any], torch: Any, *, label: str) -> Any:
    """Create one condition-local scheduler and leave the retained snapshot untouched."""
    if scheduler_state_summary(snapshot, torch) != expected:
        raise S1InstrumentationError(f"{label} retained scheduler snapshot was mutated")
    try:
        fork = copy.deepcopy(snapshot)
    except Exception as exc:
        raise S1InstrumentationError(f"{label} scheduler fork failed") from exc
    if scheduler_state_summary(fork, torch) != expected:
        raise S1InstrumentationError(f"{label} scheduler fork lost history")
    return fork


def assert_scheduler_snapshot_unchanged(snapshot: Any, expected: Mapping[str, Any], torch: Any, *, label: str) -> None:
    if scheduler_state_summary(snapshot, torch) != expected:
        raise S1InstrumentationError(f"{label} retained scheduler snapshot was mutated")


def two_pair_basis(axis: str, sign: str) -> tuple[tuple[tuple[int, float], ...], ...]:
    if axis not in AXES or sign not in SIGN_ORDER:
        raise S1InstrumentationError("two-pair basis identity changed")
    radii = (2, 5) if axis == "B1" else (2, 8)
    stride = 1 if axis == "B1" else 32
    atom_sign = 1.0 if sign == "coherent_sum" else -1.0
    rows = []
    for query in TARGET_QUERY_INDICES:
        keys = (
            (query - stride * radii[0], 0.5),
            (query + stride * radii[0], -0.5),
            (query - stride * radii[1], 0.5 * atom_sign),
            (query + stride * radii[1], -0.5 * atom_sign),
        )
        if len({key for key, _ in keys}) != 4 or not all(0 <= key < 8320 for key, _ in keys):
            raise S1InstrumentationError("two-pair support is duplicate or out of bounds")
        coefficients = np.asarray([coefficient for _, coefficient in keys], dtype=np.float64)
        if abs(float(coefficients.sum())) > 1e-12 or abs(float(np.linalg.norm(coefficients)) - 1.0) > 1e-12:
            raise S1InstrumentationError("two-pair basis lost zero-sum unit-L2 identity")
        rows.append(keys)
    return tuple(rows)


def all_basis_updates(axis: str, sign: str, amplitude: float) -> tuple[tuple[int, int, float], ...]:
    if not math.isfinite(amplitude):
        raise S1InstrumentationError("basis amplitude is non-finite")
    updates = []
    for row, keys in enumerate(two_pair_basis(axis, sign)):
        for key, coefficient in keys:
            delta = float(amplitude) * coefficient
            if delta != 0.0:
                updates.append((row, key, delta))
    return tuple(updates)


def relation_bank(probabilities: Any, torch: Any) -> Any:
    if tuple(probabilities.shape) != (1, 12, 13, 8320):
        raise S1InstrumentationError("relation probabilities must be [1,12,13,8320]")
    channels = []
    for axis, sign in RELATION_BANK_INDEX:
        rows = []
        for row, keys in enumerate(two_pair_basis(axis, sign)):
            value = sum(probabilities[0, :, row, key] * coefficient for key, coefficient in keys)
            rows.append(value)
        channels.append(torch.stack(rows, dim=0))
    bank = torch.stack(channels, dim=-1)
    if tuple(bank.shape) != (13, 12, 4) or not bool(torch.isfinite(bank).all().item()):
        raise S1InstrumentationError("relation bank must be finite [13,12,4]")
    return bank.detach().float().cpu()


class MultiPairWanProcessor:
    """Official Wan attention with a selected-row sparse two-pair bias."""

    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.context: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def set_context(
        self,
        *,
        call_index: int,
        scheduler_index: int,
        branch: str,
        active_axis: str | None,
        active_sign: str | None,
        amplitude: float,
        capture: bool,
    ) -> None:
        if branch not in BRANCH_ORDER or (active_axis is None) != (active_sign is None):
            raise S1InstrumentationError("processor context is invalid")
        self.context = {
            "call_index": int(call_index),
            "scheduler_index": int(scheduler_index),
            "branch": branch,
            "active_axis": active_axis,
            "active_sign": active_sign,
            "amplitude": float(amplitude),
            "capture": bool(capture),
        }

    def __call__(self, attn: Any, hidden_states: Any, encoder_hidden_states: Any = None, attention_mask: Any = None, rotary_emb: Any = None) -> Any:
        if self.context is None:
            raise S1InstrumentationError("processor context was not set")
        if encoder_hidden_states is not None or attention_mask is not None or rotary_emb is None:
            raise S1InstrumentationError("only frozen Wan self-attention is supported")
        try:
            from diffusers.models.attention_dispatch import dispatch_attention_fn
        except Exception as exc:  # pragma: no cover
            raise S1InstrumentationError("diffusers attention dispatcher unavailable") from exc
        torch = self.torch
        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        query = _apply_rotary(torch, query, rotary_emb)
        key = _apply_rotary(torch, key, rotary_emb)
        if tuple(query.shape) != (1, 8320, 12, 128) or tuple(key.shape) != tuple(query.shape) or tuple(value.shape) != tuple(query.shape):
            raise S1InstrumentationError("Wan QKV identity changed")
        full_heads = dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None)
        query_index = torch.tensor(TARGET_QUERY_INDICES, device=query.device, dtype=torch.long)
        target_query = query.index_select(1, query_index)
        bias = torch.zeros((1, 1, 13, 8320), device=query.device, dtype=query.dtype)
        updates: tuple[tuple[int, int, float], ...] = ()
        active = self.context["active_axis"] is not None and self.context["amplitude"] != 0.0
        if active:
            updates = all_basis_updates(self.context["active_axis"], self.context["active_sign"], self.context["amplitude"])
            for row, key_index, delta in updates:
                bias[:, :, row, key_index] += delta
        bank = None
        if self.context["capture"]:
            logits = torch.einsum("bqhd,bkhd->bhqk", target_query.float(), key.float()) / math.sqrt(128.0)
            for row, key_index, delta in updates:
                logits[:, :, row, key_index] += delta
            probabilities = torch.softmax(logits, dim=-1)
            bank = relation_bank(probabilities, torch).tolist()
        if active:
            selected = dispatch_native_selected_attention(dispatch_attention_fn, target_query, key, value, bias)
            full_heads = replace_selected_rows(full_heads, selected, query_index, active=True)
        output = full_heads.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        self.records.append(
            {
                **self.context,
                "relation_bank": bank,
                "changed_logit_count": len(updates),
                "pair_sum_zero": all(abs(sum(delta for update_row, _key, delta in updates if update_row == row)) <= 1e-12 for row in range(13)),
                "selected_bias_shape": [1, 1, 13, 8320],
                "dense_full_bias": False,
                "grad_enabled": bool(torch.is_grad_enabled()),
                "inference_mode_enabled": bool(torch.is_inference_mode_enabled()),
            }
        )
        self.context = None
        return output


def install_processor(transformer: Any, torch: Any, block: int) -> tuple[MultiPairWanProcessor, Any]:
    if block not in BLOCKS or len(transformer.blocks) != 30:
        raise S1InstrumentationError("frozen block topology unavailable")
    attention = transformer.blocks[block].attn1
    original = attention.processor
    processor = MultiPairWanProcessor(torch)
    attention.set_processor(processor)
    return processor, original


def validate_processor_record(record: Mapping[str, Any], *, active: bool, captured: bool) -> None:
    expected_updates = 52 if active else 0
    if (
        record.get("changed_logit_count") != expected_updates
        or record.get("pair_sum_zero") is not True
        or record.get("selected_bias_shape") != [1, 1, 13, 8320]
        or record.get("dense_full_bias") is not False
        or record.get("grad_enabled") is not False
        or record.get("inference_mode_enabled") is not True
        or (record.get("relation_bank") is not None) is not captured
    ):
        raise S1InstrumentationError("multi-pair processor structural contract failed")


def solve_cfg_weights(probe: Mapping[str, Any]) -> dict[str, Any]:
    odd: dict[str, dict[str, np.ndarray]] = {branch: {} for branch in BRANCH_ORDER}
    even: dict[str, dict[str, np.ndarray]] = {branch: {} for branch in BRANCH_ORDER}
    for branch in BRANCH_ORDER:
        records = {condition: np.asarray(probe[condition][branch]["relation"], dtype=np.float64) for condition in CONDITION_ORDER}
        for axis in AXES:
            odd[branch][axis], even[branch][axis], _noise, _baseline = odd_even_noise(records, axis)
    odd_columns = [np.concatenate([odd[branch][axis].ravel() for axis in AXES]) for branch in BRANCH_ORDER]
    even_columns = [np.concatenate([even[branch][axis].ravel() for axis in AXES]) for branch in BRANCH_ORDER]
    coefficients = (GUIDANCE, 1.0 - GUIDANCE)
    x = np.column_stack([coefficient * column for coefficient, column in zip(coefficients, odd_columns, strict=True)])
    y = np.column_stack([coefficient * column for coefficient, column in zip(coefficients, even_columns, strict=True)])
    signal = x.T @ x
    nuisance = y.T @ y
    if not np.isfinite(signal).all() or not np.isfinite(nuisance).all() or np.linalg.matrix_rank(signal) < 2 or np.linalg.matrix_rank(nuisance) < 2:
        raise S1InstrumentationError("CFG covariance is non-finite or rank deficient")
    nuisance_values, nuisance_vectors = np.linalg.eigh(nuisance)
    inverse_sqrt = nuisance_vectors @ np.diag(1.0 / np.sqrt(nuisance_values)) @ nuisance_vectors.T
    whitened = inverse_sqrt @ signal @ inverse_sqrt
    values, vectors = np.linalg.eigh(whitened)
    if not values[-1] > values[-2]:
        raise S1InstrumentationError("CFG maximum generalized eigenvalue is not unique")
    weight = inverse_sqrt @ vectors[:, -1]
    norm = float(np.linalg.norm(weight))
    if not math.isfinite(norm) or norm == 0.0:
        raise S1InstrumentationError("CFG weight is degenerate")
    weight /= norm
    orientations = []
    for axis_index, axis in enumerate(AXES):
        guided = coefficients[0] * weight[0] * odd["cond"][axis] + coefficients[1] * weight[1] * odd["uncond"][axis]
        orientations.append(float(np.mean(guided[..., axis_index])))
    if all(value < 0.0 for value in orientations):
        weight *= -1.0
        orientations = [-value for value in orientations]
    if not all(value > 0.0 for value in orientations) or np.any(weight == 0.0):
        raise S1InstrumentationError("CFG solution flips or erases an axis")
    return {
        "weights": {"cond": float(weight[0]), "uncond": float(weight[1])},
        "signal_covariance": signal.tolist(),
        "nuisance_covariance": nuisance.tolist(),
        "generalized_eigenvalues": values.tolist(),
        "guided_axis_orientations": orientations,
    }


def _select_relation(bank: Any, horizontal_sign: str, vertical_sign: str) -> list[Any]:
    values = np.asarray(bank, dtype=np.float64)
    if values.shape != (13, 12, 4) or not np.isfinite(values).all():
        raise S1InstrumentationError("relation bank is invalid")
    return np.stack((values[..., RELATION_BANK_INDEX[("B1", horizontal_sign)]], values[..., RELATION_BANK_INDEX[("B2", vertical_sign)]]), axis=-1).tolist()


def _numeric_probe(records: Mapping[str, Any], horizontal_sign: str, vertical_sign: str) -> dict[str, Any]:
    return {
        condition: {
            branch: {
                "relation": _select_relation(records[condition][branch]["relation_bank"], horizontal_sign, vertical_sign),
                "block": records[condition][branch]["block"],
                "velocity": records[condition][branch]["velocity"],
            }
            for branch in BRANCH_ORDER
        }
        for condition in CONDITION_ORDER
    }


def _global_overrides(records: Mapping[str, Any]) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    for branch in (*BRANCH_ORDER, "guidance"):
        for axis in AXES:
            plus_name, minus_name = (("PLUS_B1", "MINUS_B1") if axis == "B1" else ("PLUS_B2", "MINUS_B2"))
            for layer in ("block", "velocity"):
                key = "_block_full" if layer == "block" else "_velocity_full"
                values = {}
                for condition in CONDITION_ORDER:
                    if branch == "guidance":
                        uncond = records[condition]["uncond"][key]
                        cond = records[condition]["cond"][key]
                        values[condition] = uncond + GUIDANCE * (cond - uncond)
                    else:
                        values[condition] = records[condition][branch][key]
                odd = (values[plus_name] - values[minus_name]) / 2.0
                baseline = (values["OFF_R1"] + values["OFF_R2"]) / 2.0
                result[(branch, layer, axis)] = float(odd.square().mean().sqrt().item() / max(baseline.square().mean().sqrt().item(), 1e-30))
    return result


def evaluate_records(records: Mapping[str, Any], horizontal_sign: str, vertical_sign: str, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    numeric = _numeric_probe(records, horizontal_sign, vertical_sign)
    return evaluate_preregistered_statistics(numeric, thresholds, _global_overrides(records))


def stage1_ranking(evaluation: Mapping[str, Any], block: int, horizontal_sign: str, vertical_sign: str) -> tuple[Any, ...] | None:
    velocity_axes = [evaluation["cells"][f"velocity:{branch}"]["axes"][axis] for branch in (*BRANCH_ORDER, "guidance") for axis in AXES]
    relation_cells = [evaluation["cells"][f"relation:{branch}"] for branch in (*BRANCH_ORDER, "guidance")]
    if any(axis["odd_rms"] <= 0.0 or axis["odd_to_noise_or_ulp"] is None or axis["even_to_odd"] is None or axis["temporal_total_variation"] is None for axis in velocity_axes):
        return None
    if any(cell["relation_jacobian"] is None or cell["relation_jacobian"]["matrix_columns_B1_B2"][0][0] <= 0.0 or cell["relation_jacobian"]["matrix_columns_B1_B2"][1][1] <= 0.0 for cell in relation_cells):
        return None
    return (
        -min(float(axis["odd_to_noise_or_ulp"]) for axis in velocity_axes),
        max(float(axis["even_to_odd"]) for axis in velocity_axes),
        max(float(axis["temporal_total_variation"]) for axis in velocity_axes),
        max(float(evaluation["cells"]["velocity:guidance"]["axes"][axis]["global_relative_rms"]) for axis in AXES),
        block,
        SIGN_ORDER.index(horizontal_sign),
        SIGN_ORDER.index(vertical_sign),
    )


def propagation_exit_code(status: str) -> int:
    if status == STATUS_READY:
        return 0
    if status == STATUS_NO_GO:
        return 3
    raise S1InstrumentationError("unknown propagation status")


def load_inputs(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = repo_root / "configs/s1_propagation_sensitivity_multipair.json"
    plan_path = repo_root / "plans/s1_propagation_sensitivity_multipair.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    authority = {"commit": "0f85bd70de6f042c68560ed348722fc4fbd112e9", "tree": "d791cff7b5cbebb6ec690f16196bde9ee6d50bd1", "raw_sha256": "85c32f49897a6b7c6248d5fd6fe4c7a307b4b841dc7388551d095854b6189292"}
    if config.get("schema") != SCHEMA or plan.get("schema") != PLAN_SCHEMA or config.get("method_authority") != authority or plan.get("method_authority") != authority:
        raise S1InstrumentationError("propagation config/plan authority changed")
    if sha256_file(repo_root / "SSTW_METHOD_AUTHORITY.md") != authority["raw_sha256"]:
        raise S1InstrumentationError("method authority content changed")
    base = config["base_s1"]
    if sha256_file(repo_root / base["config_path"]) != base["config_raw_sha256"] or sha256_file(repo_root / base["plan_path"]) != base["plan_raw_sha256"]:
        raise S1InstrumentationError("base S1 science identity changed")
    base_config = json.loads((repo_root / base["config_path"]).read_text(encoding="utf-8"))
    if config["call_budget"]["full_path_transformer_calls"] != 170 or plan["full_path_transformer_calls"] != 170 or config["call_budget"]["possible_terminal_transformer_calls"] != [108, 138, 146, 162, 170]:
        raise S1InstrumentationError("finite transformer-call budget changed")
    return config, plan, base_config


def _condition_axis(condition: str) -> tuple[str | None, float]:
    if condition.startswith("PLUS_B"):
        return condition[-2:], 1.0
    if condition.startswith("MINUS_B"):
        return condition[-2:], -1.0
    return None, 0.0


def run_propagation_once(*, repo_root: Path, output: Path, argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise S1InstrumentationError("output must be absent under an existing parent")
    if _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise S1InstrumentationError("screen requires a clean checkout")
    config, plan, base_config = load_inputs(repo_root)
    output.mkdir()
    started = utc_now()
    pipe, torch, runtime = _load_runtime(base_config)
    transformer = pipe.transformer
    if len(transformer.blocks) != 30:
        raise S1InstrumentationError("Wan layer count is not 30")
    device = pipe._execution_device
    call_count = 0

    with torch.inference_mode():
        cond_embeddings, uncond_embeddings = pipe.encode_prompt(
            prompt=base_config["generation"]["prompt"], negative_prompt=base_config["generation"]["negative_prompt"],
            do_classifier_free_guidance=True, num_videos_per_prompt=1, device=device,
        )
        cond_embeddings = cond_embeddings.to(transformer.dtype).detach()
        uncond_embeddings = uncond_embeddings.to(transformer.dtype).detach()
        pipe.scheduler.set_timesteps(8, device=device)
        timesteps = pipe.scheduler.timesteps
        actual_timesteps = [int(value.item()) for value in timesteps]
        if len(actual_timesteps) != 8 or actual_timesteps[4] != 749:
            raise S1InstrumentationError("scheduler identity changed")
        generator = torch.Generator(device="cuda").manual_seed(base_config["generation"]["seed"])
        latents = pipe.prepare_latents(1, transformer.config.in_channels, 320, 512, 49, torch.float32, device, generator, None).detach()

    def plain_forward(latent: Any, step: int, branch: str) -> Any:
        nonlocal call_count
        call_count += 1
        embeddings = cond_embeddings if branch == "cond" else uncond_embeddings
        return transformer_forward_inference(torch, transformer, branch=branch, hidden_states=latent.clone().to(transformer.dtype), timestep=timesteps[step].expand(1), encoder_hidden_states=embeddings)

    latent_at_step3 = None
    scheduler_at_step3 = None
    scheduler_at_step3_summary = None
    for step in range(4):
        if step == 3:
            latent_at_step3 = latents.detach().clone()
            scheduler_at_step3, scheduler_at_step3_summary = capture_scheduler_snapshot(
                pipe.scheduler, torch, expected_step_index=3, label="step3"
            )
        cond_velocity = plain_forward(latents, step, "cond")
        uncond_velocity = plain_forward(latents, step, "uncond")
        with torch.inference_mode():
            guided = uncond_velocity + GUIDANCE * (cond_velocity - uncond_velocity)
            latents = pipe.scheduler.step(guided, timesteps[step], latents, return_dict=False)[0].detach()
        del cond_velocity, uncond_velocity, guided
    latent_at_step4 = latents.detach().clone()
    scheduler_at_step4, scheduler_at_step4_summary = capture_scheduler_snapshot(
        pipe.scheduler, torch, expected_step_index=4, label="step4"
    )
    if latent_at_step3 is None or scheduler_at_step3 is None or scheduler_at_step3_summary is None:
        raise S1InstrumentationError("step3 latent and scheduler were not retained")
    scheduler_forks: list[dict[str, Any]] = []

    def run_calls(
        block: int,
        horizontal_sign: str,
        vertical_sign: str,
        support: str,
        branch_weights: Mapping[str, float],
        off_records: Mapping[str, Any] | None = None,
        *,
        fork_purpose: str | None = None,
    ) -> dict[str, Any]:
        nonlocal call_count
        processor, original = install_processor(transformer, torch, block)
        target_block = transformer.blocks[block]
        parameter_before = sha256_parameters(target_block.attn1)
        captures: list[tuple[Any, float, Any]] = []
        keep_block = {"value": False}
        def block_hook(_module: Any, _inputs: Any, value: Any) -> None:
            if keep_block["value"]:
                captures.append(capture_block_output(value, torch))
        hook = target_block.register_forward_hook(block_hook)
        result: dict[str, Any] = {} if off_records is None else {name: off_records[name] for name in ("OFF_R1", "OFF_R2")}
        steps, flow_weights = FLOW_SUPPORTS[support]
        try:
            for condition in ("PLUS_B1", "MINUS_B1", "PLUS_B2", "MINUS_B2"):
                result[condition] = {}
                axis, condition_sign = _condition_axis(condition)
                basis_sign = horizontal_sign if axis == "B1" else vertical_sign
                condition_latent = (latent_at_step4 if support == "single_step4" else latent_at_step3).detach().clone()
                condition_scheduler = None
                initial_latent_sha256 = sha256_tensor(condition_latent)
                if support == "three_steps345":
                    if fork_purpose not in ("three_fit", "three_apply"):
                        raise S1InstrumentationError("three-step scheduler fork purpose is invalid")
                    condition_scheduler = fork_scheduler_snapshot(
                        scheduler_at_step3,
                        scheduler_at_step3_summary,
                        torch,
                        label=f"{fork_purpose}:{condition}",
                    )
                try:
                    for step, flow_weight in zip(steps, flow_weights, strict=True):
                        velocities = {}
                        kept = {}
                        for branch in BRANCH_ORDER:
                            call_count += 1
                            keep = step == steps[-1]
                            processor.set_context(call_index=call_count, scheduler_index=step, branch=branch, active_axis=axis, active_sign=basis_sign, amplitude=condition_sign * float(branch_weights[branch]) * flow_weight, capture=keep)
                            before = len(processor.records)
                            captures.clear()
                            keep_block["value"] = keep
                            embeddings = cond_embeddings if branch == "cond" else uncond_embeddings
                            velocity = transformer_forward_inference(torch, transformer, branch=branch, hidden_states=condition_latent.clone().to(transformer.dtype), timestep=timesteps[step].expand(1), encoder_hidden_states=embeddings)
                            if len(processor.records) != before + 1 or len(captures) != (1 if keep else 0):
                                raise S1InstrumentationError("processor or block capture count mismatch")
                            validate_processor_record(processor.records[-1], active=True, captured=keep)
                            velocities[branch] = velocity
                            if keep:
                                block_rows, block_rms, block_full = captures[0]
                                kept[branch] = {
                                    "relation_bank": processor.records[-1]["relation_bank"], "block": block_rows.tolist(),
                                    "velocity": _extract_velocity_slice(velocity).tolist(), "block_global_rms": block_rms,
                                    "_block_full": block_full, "_velocity_full": velocity.detach().float().cpu(),
                                }
                        if step != steps[-1]:
                            if condition_scheduler is None:
                                raise S1InstrumentationError("stateful scheduler fork is missing")
                            with torch.inference_mode():
                                guided = velocities["uncond"] + GUIDANCE * (velocities["cond"] - velocities["uncond"])
                                condition_latent = condition_scheduler.step(guided, timesteps[step], condition_latent, return_dict=False)[0].detach()
                        del velocities
                    result[condition] = kept
                finally:
                    if condition_scheduler is not None:
                        assert_scheduler_snapshot_unchanged(
                            scheduler_at_step3,
                            scheduler_at_step3_summary,
                            torch,
                            label=f"{fork_purpose}:{condition}",
                        )
                        scheduler_forks.append(
                            {
                                "purpose": fork_purpose,
                                "condition": condition,
                                "start_step_index": 3,
                                "initial_latent_sha256": initial_latent_sha256,
                                "snapshot_state_sha256": scheduler_at_step3_summary["state_sha256"],
                                "snapshot_unchanged": True,
                            }
                        )
                        condition_scheduler = None
        finally:
            hook.remove()
            target_block.attn1.set_processor(original)
        if sha256_parameters(target_block.attn1) != parameter_before:
            raise S1InstrumentationError("attention parameters changed")
        return result

    def shared_off_step4() -> dict[int, Any]:
        nonlocal call_count
        processors = {}
        originals = {}
        hooks = {}
        captures = {block: [] for block in BLOCKS}
        result = {block: {"OFF_R1": {}, "OFF_R2": {}} for block in BLOCKS}
        try:
            for block in BLOCKS:
                processors[block], originals[block] = install_processor(transformer, torch, block)
                hooks[block] = transformer.blocks[block].register_forward_hook(
                    lambda _module, _inputs, value, block_value=block: captures[block_value].append(capture_block_output(value, torch))
                )
            for repeat in ("OFF_R1", "OFF_R2"):
                for branch in BRANCH_ORDER:
                    call_count += 1
                    for block in BLOCKS:
                        processors[block].set_context(call_index=call_count, scheduler_index=4, branch=branch, active_axis=None, active_sign=None, amplitude=0.0, capture=True)
                        captures[block].clear()
                    embeddings = cond_embeddings if branch == "cond" else uncond_embeddings
                    velocity = transformer_forward_inference(torch, transformer, branch=branch, hidden_states=latent_at_step4.clone().to(transformer.dtype), timestep=timesteps[4].expand(1), encoder_hidden_states=embeddings)
                    for block in BLOCKS:
                        validate_processor_record(processors[block].records[-1], active=False, captured=True)
                        block_rows, block_rms, block_full = captures[block][0]
                        result[block][repeat][branch] = {
                            "relation_bank": processors[block].records[-1]["relation_bank"], "block": block_rows.tolist(),
                            "velocity": _extract_velocity_slice(velocity).tolist(), "block_global_rms": block_rms,
                            "_block_full": block_full, "_velocity_full": velocity.detach().float().cpu(),
                        }
                    del velocity
        finally:
            for block in BLOCKS:
                if block in hooks:
                    hooks[block].remove()
                    transformer.blocks[block].attn1.set_processor(originals[block])
        return result

    shared_off = shared_off_step4()
    stage1 = []
    stage1_selected_records = None
    for block in BLOCKS:
        for horizontal_sign in SIGN_ORDER:
            for vertical_sign in SIGN_ORDER:
                equal_fit_weights = {"cond": 1.0 / math.sqrt(2.0), "uncond": 1.0 / math.sqrt(2.0)}
                records = run_calls(block, horizontal_sign, vertical_sign, "single_step4", equal_fit_weights, shared_off[block])
                evaluation = evaluate_records(records, horizontal_sign, vertical_sign, config["thresholds"])
                ranking = stage1_ranking(evaluation, block, horizontal_sign, vertical_sign)
                entry = {"block": block, "horizontal_sign": horizontal_sign, "vertical_sign": vertical_sign, "usable": ranking is not None, "ranking": ranking, "evaluation": evaluation}
                stage1.append(entry)
                if ranking is not None and (stage1_selected_records is None or ranking < stage1_selected_records[0]):
                    stage1_selected_records = (ranking, entry, records)
    del records
    if stage1_selected_records is None:
        status = STATUS_NO_GO
        selected = None
        supports = []
    else:
        _ranking, selected_stage1, selected_records = stage1_selected_records
        stage1_selected_records = None
        block = selected_stage1["block"]
        horizontal_sign = selected_stage1["horizontal_sign"]
        vertical_sign = selected_stage1["vertical_sign"]
        selected_off = shared_off[block]
        shared_off = {block: selected_off}
        try:
            single_weight = solve_cfg_weights(_numeric_probe(selected_records, horizontal_sign, vertical_sign))
        except S1InstrumentationError as exc:
            single_weight = {"valid": False, "reason": str(exc)}
            single_evaluation = None
        else:
            single_weight["valid"] = True
            single_records = run_calls(block, horizontal_sign, vertical_sign, "single_step4", single_weight["weights"], selected_off)
            single_evaluation = evaluate_records(single_records, horizontal_sign, vertical_sign, config["thresholds"])
            del single_records
        del selected_records

        processor, original = install_processor(transformer, torch, block)
        target_block = transformer.blocks[block]
        captures: list[tuple[Any, float, Any]] = []
        hook = target_block.register_forward_hook(lambda _module, _inputs, value: captures.append(capture_block_output(value, torch)))
        try:
            off_step5_latent = latent_at_step4.detach().clone()
            off_scheduler = fork_scheduler_snapshot(
                scheduler_at_step4,
                scheduler_at_step4_summary,
                torch,
                label="three_off_continuation",
            )
            velocities = {}
            for branch in BRANCH_ORDER:
                call_count += 1
                processor.set_context(call_index=call_count, scheduler_index=4, branch=branch, active_axis=None, active_sign=None, amplitude=0.0, capture=False)
                captures.clear()
                embeddings = cond_embeddings if branch == "cond" else uncond_embeddings
                velocities[branch] = transformer_forward_inference(torch, transformer, branch=branch, hidden_states=off_step5_latent.clone().to(transformer.dtype), timestep=timesteps[4].expand(1), encoder_hidden_states=embeddings)
                validate_processor_record(processor.records[-1], active=False, captured=False)
            with torch.inference_mode():
                guided = velocities["uncond"] + GUIDANCE * (velocities["cond"] - velocities["uncond"])
                off_step5_latent = off_scheduler.step(guided, timesteps[4], off_step5_latent, return_dict=False)[0].detach()
            assert_scheduler_snapshot_unchanged(
                scheduler_at_step4,
                scheduler_at_step4_summary,
                torch,
                label="three_off_continuation",
            )
            scheduler_forks.append(
                {
                    "purpose": "three_off_continuation",
                    "condition": "OFF",
                    "start_step_index": 4,
                    "initial_latent_sha256": sha256_tensor(latent_at_step4),
                    "snapshot_state_sha256": scheduler_at_step4_summary["state_sha256"],
                    "snapshot_unchanged": True,
                }
            )
            del off_scheduler
            del velocities, guided
            three_off = {"OFF_R1": {}, "OFF_R2": {}}
            for repeat in ("OFF_R1", "OFF_R2"):
                for branch in BRANCH_ORDER:
                    call_count += 1
                    processor.set_context(call_index=call_count, scheduler_index=5, branch=branch, active_axis=None, active_sign=None, amplitude=0.0, capture=True)
                    captures.clear()
                    embeddings = cond_embeddings if branch == "cond" else uncond_embeddings
                    velocity = transformer_forward_inference(torch, transformer, branch=branch, hidden_states=off_step5_latent.clone().to(transformer.dtype), timestep=timesteps[5].expand(1), encoder_hidden_states=embeddings)
                    validate_processor_record(processor.records[-1], active=False, captured=True)
                    block_rows, block_rms, block_full = captures[0]
                    three_off[repeat][branch] = {
                        "relation_bank": processor.records[-1]["relation_bank"], "block": block_rows.tolist(),
                        "velocity": _extract_velocity_slice(velocity).tolist(), "block_global_rms": block_rms,
                        "_block_full": block_full, "_velocity_full": velocity.detach().float().cpu(),
                    }
                    del velocity
        finally:
            hook.remove()
            target_block.attn1.set_processor(original)
        three_fit_records = run_calls(
            block, horizontal_sign, vertical_sign, "three_steps345", equal_fit_weights, three_off,
            fork_purpose="three_fit",
        )
        try:
            three_weight = solve_cfg_weights(_numeric_probe(three_fit_records, horizontal_sign, vertical_sign))
        except S1InstrumentationError as exc:
            three_weight = {"valid": False, "reason": str(exc)}
            three_evaluation = None
        else:
            three_weight["valid"] = True
            three_records = run_calls(
                block, horizontal_sign, vertical_sign, "three_steps345", three_weight["weights"], three_off,
                fork_purpose="three_apply",
            )
            three_evaluation = evaluate_records(three_records, horizontal_sign, vertical_sign, config["thresholds"])
            del three_records
        del three_fit_records
        supports = [
            {"support": "single_step4", "cfg": single_weight, "evaluation": single_evaluation, "eligible": bool(single_evaluation is not None and single_evaluation["all_cells_pass"])},
            {"support": "three_steps345", "cfg": three_weight, "evaluation": three_evaluation, "eligible": bool(three_evaluation is not None and three_evaluation["all_cells_pass"])},
        ]
        eligible = [item for item in supports if item["eligible"]]
        if eligible:
            def support_rank(item: Mapping[str, Any]) -> tuple[Any, ...]:
                evaluation = item["evaluation"]
                axes = [evaluation["cells"][f"velocity:{branch}"]["axes"][axis] for branch in (*BRANCH_ORDER, "guidance") for axis in AXES]
                return (-min(float(v["odd_to_noise_or_ulp"]) for v in axes), max(float(v["even_to_odd"]) for v in axes), max(float(v["temporal_total_variation"]) for v in axes), max(float(evaluation["cells"]["velocity:guidance"]["axes"][axis]["global_relative_rms"]) for axis in AXES), config["stage2"]["support_order"].index(item["support"]))
            eligible.sort(key=support_rank)
            selected = {"block": block, "horizontal_sign": horizontal_sign, "vertical_sign": vertical_sign, "support": eligible[0]["support"], "cfg_weights": eligible[0]["cfg"]["weights"]}
            status = STATUS_READY
        else:
            selected = None
            status = STATUS_NO_GO

    if call_count not in config["call_budget"]["possible_terminal_transformer_calls"]:
        raise S1InstrumentationError(f"finite call budget mismatch: {call_count}")
    stats = {
        "schema": SCHEMA,
        "diagnostic_class": DIAGNOSTIC_CLASS,
        "initial_latent_sha256": sha256_tensor(latent_at_step3),
        "scheduler_timesteps": actual_timesteps,
        "scheduler_snapshots": {
            "step3": scheduler_at_step3_summary,
            "step4": scheduler_at_step4_summary,
        },
        "scheduler_forks": scheduler_forks,
        "transformer_calls": call_count,
        "stage1": stage1,
        "stage2_supports": supports,
        "selected_construction": selected,
    }
    audit = {"schema": SCHEMA, "diagnostic_class": DIAGNOSTIC_CLASS, "status": status, "source": {"head": _git(repo_root, "rev-parse", "HEAD"), "tree": _git(repo_root, "rev-parse", "HEAD^{tree}"), "dirty": False}, "runtime": runtime, "transformer_calls": call_count, "VAE_was_run": False, "MP4_was_written": False, "S2_was_run": False, "formal_result": False, "stage_progression_allowed": False, "started_at": started, "ended_at": utc_now()}
    payloads = {"stats.json": stats, "audit.json": audit, "config.json": config, "plan.json": plan, "command.json": {"argv": list(argv), "cwd": str(cwd), "exit_code": propagation_exit_code(status)}}
    for name, payload in payloads.items():
        (output / name).write_bytes(canonical_json_bytes(payload) + b"\n")
    files = sorted(output.glob("*.json"))
    (output / "checksums.sha256").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    return {"status": status, "diagnostic_class": DIAGNOSTIC_CLASS, "output": str(output), "transformer_calls": call_count, "selected_construction": selected, "formal_result": False, "stage_progression_allowed": False}
