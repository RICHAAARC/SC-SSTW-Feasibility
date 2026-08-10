"""Sparse real-Wan antisymmetric Patch-relation injection.

The public dataclasses retain the minimal method contract.  The S1 processor
reproduces the diffusers 0.35.2 Wan self-attention path and only recomputes the
thirteen frozen query rows; it never materializes an 8320 by 8320 bias tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

Vector2 = tuple[float, float]


@dataclass(frozen=True)
class RelationBasis:
    query_index: int
    patch_a: int
    patch_b: int
    coefficient: float = 1.0

    def validate(self) -> None:
        if self.patch_a == self.patch_b or self.query_index < 0 or min(self.patch_a, self.patch_b) < 0:
            raise ValueError("basis must specify distinct non-negative patch indices")


@dataclass(frozen=True)
class InjectionSpec:
    basis_1: RelationBasis
    basis_2: RelationBasis
    strength: float
    active_flow_steps: frozenset[int]


def relation_bias_updates(state: Vector2, flow_step: int, spec: InjectionSpec) -> tuple[tuple[int, int, float], ...]:
    """Return only the antisymmetric logit updates for the frozen interface."""
    if flow_step not in spec.active_flow_steps:
        return ()
    spec.basis_1.validate()
    spec.basis_2.validate()
    updates: list[tuple[int, int, float]] = []
    for amplitude, basis in zip(state, (spec.basis_1, spec.basis_2), strict=True):
        delta = spec.strength * amplitude * basis.coefficient
        updates.extend(((basis.query_index, basis.patch_a, delta), (basis.query_index, basis.patch_b, -delta)))
    return tuple(updates)


TARGET_QUERY_INDICES = (336, 976, 1616, 2256, 2896, 3536, 4176, 4816, 5456, 6096, 6736, 7376, 8016)
B1_KEY_PAIRS = tuple((query - 1, query + 1) for query in TARGET_QUERY_INDICES)
B2_KEY_PAIRS = tuple((query - 32, query + 32) for query in TARGET_QUERY_INDICES)
PAIR_COEFFICIENTS = (1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0))


def validate_frozen_sparse_geometry(token_count: int = 8320) -> None:
    if len(TARGET_QUERY_INDICES) != 13 or len(set(TARGET_QUERY_INDICES)) != 13:
        raise ValueError("frozen target-query geometry changed")
    for query, horizontal, vertical in zip(TARGET_QUERY_INDICES, B1_KEY_PAIRS, B2_KEY_PAIRS, strict=True):
        if not (0 <= query < token_count) or len({query, *horizontal, *vertical}) != 5:
            raise ValueError("frozen local relation geometry is invalid")
        if horizontal != (query - 1, query + 1) or vertical != (query - 32, query + 32):
            raise ValueError("frozen relation pairs are not local")
    if abs(sum(value * value for value in PAIR_COEFFICIENTS) - 1.0) > 1e-15 or abs(sum(PAIR_COEFFICIENTS)) > 1e-15:
        raise ValueError("frozen antisymmetric coefficients changed")


def apply_sparse_logit_bias_numpy(
    logits: np.ndarray,
    query_indices: Sequence[int],
    b1_pairs: Sequence[Sequence[int]],
    b2_pairs: Sequence[Sequence[int]],
    state: Vector2,
    strength: float,
) -> tuple[np.ndarray, tuple[tuple[int, int, float], ...]]:
    """Reference sparse-logit update used by CPU-only formula tests."""

    source = np.asarray(logits)
    if source.ndim != 3 or source.shape[1] <= max(query_indices) or source.shape[2] <= max(max(pair) for pair in (*b1_pairs, *b2_pairs)):
        raise ValueError("logits do not contain the frozen rows and keys")
    if len(query_indices) != len(b1_pairs) or len(query_indices) != len(b2_pairs):
        raise ValueError("sparse geometry lengths differ")
    output = source.copy()
    updates: list[tuple[int, int, float]] = []
    for row, pair_1, pair_2 in zip(query_indices, b1_pairs, b2_pairs, strict=True):
        for amplitude, pair in zip(state, (pair_1, pair_2), strict=True):
            for key, coefficient in zip(pair, PAIR_COEFFICIENTS, strict=True):
                delta = float(strength) * float(amplitude) * coefficient
                output[:, row, key] += delta
                if delta != 0.0:
                    updates.append((int(row), int(key), delta))
    return output, tuple(updates)


def relation_probability_contrasts_numpy(
    probabilities: np.ndarray,
    query_indices: Sequence[int],
    b1_pairs: Sequence[Sequence[int]],
    b2_pairs: Sequence[Sequence[int]],
) -> np.ndarray:
    """Return Q by H by 2 local pair-probability contrasts."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("probabilities must be H by Q by K")
    records = []
    for query, pair_1, pair_2 in zip(query_indices, b1_pairs, b2_pairs, strict=True):
        records.append(
            np.stack(
                (
                    values[:, query, pair_1[0]] - values[:, query, pair_1[1]],
                    values[:, query, pair_2[0]] - values[:, query, pair_2[1]],
                ),
                axis=-1,
            )
        )
    return np.stack(records, axis=0)


def selected_bias_updates(state: Vector2, strength: float) -> tuple[tuple[int, int, int, float], ...]:
    """Return row-local additive-mask updates for the thirteen frozen queries."""

    validate_frozen_sparse_geometry()
    updates: list[tuple[int, int, int, float]] = []
    for row_offset, (query, pair_1, pair_2) in enumerate(
        zip(TARGET_QUERY_INDICES, B1_KEY_PAIRS, B2_KEY_PAIRS, strict=True)
    ):
        for amplitude, pair in zip(state, (pair_1, pair_2), strict=True):
            for key_index, coefficient in zip(pair, PAIR_COEFFICIENTS, strict=True):
                delta = float(strength) * float(amplitude) * coefficient
                if delta != 0.0:
                    updates.append((row_offset, query, key_index, delta))
    return tuple(updates)


def dispatch_native_selected_attention(
    dispatch_attention_fn: Any,
    target_query: Any,
    key: Any,
    value: Any,
    additive_bias: Any,
) -> Any:
    """Run the native diffusers/PyTorch SDPA path for only selected query rows."""

    if tuple(target_query.shape[:1] + target_query.shape[2:]) != tuple(key.shape[:1] + key.shape[2:]):
        raise ValueError("selected query and key batch/head dimensions differ")
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("key and value shapes differ")
    if tuple(additive_bias.shape) != (1, 1, int(target_query.shape[1]), int(key.shape[1])):
        raise ValueError("selected additive bias is not broadcastable [1,1,Q,K]")
    return dispatch_attention_fn(
        target_query,
        key,
        value,
        attn_mask=additive_bias,
        dropout_p=0.0,
        is_causal=False,
        backend="native",
    )


def replace_selected_rows(full_heads: Any, selected_heads: Any, query_index: Any, *, active: bool) -> Any:
    """Keep OFF byte-for-byte on the full Wan output; replace only active rows."""

    if not active:
        return full_heads
    replacement = full_heads.clone()
    replacement[:, query_index] = selected_heads.to(dtype=full_heads.dtype)
    return replacement


def _apply_rotary(torch: Any, hidden_states: Any, rotary_emb: Any) -> Any:
    freqs_cos, freqs_sin = rotary_emb
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = x1 * cos - x2 * sin
    output[..., 1::2] = x1 * sin + x2 * cos
    return output.type_as(hidden_states)


class S1WanRelationProcessor:
    """Version-locked Wan self-attention processor with sparse row replacement."""

    def __init__(self, torch_module: Any, *, strength: float = 1.0) -> None:
        validate_frozen_sparse_geometry()
        self.torch = torch_module
        self.strength = float(strength)
        self.context: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def set_context(self, *, call_index: int, scheduler_index: int, timestep: int, condition: str, branch: str, state: Vector2) -> None:
        if branch not in {"cond", "uncond"} or len(state) != 2:
            raise ValueError("invalid S1 processor context")
        self.context = {
            "call_index": int(call_index), "scheduler_index": int(scheduler_index), "timestep": int(timestep),
            "condition": str(condition), "branch": branch, "state": (float(state[0]), float(state[1])),
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
            raise RuntimeError("S1 processor context was not set")
        if encoder_hidden_states is not None or attention_mask is not None or rotary_emb is None:
            raise RuntimeError("S1 processor only supports version-locked Wan self-attention")
        if getattr(attn, "fused_projections", False):
            raise RuntimeError("fused Wan projections are outside the frozen S1 path")
        torch = self.torch
        grad_enabled = bool(torch.is_grad_enabled())
        inference_mode_enabled = bool(torch.is_inference_mode_enabled())
        try:
            from diffusers.models.attention_dispatch import dispatch_attention_fn
        except Exception as exc:  # pragma: no cover - production dependency boundary
            raise RuntimeError("diffusers 0.35.2 attention dispatcher unavailable") from exc

        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        query = _apply_rotary(torch, query, rotary_emb)
        key = _apply_rotary(torch, key, rotary_emb)
        if tuple(query.shape[1:]) != (8320, 12, 128) or tuple(key.shape) != tuple(query.shape) or tuple(value.shape) != tuple(query.shape):
            raise RuntimeError("Wan QKV identity is not [B,8320,12,128]")

        full_heads = dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None)
        query_index = torch.tensor(TARGET_QUERY_INDICES, device=query.device, dtype=torch.long)
        target_query = query.index_select(1, query_index)
        diagnostic_logits = torch.einsum("bqhd,bkhd->bhqk", target_query.float(), key.float()) / math.sqrt(128.0)
        state = self.context["state"]
        sparse_updates = selected_bias_updates(state, self.strength)
        additive_bias = torch.zeros(
            (1, 1, len(TARGET_QUERY_INDICES), int(key.shape[1])),
            device=query.device,
            dtype=query.dtype,
        )
        bias_updates: list[dict[str, float | int]] = []
        for row_offset, query_index_value, key_index, delta in sparse_updates:
            diagnostic_logits[:, :, row_offset, key_index] += delta
            additive_bias[:, :, row_offset, key_index] += delta
            bias_updates.append({"query": query_index_value, "key": key_index, "delta": delta})
        if not bool(torch.isfinite(additive_bias).all().item()):
            raise RuntimeError("selected additive bias is non-finite")
        probabilities = torch.softmax(diagnostic_logits, dim=-1)
        if not bool(torch.isfinite(probabilities).all().item()):
            raise RuntimeError("manual relation observation is non-finite")
        selected_heads = dispatch_native_selected_attention(
            dispatch_attention_fn, target_query, key, value, additive_bias
        )
        if tuple(selected_heads.shape) != tuple(target_query.shape) or not bool(torch.isfinite(selected_heads).all().item()):
            raise RuntimeError("native selected-row SDPA output is invalid")
        reference = full_heads.index_select(1, query_index).float()
        difference = selected_heads.float() - reference
        reference_rms = reference.square().mean().sqrt()
        relative_rms = float((difference.square().mean().sqrt() / torch.clamp(reference_rms, min=1e-30)).item())
        global_ulp = float((torch.clamp(reference.abs().max(), min=1e-30) * (2.0 ** -7)).item())
        max_ulp = float(difference.abs().max().item() / max(global_ulp, 2.0 ** -133))

        pair_records = []
        for row_offset, (pair_1, pair_2) in enumerate(zip(B1_KEY_PAIRS, B2_KEY_PAIRS, strict=True)):
            pair_records.append(
                torch.stack(
                    (
                        probabilities[:, :, row_offset, pair_1[0]] - probabilities[:, :, row_offset, pair_1[1]],
                        probabilities[:, :, row_offset, pair_2[0]] - probabilities[:, :, row_offset, pair_2[1]],
                    ), dim=-1,
                )
            )
        relation = torch.stack(pair_records, dim=1).mean(dim=0).detach().float().cpu().tolist()

        active = state != (0.0, 0.0)
        full_heads = replace_selected_rows(full_heads, selected_heads, query_index, active=active)
        output = full_heads.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        record = dict(self.context)
        record.update({
            "qkv_shape": [int(value) for value in query.shape], "qkv_dtype": str(query.dtype),
            "relation": relation, "bias_updates": bias_updates,
            "pair_sums_zero": all(abs(left["delta"] + right["delta"]) <= 1e-12 for left, right in zip(bias_updates[::2], bias_updates[1::2], strict=True)),
            "changed_logit_count": len(bias_updates), "dense_bias_materialized": False,
            "selected_bias_shape": [1, 1, len(TARGET_QUERY_INDICES), int(key.shape[1])],
            "selected_native_backend": "native",
            "selected_rows_replaced": len(TARGET_QUERY_INDICES) if active else 0,
            "lambda_zero_reference_relative_rms": relative_rms,
            "lambda_zero_reference_max_bfloat16_ulp": max_ulp,
            "non_target_rows_replaced": 0,
            "grad_enabled": grad_enabled,
            "inference_mode_enabled": inference_mode_enabled,
        })
        self.records.append(record)
        self.context = None
        return output


def install_s1_processor(transformer: Any, torch_module: Any, *, block_index: int = 14, strength: float = 1.0) -> tuple[S1WanRelationProcessor, Any]:
    """Install only on the frozen Wan self-attention callsite."""

    if block_index != 14 or not hasattr(transformer, "blocks") or len(transformer.blocks) != 30:
        raise ValueError("frozen block identity is not available")
    attention = transformer.blocks[block_index].attn1
    if not hasattr(attention, "set_processor") or not hasattr(attention, "processor"):
        raise ValueError("Wan attention processor interface unavailable")
    original = attention.processor
    processor = S1WanRelationProcessor(torch_module, strength=strength)
    attention.set_processor(processor)
    return processor, original
