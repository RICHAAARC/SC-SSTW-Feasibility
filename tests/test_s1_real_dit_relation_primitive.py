from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sstw.relation_injector import (
    B1_KEY_PAIRS,
    B2_KEY_PAIRS,
    PAIR_COEFFICIENTS,
    TARGET_QUERY_INDICES,
    S1WanRelationProcessor,
    apply_sparse_logit_bias_numpy,
    install_s1_processor,
    relation_probability_contrasts_numpy,
    validate_frozen_sparse_geometry,
)
from sstw.s1_real_dit_relation_primitive import (
    BRANCH_ORDER,
    CONDITION_ORDER,
    LAYER_ORDER,
    evaluate_preregistered_statistics,
    load_frozen_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_authority_config_conditions_and_exact20_are_frozen() -> None:
    authority = ROOT / "SSTW_METHOD_AUTHORITY.md"
    assert hashlib.sha256(authority.read_bytes()).hexdigest() == "1a7d9b23e19f3414f400524e150bcaff3aad05ceab2193a44769fede0673430c"
    config, plan = load_frozen_inputs(ROOT)
    assert config["conditions"] == list(CONDITION_ORDER)
    assert config["call_budget"] == {
        "prefix_steps": [0, 1, 2, 3], "prefix_branches": ["cond", "uncond"],
        "prefix_transformer_calls": 8, "probe_step": 4, "probe_conditions": 6,
        "probe_branches": ["cond", "uncond"], "probe_transformer_calls": 12,
        "total_transformer_calls": 20, "scheduler_steps_executed": 4, "retry_count": 0,
    }
    assert plan["exact_transformer_calls"] == 20
    assert config["flow_support"] == {
        "scheduler_index": 4, "expected_timestep": 749, "support": [4], "lambda": 1.0,
        "inject_cond_branch": True, "inject_uncond_branch": True,
    }
    assert config["generation"]["prompt"] == "locked camera, dark matte background, a single bright white cube moving slowly across the center, simple studio lighting, no text, no cuts"
    assert config["generation"]["seed"] == 1275


def test_frozen_geometry_has_13_queries_two_orthogonal_unit_pairs() -> None:
    validate_frozen_sparse_geometry()
    assert TARGET_QUERY_INDICES == tuple(336 + 640 * index for index in range(13))
    assert all(pair == (query - 1, query + 1) for query, pair in zip(TARGET_QUERY_INDICES, B1_KEY_PAIRS, strict=True))
    assert all(pair == (query - 32, query + 32) for query, pair in zip(TARGET_QUERY_INDICES, B2_KEY_PAIRS, strict=True))
    assert sum(value * value for value in PAIR_COEFFICIENTS) == pytest.approx(1.0)
    assert sum(PAIR_COEFFICS := PAIR_COEFFICIENTS) == pytest.approx(0.0)
    assert set(B1_KEY_PAIRS[0]).isdisjoint(B2_KEY_PAIRS[0]) and len(PAIR_COEFFICS) == 2


def test_sparse_numpy_reference_changes_only_four_keys_and_is_odd() -> None:
    logits = np.zeros((3, 3, 8), dtype=np.float64)
    query = (1,)
    b1 = ((2, 3),)
    b2 = ((5, 6),)
    plus, plus_updates = apply_sparse_logit_bias_numpy(logits, query, b1, b2, (1.0, 1.0), 1.0)
    minus, minus_updates = apply_sparse_logit_bias_numpy(logits, query, b1, b2, (-1.0, -1.0), 1.0)
    changed = np.argwhere(plus != logits)
    assert changed.shape[0] == 3 * 4
    assert set(changed[:, 1]) == {1} and set(changed[:, 2]) == {2, 3, 5, 6}
    assert np.array_equal(plus + minus, logits * 2.0)
    assert all(left[0] == right[0] and left[2] + right[2] == pytest.approx(0.0) for left, right in zip(plus_updates[::2], plus_updates[1::2], strict=True))
    assert len(plus_updates) == len(minus_updates) == 4
    zero, updates = apply_sparse_logit_bias_numpy(logits, query, b1, b2, (0.0, 0.0), 1.0)
    assert np.array_equal(zero, logits) and updates == ()


def test_sparse_probability_jacobian_has_correct_sign_low_cross_and_common_mode() -> None:
    logits = np.zeros((2, 3, 8), dtype=np.float64)
    query, b1, b2 = (1,), ((2, 3),), ((5, 6),)
    responses = {}
    for name, state in {"p1": (1.0, 0.0), "m1": (-1.0, 0.0), "p2": (0.0, 1.0), "m2": (0.0, -1.0)}.items():
        biased, _ = apply_sparse_logit_bias_numpy(logits, query, b1, b2, state, 1.0)
        probabilities = np.exp(biased - biased.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        responses[name] = relation_probability_contrasts_numpy(probabilities, query, b1, b2)
    odd_1 = (responses["p1"] - responses["m1"]) / 2.0
    odd_2 = (responses["p2"] - responses["m2"]) / 2.0
    even_1 = (responses["p1"] + responses["m1"]) / 2.0
    even_2 = (responses["p2"] + responses["m2"]) / 2.0
    assert odd_1[..., 0].mean() > 0 and odd_2[..., 1].mean() > 0
    assert np.max(np.abs(odd_1[..., 1])) < 1e-15 and np.max(np.abs(odd_2[..., 0])) < 1e-15
    assert np.max(np.abs(even_1)) < 1e-15 and np.max(np.abs(even_2)) < 1e-15


def _synthetic_probe() -> dict[str, object]:
    probe: dict[str, object] = {}
    relation_basis = {
        "B1": np.tile(np.asarray([[[0.1, 0.0]]]), (13, 1, 1)),
        "B2": np.tile(np.asarray([[[0.0, 0.1]]]), (13, 1, 1)),
    }
    flat_basis = {
        "B1": np.tile(np.asarray([[0.1, 0.0]]), (13, 1)),
        "B2": np.tile(np.asarray([[0.0, 0.1]]), (13, 1)),
    }
    for condition in CONDITION_ORDER:
        if condition.endswith("B1"):
            sign, axis = (-1.0 if condition.startswith("MINUS") else 1.0), "B1"
        elif condition.endswith("B2"):
            sign, axis = (-1.0 if condition.startswith("MINUS") else 1.0), "B2"
        else:
            sign, axis = 0.0, "B1"
        probe[condition] = {}
        for branch in BRANCH_ORDER:
            base_relation = np.zeros((13, 1, 2))
            base_flat = np.ones((13, 2))
            probe[condition][branch] = {
                "relation": (base_relation + sign * relation_basis[axis]).tolist(),
                "block": (base_flat + sign * flat_basis[axis]).tolist(),
                "velocity": (base_flat + sign * flat_basis[axis]).tolist(),
            }
    return probe


def test_preregistered_O_E_N_gate_is_per_layer_branch_axis() -> None:
    config, _ = load_frozen_inputs(ROOT)
    overrides = {(branch, layer, axis): 0.005 for branch in (*BRANCH_ORDER, "guidance") for layer in ("block", "velocity") for axis in ("B1", "B2")}
    result = evaluate_preregistered_statistics(_synthetic_probe(), config["thresholds"], overrides)
    assert result["all_cells_pass"] is True
    assert set(result["cells"]) == {f"{layer}:{branch}" for branch in (*BRANCH_ORDER, "guidance") for layer in LAYER_ORDER}
    assert all(cell["axis_absolute_cosine"] < 0.5 and cell["passed"] for cell in result["cells"].values())


def test_fake_wan_adapter_installs_only_block14_processor() -> None:
    class Attention:
        def __init__(self) -> None:
            self.processor = object()

        def set_processor(self, value: object) -> None:
            self.processor = value

    class Block:
        def __init__(self) -> None:
            self.attn1 = Attention()

    class Transformer:
        def __init__(self) -> None:
            self.blocks = [Block() for _ in range(30)]

    transformer = Transformer()
    originals = [block.attn1.processor for block in transformer.blocks]
    processor, original = install_s1_processor(transformer, object(), block_index=14, strength=1.0)
    assert isinstance(processor, S1WanRelationProcessor) and original is originals[14]
    assert transformer.blocks[14].attn1.processor is processor
    assert all(block.attn1.processor is originals[index] for index, block in enumerate(transformer.blocks) if index != 14)


def test_real_processor_source_has_qkv_norm_rope_sparse_rows_and_no_proxy() -> None:
    source = inspect.getsource(S1WanRelationProcessor)
    for token in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.norm_q", "attn.norm_k", "_apply_rotary", "dispatch_attention_fn", "torch.softmax", "TARGET_QUERY_INDICES"):
        assert token in source
    assert "8320, 8320" not in source and "register_forward_hook" not in source
    forbidden = ("output_residual", "latent_carrier", "vae.decode", "pixel_carrier")
    assert all(token not in source.lower() for token in forbidden)


def test_runner_source_freezes_prefix_fork_and_no_VAE_or_MP4_execution() -> None:
    source = (ROOT / "src/sstw/s1_real_dit_relation_primitive.py").read_text(encoding="utf-8")
    assert "for scheduler_index in range(4)" in source
    assert "probe_latent = latents.detach().clone()" in source
    assert "for condition in CONDITION_ORDER" in source
    assert source.count("pipe.scheduler.step(") == 1
    assert ".vae.decode(" not in source and "encode_saved_mp4" not in source
