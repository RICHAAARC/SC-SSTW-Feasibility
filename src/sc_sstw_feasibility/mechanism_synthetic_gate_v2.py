"""Protocol v2 Gate 1B synthetic mechanism chain.

This module is synthetic evidence only. Acquisition accepts observations and
public configuration only. Truth provenance is used after acquisition solely
to measure coverage; it never enters candidate selection or calibration.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

from .aisb import BurstCandidate, BurstTemplate, best_non_overlapping_sequence, scan_burst_candidates
from .calibration import CalibrationResult, calibrate_from_pilot_pairs, equalize_observations
from .linalg import condition_number_2d_columns, matvec
from .sync import dynamic_time_sync


Vector2 = tuple[float, float]


class ProtocolError(RuntimeError):
    """Raised when the frozen protocol is missing or internally inconsistent."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _write_canonical(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    path.write_bytes(data + b"\n")
    return hashlib.sha256(data).hexdigest()


def _derived_seed(config: dict[str, Any], domain: str, case_index: int) -> int:
    seed_config = config["synthetic_protocol"]["seed_derivation"]
    if domain not in seed_config["domains"]:
        raise ProtocolError(f"undeclared seed domain: {domain}")
    payload = f"{seed_config['master']}|{domain}|{case_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _rng(seed: int) -> random.Random:
    rng = random.Random()
    rng.seed(seed, version=2)
    return rng


def load_and_validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    protocol = config["synthetic_protocol"]
    if config["claim_scope"] != "mechanism_feasibility_only":
        raise ProtocolError("unexpected claim scope")
    if protocol["observation_dimension"] != 2:
        raise ProtocolError("synthetic observation must be exactly two-dimensional")
    if protocol["channel"]["matrix_shape"] != [2, 2]:
        raise ProtocolError("synthetic channel must be 2x2")
    if protocol["null_protocol"]["observation_dimension"] != 2:
        raise ProtocolError("null observations must be exactly two-dimensional")
    points = protocol["template"]["points"]
    if len(points) != 12 or protocol["template"]["allowed_missing_counts"] != [1, 2]:
        raise ProtocolError("frozen double-redundant template contract mismatch")
    acquisition = protocol["acquisition"]
    expected = {"12": 1, "11": 12, "10": 66}
    if acquisition["expected_candidate_count_per_start_by_observed_length"] != expected:
        raise ProtocolError("candidate-count contract mismatch")
    if acquisition["top_k_per_start_per_observed_length"] < max(expected.values()):
        raise ProtocolError("candidate truncation is forbidden")
    if acquisition["candidate_truncation"] != "none" or acquisition["candidate_overflow"] != "fail":
        raise ProtocolError("candidate truncation/overflow contract mismatch")
    if acquisition["candidate_key_access"] != "forbidden":
        raise ProtocolError("acquisition must be key-independent")
    artifact = protocol["ambiguity_artifact"]
    if artifact["calibration_input"] != "readback_frozen_ambiguity_artifact":
        raise ProtocolError("calibration must read back the frozen ambiguity artifact")
    if artifact["candidate_key_access"] != "forbidden":
        raise ProtocolError("ambiguity construction must be key-independent")
    segments = protocol["sequence"]["segments"]
    if sum(int(segment["length"]) for segment in segments) != protocol["sequence"]["total_window_count_before_edit"]:
        raise ProtocolError("sequence length mismatch")
    cases = protocol["case_protocols"]
    if [case["case_index"] for case in cases] != list(range(config["gate_1b"]["case_count"])):
        raise ProtocolError("case indices must be complete and ordered")
    keys = config["candidate_keys"]
    if len(keys["wrong_key_ids"]) != config["gate_1b"]["wrong_key_count"]:
        raise ProtocolError("wrong-key count mismatch")
    if protocol["scoring"]["search_truncation"] != "none":
        raise ProtocolError("key scoring must be exhaustive")
    for domain, expected_seed in protocol["seed_derivation"]["golden_case_0"].items():
        if _derived_seed(config, domain, 0) != expected_seed:
            raise ProtocolError(f"seed derivation mismatch for {domain}")
    required = {"burst_interior_single_deletion", "burst_interior_double_deletion", "private_frame_deletion", "private_frame_duplication"}
    if {case["edit"] for case in cases} != required:
        raise ProtocolError("required edit cases are incomplete")
    return config


def _template(config: dict[str, Any]) -> BurstTemplate:
    record = config["synthetic_protocol"]["template"]
    return BurstTemplate(
        template_id=record["template_id"],
        points=tuple((float(point["xy"][0]), float(point["xy"][1])) for point in record["points"]),
    )


def _private_state(key: str, global_index: int, phase: float, base: float, step: float) -> tuple[Vector2, float]:
    digest = hashlib.sha256(f"{key}:{global_index}".encode("utf-8")).digest()
    sign = 1 if digest[0] & 1 else -1
    next_phase = phase + base + step * sign
    return (math.cos(next_phase), math.sin(next_phase)), next_phase


def _initial_phase(key: str) -> float:
    digest = hashlib.sha256(f"{key}:phase".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed).random() * 2.0 * math.pi


def build_state_records(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    protocol = config["synthetic_protocol"]
    template_points = protocol["template"]["points"]
    private = protocol["sequence"]["private_state_generator"]
    phase = _initial_phase(key)
    records: list[dict[str, Any]] = []
    for segment in protocol["sequence"]["segments"]:
        if segment["kind"] == "public":
            if segment["length"] != len(template_points):
                raise ProtocolError("public segment length must equal template length")
            for template_index, point in enumerate(template_points):
                records.append(
                    {
                        "source_index": len(records),
                        "segment_id": segment["segment_id"],
                        "kind": "public",
                        "template_index": template_index,
                        "logical_state_id": point["logical_state_id"],
                        "state": [float(point["xy"][0]), float(point["xy"][1])],
                    }
                )
        elif segment["kind"] == "private":
            for private_offset in range(segment["length"]):
                state, phase = _private_state(
                    key,
                    len(records),
                    phase,
                    float(private["base_frequency"]),
                    float(private["phase_step"]),
                )
                records.append(
                    {
                        "source_index": len(records),
                        "segment_id": segment["segment_id"],
                        "kind": "private",
                        "private_offset": private_offset,
                        "state": [state[0], state[1]],
                    }
                )
        else:
            raise ProtocolError(f"unsupported segment kind: {segment['kind']}")
    return records


def _channel(config: dict[str, Any], case_index: int) -> tuple[list[list[float]], list[float], float]:
    case = config["synthetic_protocol"]["channel"]["case_parameters"][case_index]
    if case["case_index"] != case_index:
        raise ProtocolError("constructed channel case mapping mismatch")
    c1, t1 = math.cos(case["theta_1"]), math.sin(case["theta_1"])
    c2, t2 = math.cos(case["theta_2"]), math.sin(case["theta_2"])
    s1, s2 = float(case["s_1"]), float(case["s_2"])
    matrix = [
        [c1 * s1 * c2 - t1 * s2 * t2, -c1 * s1 * t2 - t1 * s2 * c2],
        [t1 * s1 * c2 + c1 * s2 * t2, -t1 * s1 * t2 + c1 * s2 * c2],
    ]
    bias = [float(value) for value in case["bias"]]
    return matrix, bias, condition_number_2d_columns(matrix)


def observe_records(config: dict[str, Any], records: list[dict[str, Any]], case_index: int) -> list[dict[str, Any]]:
    matrix, bias, channel_condition = _channel(config, case_index)
    threshold = float(config["saved_video_gate"]["maximum_channel_condition_number"])
    if not math.isfinite(channel_condition) or channel_condition > threshold:
        raise ProtocolError(
            f"case {case_index} frozen channel condition {channel_condition} exceeds {threshold}; no resampling allowed"
        )
    noise_std = float(config["synthetic_protocol"]["channel"]["noise_std"])
    noise = _rng(_derived_seed(config, "synthetic.noise", case_index))
    observed: list[dict[str, Any]] = []
    for record in records:
        clean = matvec(matrix, record["state"])
        q = [clean[index] + bias[index] + noise.gauss(0.0, noise_std) for index in range(2)]
        observed.append({**record, "q": q})
    return observed


def apply_declared_edit(records: list[dict[str, Any]], case: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[int] = []
    for index, record in enumerate(records):
        if record["segment_id"] != case["segment_id"]:
            continue
        if "template_indices" in case and record.get("template_index") in case["template_indices"]:
            targets.append(index)
        if "private_offset" in case and record.get("private_offset") == case["private_offset"]:
            targets.append(index)
    expected = len(case.get("template_indices", [case.get("private_offset")]))
    if len(targets) != expected:
        raise ProtocolError(f"edit target mismatch for case {case['case_index']}: {targets}")
    if case["edit"].endswith("deletion"):
        target_set = set(targets)
        return [record.copy() for index, record in enumerate(records) if index not in target_set]
    if case["edit"] == "private_frame_duplication":
        target = targets[0]
        edited: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            edited.append(record.copy())
            if index == target:
                duplicate = record.copy()
                duplicate["duplicated_observation"] = True
                edited.append(duplicate)
        return edited
    raise ProtocolError(f"unsupported edit: {case['edit']}")


def acquire_public_ambiguity(observations: list[list[float]], config: dict[str, Any]) -> list[BurstCandidate]:
    """Key-independent public acquisition; no candidate key is accepted."""

    protocol = config["synthetic_protocol"]
    acquisition = protocol["acquisition"]
    candidates = scan_burst_candidates(
        observations,
        (_template(config),),
        top_k_per_start=int(acquisition["top_k_per_start_per_observed_length"]),
        allow_single_deletion=True,
        allow_double_deletion=True,
    )
    return best_non_overlapping_sequence(
        candidates,
        burst_length=12,
        residual_threshold=float(acquisition["residual_threshold"]),
        maximize_count=bool(acquisition["maximize_non_overlapping_count"]),
    )


def _missing_list(candidate: BurstCandidate) -> list[int]:
    missing = candidate.missing_template_index
    if missing is None:
        return []
    if isinstance(missing, int):
        return [missing]
    return list(missing)


def _candidate_payload(candidate: BurstCandidate) -> dict[str, Any]:
    return {
        "start_index": candidate.start_index,
        "template_id": candidate.template_id,
        "residual": candidate.residual,
        "observed_length": candidate.observed_length,
        "missing_indices": _missing_list(candidate),
    }


def _candidate_from_payload(payload: dict[str, Any]) -> BurstCandidate:
    missing_values = payload["missing_indices"]
    missing: int | tuple[int, ...] | None
    if not missing_values:
        missing = None
    elif len(missing_values) == 1:
        missing = int(missing_values[0])
    else:
        missing = tuple(int(value) for value in missing_values)
    return BurstCandidate(
        start_index=int(payload["start_index"]),
        template_id=str(payload["template_id"]),
        residual=float(payload["residual"]),
        observed_length=int(payload["observed_length"]),
        missing_template_index=missing,
    )


def _canonical_candidate_payloads(candidates: list[BurstCandidate]) -> list[dict[str, Any]]:
    payloads = [_candidate_payload(candidate) for candidate in candidates]
    return sorted(
        payloads,
        key=lambda item: (
            item["start_index"],
            item["observed_length"],
            item["template_id"],
            item["missing_indices"],
            item["residual"],
        ),
    )


def _candidate_template_observations(
    candidate: BurstCandidate,
    edited: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[tuple[dict[str, Any], list[float]]]:
    points = config["synthetic_protocol"]["template"]["points"]
    missing = set(_missing_list(candidate))
    retained_indices = [index for index in range(len(points)) if index not in missing]
    window = edited[candidate.start_index : candidate.start_index + candidate.observed_length]
    if len(window) != len(retained_indices):
        raise ProtocolError("candidate window/template mapping mismatch")
    return [(points[template_index], record["q"]) for template_index, record in zip(retained_indices, window, strict=True)]


def _semantic_truth_coverage(
    candidates: list[BurstCandidate],
    edited: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[int, int, int]:
    public_segments = {
        segment["segment_id"]
        for segment in config["synthetic_protocol"]["sequence"]["segments"]
        if segment["kind"] == "public"
    }
    template_length = len(config["synthetic_protocol"]["template"]["points"])
    covered_segments: set[str] = set()
    valid_candidate_count = 0
    for candidate in candidates:
        missing = set(_missing_list(candidate))
        implied = [index for index in range(template_length) if index not in missing]
        window = edited[candidate.start_index : candidate.start_index + candidate.observed_length]
        segment_ids = {record["segment_id"] for record in window}
        actual = [record.get("template_index") for record in window]
        valid = (
            len(window) == candidate.observed_length
            and len(segment_ids) == 1
            and next(iter(segment_ids), None) in public_segments
            and all(record.get("kind") == "public" for record in window)
            and actual == implied
        )
        if valid:
            valid_candidate_count += 1
            covered_segments.update(segment_ids)
    return len(covered_segments), len(public_segments), len(candidates) - valid_candidate_count


def _calibration_and_heldout(
    candidates: list[BurstCandidate],
    edited: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[CalibrationResult, float, int, int]:
    schedule = config["state_schedule"]
    calibration_ids = set(schedule["public_calibration_point_ids"])
    held_out_ids = set(schedule["public_held_out_point_ids"])
    calibration_pairs: list[tuple[Vector2, list[float]]] = []
    heldout_pairs: list[tuple[Vector2, list[float]]] = []
    for candidate in candidates:
        for point, observation in _candidate_template_observations(candidate, edited, config):
            logical_id = point["logical_state_id"]
            state = (float(point["xy"][0]), float(point["xy"][1]))
            if logical_id in calibration_ids:
                calibration_pairs.append((state, observation))
            elif logical_id in held_out_ids:
                heldout_pairs.append((state, observation))
    occurrence = config["synthetic_protocol"]["public_occurrence_aggregation"]
    if len(calibration_pairs) < occurrence["minimum_retained_calibration_rows"]:
        raise ProtocolError("insufficient retained public calibration rows")
    if len(heldout_pairs) < occurrence["minimum_retained_held_out_rows"]:
        raise ProtocolError("insufficient retained public held-out rows")
    calibration = calibrate_from_pilot_pairs(calibration_pairs)
    estimated = equalize_observations([observation for _, observation in heldout_pairs], calibration)
    heldout_error = sum(
        (estimate[0] - target[0]) ** 2 + (estimate[1] - target[1]) ** 2
        for estimate, (target, _observation) in zip(estimated, heldout_pairs, strict=True)
    ) / len(heldout_pairs)
    return calibration, heldout_error, len(calibration_pairs), len(heldout_pairs)


def _calibration_payload(calibration: CalibrationResult) -> dict[str, Any]:
    return {
        "matrix": calibration.matrix,
        "bias": calibration.bias,
        "condition_number": calibration.condition_number,
        "public_fit_mse": calibration.pilot_reconstruction_mse,
    }


def _calibration_from_payload(payload: dict[str, Any]) -> CalibrationResult:
    return CalibrationResult(
        matrix=[[float(value) for value in row] for row in payload["matrix"]],
        bias=[float(value) for value in payload["bias"]],
        condition_number=float(payload["condition_number"]),
        pilot_reconstruction_mse=float(payload["public_fit_mse"]),
    )


def _score_all_keys(
    edited: list[dict[str, Any]],
    calibration: CalibrationResult,
    config: dict[str, Any],
) -> tuple[str, float, list[dict[str, Any]]]:
    q = [record["q"] for record in edited]
    equalized = equalize_observations(q, calibration)
    keys = config["candidate_keys"]
    candidate_keys = [keys["owner_key_id"], *keys["wrong_key_ids"]]
    scoring = config["synthetic_protocol"]["scoring"]
    results: list[dict[str, Any]] = []
    for key in candidate_keys:
        candidate_states = [tuple(record["state"]) for record in build_state_records(config, key)]
        result = dynamic_time_sync(
            equalized,
            candidate_states,
            skip_penalty=float(scoring["skip_penalty"]),
            repeat_penalty=float(scoring["repeat_penalty"]),
        )
        results.append({"key_id": key, "score": result.score})
    owner_score = results[0]["score"]
    best_wrong = max(results[1:], key=lambda item: item["score"])
    return str(best_wrong["key_id"]), float(owner_score - best_wrong["score"]), results


def _null_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    null = config["synthetic_protocol"]["null_protocol"]
    results = []
    for case_index in range(null["case_count"]):
        rng = _rng(_derived_seed(config, "synthetic.null.observation", case_index))
        observations = [
            [rng.gauss(0.0, 1.0) for _dimension in range(2)]
            for _window in range(null["window_count"])
        ]
        accepted = acquire_public_ambiguity(observations, config)
        results.append(
            {
                "null_case_index": case_index,
                "accepted_count": len(accepted),
                "false_acquisition": bool(accepted),
            }
        )
    return results


def _run_gate_core(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_and_validate_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_digest = _write_canonical(output_dir / "config.json", config)
    owner_key = config["candidate_keys"]["owner_key_id"]
    case_results: list[dict[str, Any]] = []
    truth_numerator = 0
    truth_denominator = 0
    for case in config["synthetic_protocol"]["case_protocols"]:
        case_index = int(case["case_index"])
        case_dir = output_dir / "artifacts" / f"case_{case_index:02d}"
        state_records = build_state_records(config, owner_key)
        observed = observe_records(config, state_records, case_index)
        edited = apply_declared_edit(observed, case)
        observation_payload = {
            "case_index": case_index,
            "edit": case["edit"],
            "q": [record["q"] for record in edited],
        }
        observation_digest = _write_canonical(case_dir / "observation.json", observation_payload)

        accepted = acquire_public_ambiguity(observation_payload["q"], config)
        ambiguity_payload = {
            "case_index": case_index,
            "observation_digest": observation_digest,
            "candidates": _canonical_candidate_payloads(accepted),
        }
        ambiguity_digest = _write_canonical(case_dir / "ambiguity_set.json", ambiguity_payload)
        readback = json.loads((case_dir / "ambiguity_set.json").read_text(encoding="utf-8"))
        if canonical_digest(readback) != ambiguity_digest:
            raise ProtocolError("ambiguity readback digest mismatch")
        frozen_candidates = [_candidate_from_payload(payload) for payload in readback["candidates"]]

        covered, truth_count, false_candidate_count = _semantic_truth_coverage(
            frozen_candidates, edited, config
        )
        truth_numerator += covered
        truth_denominator += truth_count

        calibration, heldout_mse, calibration_rows, heldout_rows = _calibration_and_heldout(
            frozen_candidates, edited, config
        )
        calibration_payload = {
            **_calibration_payload(calibration),
            "case_index": case_index,
            "observation_digest": observation_digest,
            "ambiguity_digest": ambiguity_digest,
            "calibration_logical_ids": config["state_schedule"]["public_calibration_point_ids"],
            "held_out_logical_ids": config["state_schedule"]["public_held_out_point_ids"],
            "calibration_row_count": calibration_rows,
            "held_out_row_count": heldout_rows,
            "held_out_mse": heldout_mse,
        }
        calibration_digest = _write_canonical(case_dir / "calibration.json", calibration_payload)
        calibration_readback = json.loads((case_dir / "calibration.json").read_text(encoding="utf-8"))
        frozen_calibration = _calibration_from_payload(calibration_readback)

        best_wrong_key, margin, scores = _score_all_keys(edited, frozen_calibration, config)
        owner_score = scores[0]["score"]
        best_wrong_score = max(item["score"] for item in scores[1:])
        scores_payload = {
            "case_index": case_index,
            "observation_digest": observation_digest,
            "ambiguity_digest": ambiguity_digest,
            "calibration_digest": calibration_digest,
            "owner_key_id": owner_key,
            "best_wrong_key_id": best_wrong_key,
            "owner_score": owner_score,
            "best_wrong_score": best_wrong_score,
            "score_margin": margin,
            "scores": scores,
        }
        scores_digest = _write_canonical(case_dir / "scores.json", scores_payload)
        case_results.append(
            {
                "case_index": case_index,
                "edit": case["edit"],
                "truth_count": truth_count,
                "truth_covered": covered,
                "accepted_count": len(frozen_candidates),
                "false_candidate_count": false_candidate_count,
                "channel_condition_number": calibration.condition_number,
                "public_fit_mse": calibration.pilot_reconstruction_mse,
                "public_held_out_mse": heldout_mse,
                "owner_score": owner_score,
                "best_wrong_score": best_wrong_score,
                "score_margin": margin,
                "owner_strictly_best": margin > 0.0,
                "observation_digest": observation_digest,
                "ambiguity_digest": ambiguity_digest,
                "calibration_digest": calibration_digest,
                "scores_digest": scores_digest,
            }
        )

    null_cases = _null_cases(config)
    _write_canonical(output_dir / "artifacts" / "null_cases.json", null_cases)
    gate = config["gate_1b"]
    saved = config["saved_video_gate"]
    truth_coverage = truth_numerator / truth_denominator if truth_denominator else math.nan
    false_acquisition_rate = sum(case["false_acquisition"] for case in null_cases) / len(null_cases) if null_cases else math.nan
    owner_positive_fraction = sum(case["owner_strictly_best"] for case in case_results) / len(case_results) if case_results else math.nan
    minimum_margin = min(case["score_margin"] for case in case_results)
    maximum_condition = max(case["channel_condition_number"] for case in case_results)
    maximum_heldout_mse = max(case["public_held_out_mse"] for case in case_results)
    checks = [
        {"name": "truth_coverage", "value": truth_coverage, "comparison": ">=", "threshold": gate["minimum_truth_coverage"], "pass": truth_coverage >= gate["minimum_truth_coverage"]},
        {"name": "false_acquisition_rate", "value": false_acquisition_rate, "comparison": "<=", "threshold": gate["maximum_false_acquisition_rate"], "pass": false_acquisition_rate <= gate["maximum_false_acquisition_rate"]},
        {"name": "owner_positive_fraction", "value": owner_positive_fraction, "comparison": ">=", "threshold": gate["minimum_owner_positive_fraction"], "pass": owner_positive_fraction >= gate["minimum_owner_positive_fraction"]},
        {"name": "minimum_score_margin", "value": minimum_margin, "comparison": ">=", "threshold": gate["minimum_score_margin"], "pass": minimum_margin >= gate["minimum_score_margin"]},
        {"name": "maximum_channel_condition_number", "value": maximum_condition, "comparison": "<=", "threshold": saved["maximum_channel_condition_number"], "pass": maximum_condition <= saved["maximum_channel_condition_number"]},
        {"name": "maximum_public_held_out_mse", "value": maximum_heldout_mse, "comparison": "<=", "threshold": saved["maximum_public_held_out_mse"], "pass": maximum_heldout_mse <= saved["maximum_public_held_out_mse"]},
    ]
    metrics = {
        "evidence_kind": "frozen_2d_synthetic_only_no_video_no_gpu",
        "config_digest": config_digest,
        "case_count": len(case_results),
        "null_case_count": len(null_cases),
        "truth_coverage": truth_coverage,
        "false_acquisition_rate": false_acquisition_rate,
        "owner_positive_fraction": owner_positive_fraction,
        "minimum_score_margin": minimum_margin,
        "maximum_channel_condition_number": maximum_condition,
        "maximum_public_held_out_mse": maximum_heldout_mse,
        "checks": checks,
        "cases": case_results,
        "gate_pass": all(check["pass"] for check in checks),
        "paper_claim": False,
        "gpu_claim": False,
        "saved_video_claim": False,
    }
    _write_canonical(output_dir / "metrics.json", metrics)
    return metrics


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_gate1a_admission(config_path: Path, gate1a_path: Path, lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    evidence = json.loads(gate1a_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if lock["gate_0_decision"] != "GATE_PASS" or lock["gate_1a_decision"] != "GATE_PASS":
        raise ProtocolError("Gate 0 and Gate 1A admission are required")
    if _sha256_file(gate1a_path) != lock["gate_1a_evidence_sha256"]:
        raise ProtocolError("Gate 1A evidence SHA mismatch")
    if _sha256_file(config_path) != lock["config_sha256"]:
        raise ProtocolError("v2 config SHA mismatch")
    channel_digest = canonical_digest(config["synthetic_protocol"]["channel"] )
    if channel_digest != lock["channel_sha256"]:
        raise ProtocolError("canonical channel SHA mismatch")
    if evidence["config_sha256"] != lock["config_sha256"] or evidence["channel_sha256"] != lock["channel_sha256"]:
        raise ProtocolError("Gate 1A embedded digest mismatch")
    tolerance = float(lock["matrix_comparison_absolute_tolerance"] )
    if len(evidence["cases"]) != 8:
        raise ProtocolError("Gate 1A must contain eight matrix records")
    for record in evidence["cases"]:
        index = int(record["case_index"] )
        matrix, bias, condition = _channel(config, index)
        for actual_row, frozen_row in zip(matrix, record["A"], strict=True):
            if any(abs(actual - frozen) > tolerance for actual, frozen in zip(actual_row, frozen_row, strict=True)):
                raise ProtocolError("Gate 1A matrix mismatch for case {}".format(index))
        if any(abs(actual - frozen) > tolerance for actual, frozen in zip(bias, record["b"], strict=True)):
            raise ProtocolError("Gate 1A bias mismatch for case {}".format(index))
        if abs(condition - float(record["condition_number"])) > tolerance:
            raise ProtocolError("Gate 1A condition mismatch for case {}".format(index))
    return {
        "lock_sha256": _sha256_file(lock_path),
        "gate1a_evidence_sha256": _sha256_file(gate1a_path),
        "config_sha256": _sha256_file(config_path),
        "channel_sha256": channel_digest,
        "matrix_record_count": len(evidence["cases"]),
    }


def run_gate(config_path: Path, gate1a_path: Path, lock_path: Path, output_dir: Path) -> dict[str, Any]:
    admission = _validate_gate1a_admission(config_path, gate1a_path, lock_path)
    metrics = _run_gate_core(config_path, output_dir)
    metrics["gate"] = "Gate 1B"
    metrics["gate1a_admission"] = admission
    _write_canonical(output_dir / "metrics.json", metrics)
    return metrics
