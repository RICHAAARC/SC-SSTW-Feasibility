"""Single source of truth for the G0-to-RC1 prerequisite artifacts.

The producer receives only an internally computed G0 audit and final readout.
The RC1 consumer validates the same schemas and bindings from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


G0_SUCCESS_STATUS = "READY_TO_PREREGISTER_FRESH_GPU_GATE"
G0_PROTOCOL_ID = "sc_sstw_learned_observation_l1_v2_development"
G0_OUTPUT_SCHEMA = "sc_sstw_l1_v2_development_audit_v1"
FRONTEND_SCHEMA = "sc_sstw_rc1_frozen_frontend_v2"
READOUT_SCHEMA = "sc_sstw_rc1_frozen_readout_v2"
ALLOWED_CANDIDATES = ("A1", "A2")
FIT_DATASET_IDS = (41001, 41002, 41003, 41004)
ALL_INPUT_IDS = FIT_DATASET_IDS + (41005, 41006)
PREREQUISITE_FILES = ("audit.json", "frozen_frontend.json", "readout.json")
G0_SUCCESS_PACKAGE_FILES = (
    "audit.json",
    "authorization_manifest.json",
    "command.txt",
    "config.json",
    "frozen_frontend.json",
    "readout.json",
)


@dataclass(frozen=True)
class PrerequisiteArtifacts:
    audit_bytes: bytes
    frontend_bytes: bytes
    readout_bytes: bytes


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def frontend_definition(candidate: str) -> dict[str, Any]:
    if candidate not in ALLOWED_CANDIDATES:
        raise ValueError("selected candidate must be A1 or A2")
    return {
        "candidate": candidate,
        "normalization": {"mad_scale": 1.4826, "mad_floor": 1e-6, "clip_min": -6.0, "clip_max": 6.0},
        "high_pass": candidate == "A2",
        "readout": {"kind": "linear_affine", "shape": [31, 2], "fit_intercept": True},
    }


def build_prerequisite_artifacts(audit: Mapping[str, Any], readout: np.ndarray) -> PrerequisiteArtifacts:
    """Build artifacts only from a genuine internally computed G0 success."""

    selected = audit.get("selected_candidate")
    if (
        audit.get("protocol_id") != G0_PROTOCOL_ID
        or audit.get("output_schema") != G0_OUTPUT_SCHEMA
        or audit.get("status") != G0_SUCCESS_STATUS
        or audit.get("valid_experiment") is not True
        or audit.get("formal_result") is not False
        or audit.get("stage_progression_allowed") is not False
        or selected not in ALLOWED_CANDIDATES
    ):
        raise ValueError("only a bounded valid G0 success can produce RC1 prerequisite artifacts")
    source = audit.get("source_state")
    source_hashes = audit.get("source_file_sha256")
    input_hashes = audit.get("input_sha256")
    config_identity = audit.get("config_identity")
    manifest_identity = audit.get("manifest_identity")
    if (
        not isinstance(source, Mapping)
        or source.get("dirty") is not False
        or not is_commit(source.get("head"))
        or not is_commit(source.get("tree"))
        or not isinstance(source_hashes, Mapping)
        or not source_hashes
        or not all(is_sha256(value) for value in source_hashes.values())
        or not isinstance(input_hashes, Mapping)
        or tuple(sorted(input_hashes)) != tuple(str(value) for value in ALL_INPUT_IDS)
        or not all(is_sha256(value) for value in input_hashes.values())
        or not isinstance(config_identity, Mapping)
        or not is_sha256(config_identity.get("sha256"))
        or not isinstance(manifest_identity, Mapping)
        or not is_sha256(manifest_identity.get("sha256"))
    ):
        raise ValueError("G0 success identity is incomplete")
    traces = [item for item in audit.get("candidates", ()) if isinstance(item, Mapping) and item.get("candidate") == selected]
    if len(traces) != 1 or traces[0].get("development_gate_pass") is not True:
        raise ValueError("G0 selected-candidate trace is contradictory")
    thresholds = traces[0].get("derived_envelope")
    if not isinstance(thresholds, Mapping) or not all(math.isfinite(float(value)) for value in thresholds.values()):
        raise ValueError("G0 selected-candidate thresholds are incomplete")
    array = np.asarray(readout, dtype=np.float64)
    if array.shape != (31, 2) or not np.isfinite(array).all():
        raise ValueError("final G0 readout must be finite 31x2")

    audit_bytes = canonical_json_bytes(dict(audit)) + b"\n"
    audit_sha = sha256_bytes(audit_bytes)
    readout_payload = {
        "schema_version": 2,
        "readout_schema": READOUT_SCHEMA,
        "selected_candidate": selected,
        "shape": [31, 2],
        "fit_dataset_ids": list(FIT_DATASET_IDS),
        "input_sha256": {str(value): input_hashes[str(value)] for value in FIT_DATASET_IDS},
        "source_head": source["head"],
        "source_tree": source["tree"],
        "config_sha256": config_identity["sha256"],
        "coefficients": array.tolist(),
    }
    readout_bytes = canonical_json_bytes(readout_payload) + b"\n"
    readout_sha = sha256_bytes(readout_bytes)
    definition_sha = sha256_bytes(canonical_json_bytes(frontend_definition(str(selected))))
    frontend_payload = {
        "schema_version": 2,
        "frontend_schema": FRONTEND_SCHEMA,
        "selected_candidate": selected,
        "frontend_definition_sha256": definition_sha,
        "readout_path": "readout.json",
        "readout_sha256": readout_sha,
        "thresholds": dict(thresholds),
        "g0_identity": {
            "protocol_id": G0_PROTOCOL_ID,
            "output_schema": G0_OUTPUT_SCHEMA,
            "audit_sha256": audit_sha,
            "manifest_sha256": manifest_identity["sha256"],
            "config_identity": dict(config_identity),
            "source_state": {"head": source["head"], "tree": source["tree"], "dirty": False},
            "source_file_sha256": dict(source_hashes),
            "input_sha256": dict(input_hashes),
        },
    }
    return PrerequisiteArtifacts(
        audit_bytes=audit_bytes,
        frontend_bytes=canonical_json_bytes(frontend_payload) + b"\n",
        readout_bytes=readout_bytes,
    )
