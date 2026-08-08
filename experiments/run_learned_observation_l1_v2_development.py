#!/usr/bin/env python3
"""Run the manifest-authorized, CPU-only Observation L1-v2 development gate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.learned_observation_l1_v2 import (  # noqa: E402
    InvalidExperiment,
    build_selected_prerequisite,
    canonical_json_bytes,
    evaluate_gate,
    invalid_audit,
    preflight,
    read_source_state,
    sha256_bytes,
    valid_audit,
)


def _write_package(
    output: Path,
    audit: dict[str, Any],
    command: list[str],
    *,
    manifest_bytes: bytes | None = None,
    config_bytes: bytes | None = None,
    prerequisite_artifacts: Any | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    audit_bytes = canonical_json_bytes(audit) + b"\n"
    if prerequisite_artifacts is not None and prerequisite_artifacts.audit_bytes != audit_bytes:
        raise InvalidExperiment("PREREQUISITE_EXPORT_INTEGRITY_FAILURE", "exported prerequisite does not bind the written G0 audit")
    (output / "audit.json").write_bytes(audit_bytes)
    (output / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    if manifest_bytes is not None:
        (output / "authorization_manifest.json").write_bytes(manifest_bytes)
    if config_bytes is not None:
        (output / "config.json").write_bytes(config_bytes)
    if prerequisite_artifacts is not None:
        (output / "frozen_frontend.json").write_bytes(prerequisite_artifacts.frontend_bytes)
        (output / "readout.json").write_bytes(prerequisite_artifacts.readout_bytes)
    lines = []
    for path in sorted(output.iterdir()):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _invalid_context(repo_root: Path, manifest: Path) -> dict[str, Any]:
    context: dict[str, Any] = {"manifest_path": str(manifest)}
    try:
        context["manifest_sha256"] = sha256_bytes(manifest.read_bytes())
    except OSError:
        context["manifest_readable"] = False
    try:
        context["source_state"] = read_source_state(repo_root).as_dict()
    except InvalidExperiment:
        context["source_state"] = {"git_readable": False}
    return context


def run(
    manifest: Path,
    output: Path,
    *,
    source_commit: str | None = None,
    repo_root: Path = ROOT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    invocation = command or [
        sys.executable,
        str(repo_root / "experiments" / "run_learned_observation_l1_v2_development.py"),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    ]
    if source_commit is not None and "--source-commit" not in invocation:
        invocation.extend(("--source-commit", source_commit))
    try:
        checked = preflight(repo_root, manifest, declared_source_commit=source_commit)
        candidates, selected = evaluate_gate(checked.features)
        audit = valid_audit(checked, candidates, selected)
        prerequisite_artifacts = build_selected_prerequisite(checked, audit, selected) if selected is not None else None
        _write_package(
            output,
            audit,
            invocation,
            manifest_bytes=checked.manifest_bytes,
            config_bytes=checked.config_bytes,
            prerequisite_artifacts=prerequisite_artifacts,
        )
    except InvalidExperiment as error:
        audit = invalid_audit(error, _invalid_context(repo_root, manifest))
        _write_package(output, audit, invocation)
    except Exception as error:  # Fail closed without exposing partial metrics.
        audit = invalid_audit(
            InvalidExperiment("UNEXPECTED_RUNTIME_FAILURE", type(error).__name__),
            {"failure_phase": "preflight_or_science"},
        )
        _write_package(output, audit, invocation)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    arguments = parser.parse_args()
    audit = run(
        arguments.manifest,
        arguments.output,
        source_commit=arguments.source_commit,
        command=sys.argv,
    )
    print(audit["status"])
    return 0 if audit["valid_experiment"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
