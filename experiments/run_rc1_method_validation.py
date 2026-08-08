#!/usr/bin/env python3
"""Fail-closed CLI for the frozen RC1 matched-triplet screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sc_sstw_feasibility.rc1_method_validation import (  # noqa: E402
    InvalidExperiment,
    PrerequisiteNotMet,
    STATUS_FAIL,
    STATUS_PASS,
    canonical_json_bytes,
    evaluate_execution,
    invalid_audit,
    preflight,
    prerequisite_audit,
    sha256_bytes,
    sha256_file,
    valid_audit,
    validate_execution_record,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--execution-package", type=Path)
    source.add_argument("--generate", action="store_true")
    parser.add_argument("--synthetic-fixture", action="store_true", help="admit only explicitly labeled pure synthetic feature artifacts")
    return parser


def _minimal_context(manifest_path: Path) -> dict[str, Any]:
    context: dict[str, Any] = {"manifest_path": str(manifest_path)}
    try:
        context["manifest_sha256"] = sha256_file(manifest_path)
    except OSError:
        context["manifest_sha256"] = None
    return context


def _write_package(output: Path, audit: dict[str, Any], command: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "audit.json"
    command_path = output / "command.json"
    if audit_path.exists() or command_path.exists() or (output / "checksums.sha256").exists():
        raise RuntimeError("audit package output files already exist")
    audit_path.write_bytes(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    command_path.write_bytes(canonical_json_bytes(command) + b"\n")
    checksums = f"{sha256_file(audit_path)}  audit.json\n{sha256_file(command_path)}  command.json\n"
    (output / "checksums.sha256").write_text(checksums, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command = [str(Path(sys.argv[0]).name), *(argv if argv is not None else sys.argv[1:])]
    context = _minimal_context(arguments.manifest)
    audit: dict[str, Any]
    exit_code = 2
    try:
        frozen = preflight(REPO_ROOT, arguments.manifest.resolve(), declared_source_commit=arguments.source_commit)
        if arguments.generate and arguments.synthetic_fixture:
            raise InvalidExperiment("EXECUTION_MODE_MISMATCH", "GPU generation cannot be labeled as a synthetic fixture")
        if arguments.generate:
            from src.sc_sstw_feasibility.rc1_gpu_generation import RC1GPUGenerationError, run_gpu_generation

            try:
                execution_path = run_gpu_generation(frozen, arguments.output / "generation", command)
            except RC1GPUGenerationError as exc:
                raise InvalidExperiment("GENERATION_INTEGRITY_FAILURE", str(exc)) from exc
        elif arguments.execution_package is not None:
            execution_path = arguments.execution_package.resolve()
        else:
            raise PrerequisiteNotMet("EXECUTION_PACKAGE_NOT_PROVIDED", "preflight passed but no fresh matched-triplet execution was supplied")
        record, artifacts = validate_execution_record(execution_path, frozen, synthetic_fixture=arguments.synthetic_fixture)
        group_results, passed = evaluate_execution(record, execution_path, artifacts, frozen, synthetic_fixture=arguments.synthetic_fixture)
        audit = valid_audit(frozen, record, sha256_file(execution_path), group_results, passed)
        exit_code = 0 if audit["status"] == STATUS_PASS else 3
    except PrerequisiteNotMet as exc:
        audit = prerequisite_audit(exc, context)
    except InvalidExperiment as exc:
        audit = invalid_audit(exc, context)
    try:
        _write_package(arguments.output, audit, command)
    except (OSError, RuntimeError) as exc:
        print(json.dumps({"status": "INVALID_EXPERIMENT", "reason_code": "AUDIT_PACKAGE_WRITE_FAILURE", "detail": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": audit["status"], "reason_code": audit["reason_code"], "audit_sha256": sha256_bytes((arguments.output / "audit.json").read_bytes())}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
