#!/usr/bin/env python3
"""Fail-closed CLI for the frozen RC1 matched-triplet screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
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
    RUNNER_PATH,
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


def _sanitized_command(arguments: list[str]) -> dict[str, Any]:
    return {"runner": RUNNER_PATH, "argument_names": [item for item in arguments if item.startswith("--")]}


def _write_package(output: Path, audit: dict[str, Any], command: list[str]) -> None:
    if not output.is_dir() or output.is_symlink():
        raise RuntimeError("audit package root is not an owned real directory")
    audit_path = output / "audit.json"
    command_path = output / "command.json"
    if audit_path.exists() or command_path.exists() or (output / "checksums.sha256").exists():
        raise RuntimeError("audit package output files already exist")
    audit_bytes = json.dumps(audit, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    command_bytes = canonical_json_bytes(_sanitized_command(command)) + b"\n"
    checksums = f"{sha256_bytes(audit_bytes)}  audit.json\n{sha256_bytes(command_bytes)}  command.json\n".encode("utf-8")
    temporary = {"audit": output / ".audit.json.tmp", "command": output / ".command.json.tmp", "checksums": output / ".checksums.sha256.tmp"}
    for path in temporary.values():
        if path.exists():
            raise RuntimeError("temporary audit package path already exists")
    try:
        temporary["audit"].write_bytes(audit_bytes)
        temporary["command"].write_bytes(command_bytes)
        temporary["checksums"].write_bytes(checksums)
        temporary["command"].replace(command_path)
        temporary["checksums"].replace(output / "checksums.sha256")
        temporary["audit"].replace(audit_path)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def _fallback_root(requested: Path, reason: str) -> Path:
    token = hashlib.sha256(f"{requested.resolve()}:{reason}".encode("utf-8")).hexdigest()[:16]
    return requested.parent / f"{requested.name}.invalid-{token}"


def _claim_output(requested: Path) -> tuple[Path, InvalidExperiment | None]:
    try:
        requested.mkdir(parents=True, exist_ok=False)
        return requested, None
    except Exception as exc:
        fallback = _fallback_root(requested, "output_claim")
        try:
            fallback.mkdir(parents=False, exist_ok=False)
        except Exception as fallback_exc:
            raise RuntimeError("both requested output and deterministic fallback are unavailable") from fallback_exc
        return fallback, InvalidExperiment("OUTPUT_DIRECTORY_UNAVAILABLE", type(exc).__name__)


def _unexpected_error(error: Exception) -> InvalidExperiment:
    if isinstance(error, MemoryError) or (isinstance(error, RuntimeError) and "out of memory" in str(error).lower()):
        return InvalidExperiment("UNEXPECTED_MEMORY_FAILURE", "ordinary memory failure")
    if isinstance(error, subprocess.SubprocessError):
        return InvalidExperiment("UNEXPECTED_SUBPROCESS_FAILURE", "ordinary subprocess failure")
    if isinstance(error, TypeError):
        return InvalidExperiment("UNEXPECTED_TYPE_FAILURE", "ordinary type failure")
    return InvalidExperiment("UNEXPECTED_RUNTIME_FAILURE", "ordinary runtime failure")


def main(argv: list[str] | None = None, *, _test_generation_backend: Any | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command = [str(Path(sys.argv[0]).name), *(argv if argv is not None else sys.argv[1:])]
    context: dict[str, Any] = {}
    audit: dict[str, Any]
    exit_code = 2
    phase = "output_claim"
    try:
        package_root, claim_error = _claim_output(arguments.output)
    except Exception as exc:
        print(json.dumps({"status": "INVALID_EXPERIMENT", "reason_code": "AUDIT_PACKAGE_WRITE_FAILURE", "failure_phase": "output_claim", "exception_type": type(exc).__name__}, sort_keys=True), file=sys.stderr)
        return 2
    try:
        if claim_error is not None:
            raise claim_error
        phase = "prerequisite_preflight"
        frozen = preflight(REPO_ROOT, arguments.manifest.resolve(), declared_source_commit=arguments.source_commit)
        context = {
            "manifest_sha256": sha256_bytes(frozen.manifest_bytes),
            "source_head": frozen.source_state.head,
            "source_tree": frozen.source_state.tree,
            "config_sha256": sha256_bytes(frozen.config_bytes),
            "prerequisite_checksums_sha256": frozen.prerequisite.checksums_sha256,
        }
        if arguments.synthetic_fixture and frozen.manifest["evidence_policy"]["synthetic_fixture_permitted"] is not True:
            raise InvalidExperiment("EXECUTION_MODE_MISMATCH", "authorization manifest does not permit test-only synthetic execution")
        if arguments.generate and arguments.synthetic_fixture:
            raise InvalidExperiment("EXECUTION_MODE_MISMATCH", "GPU generation cannot be labeled as a synthetic fixture")
        if arguments.generate:
            phase = "generation"
            from src.sc_sstw_feasibility.rc1_gpu_generation import RC1GPUGenerationError, run_gpu_generation

            try:
                outcome = run_gpu_generation(frozen, package_root / "generation", command, _test_backend=_test_generation_backend)
                execution_path = outcome.record_path
                generation_receipt = outcome.receipt
                cpu_only_test_harness = outcome.cpu_only_test_harness
            except RC1GPUGenerationError as exc:
                raise InvalidExperiment("GENERATION_INTEGRITY_FAILURE", "GPU generation helper failed integrity") from exc
        elif arguments.execution_package is not None:
            phase = "execution_entry"
            if not arguments.synthetic_fixture:
                raise InvalidExperiment("EXTERNAL_PRODUCTION_PACKAGE_FORBIDDEN", "production execution must originate in this runner process via --generate")
            execution_path = arguments.execution_package.resolve()
            generation_receipt = None
            cpu_only_test_harness = False
        else:
            raise PrerequisiteNotMet("EXECUTION_PACKAGE_NOT_PROVIDED", "preflight passed but no fresh matched-triplet execution was supplied")
        phase = "execution_package_validation"
        record, artifacts = validate_execution_record(
            execution_path,
            frozen,
            synthetic_fixture=arguments.synthetic_fixture,
            generation_receipt=generation_receipt,
            allow_cpu_test_harness=cpu_only_test_harness,
        )
        phase = "saved_mp4_decode_and_evaluation"
        group_results, passed = evaluate_execution(record, execution_path, artifacts, frozen, synthetic_fixture=arguments.synthetic_fixture)
        audit = valid_audit(
            frozen,
            record,
            sha256_file(execution_path),
            group_results,
            passed,
            cpu_only_test_harness=cpu_only_test_harness,
        )
        exit_code = 0 if audit["status"] == STATUS_PASS else 3
    except PrerequisiteNotMet as exc:
        audit = prerequisite_audit(exc, {**context, "failure_phase": phase, "exception_type": type(exc).__name__})
    except InvalidExperiment as exc:
        audit = invalid_audit(exc, {**context, "failure_phase": phase, "exception_type": type(exc).__name__})
    except Exception as exc:
        error = _unexpected_error(exc)
        audit = invalid_audit(error, {**context, "failure_phase": phase, "exception_type": type(exc).__name__})
        exit_code = 2
    phase = "audit_package_write"
    try:
        _write_package(package_root, audit, command)
    except Exception as exc:
        fallback = _fallback_root(arguments.output, "package_write")
        try:
            fallback.mkdir(parents=False, exist_ok=False)
            audit = invalid_audit(_unexpected_error(exc), {**context, "failure_phase": phase, "exception_type": type(exc).__name__})
            audit["reason_code"] = "AUDIT_PACKAGE_WRITE_FAILURE"
            _write_package(fallback, audit, command)
            package_root = fallback
            exit_code = 2
        except Exception as fallback_exc:
            print(json.dumps({"status": "INVALID_EXPERIMENT", "reason_code": "AUDIT_PACKAGE_WRITE_FAILURE", "failure_phase": phase, "exception_type": type(fallback_exc).__name__}, sort_keys=True), file=sys.stderr)
            return 2
    print(json.dumps({"status": audit["status"], "reason_code": audit["reason_code"], "audit_path": str(package_root / "audit.json"), "audit_sha256": sha256_bytes((package_root / "audit.json").read_bytes())}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
