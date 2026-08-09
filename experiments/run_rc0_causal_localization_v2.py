#!/usr/bin/env python3
"""Fail-closed runner for the frozen RC0 causal-localization screen."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sc_sstw_feasibility.rc0_minimal_implementation import (  # noqa: E402
    PACKAGE_STATUS_INSUFFICIENT,
    PACKAGE_STATUS_INVALID,
    PACKAGE_STATUS_P_FAIL,
    PACKAGE_STATUS_P_PASS_R_FAIL,
    PACKAGE_STATUS_P_PASS_R_PASS,
    RUNNER_PATH,
    InsufficientEvidence,
    InvalidExperiment,
    Preflight,
    canonical_json_bytes,
    evaluate_execution,
    evaluation_audit,
    invalid_audit,
    preflight,
    sha256_bytes,
    sha256_file,
    validate_execution_record,
    validate_status_mapping,
    is_diagnostic_preflight,
)


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_FALLBACK_LIMIT = 1000
_TestFault = Callable[[str, Path | None, Path | None], None]


@dataclass(frozen=True, slots=True)
class RunOutcome:
    audit: dict[str, Any]
    actual_package_path: Path | None
    package_written: bool
    exit_code: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--diagnostic-bootstrap", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    # Parsed only so forbidden entry modes can be rejected before resolving or
    # reading their values.  No production path consumes these arguments.
    parser.add_argument("--execution-package", help=argparse.SUPPRESS)
    parser.add_argument("--synthetic-fixture", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    parser.add_argument("--device", help=argparse.SUPPRESS)
    parser.add_argument("--receipt", help=argparse.SUPPRESS)
    return parser


def _inject(fault: _TestFault | None, point: str, staging: Path | None = None, target: Path | None = None) -> None:
    if fault is not None:
        fault(point, staging, target)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _validate_output_parent(output: Path) -> None:
    if output.parent == output or not output.name:
        raise InvalidExperiment("OUTPUT_PARENT_INVALID", "output must name a child of an existing real directory")
    current = Path(output.anchor)
    for part in output.parent.parts[1:]:
        current /= part
        entry = _lstat(current)
        if entry is None or stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
            raise InvalidExperiment("OUTPUT_PARENT_INVALID", "output parent must be an existing symlink-free directory")


def _assert_output_unclaimed(output: Path) -> None:
    _validate_output_parent(output)
    if _lstat(output) is not None:
        raise InvalidExperiment("OUTPUT_TARGET_EXISTS", "output target is already owned")


def _create_staging(target: Path, kind: str, fault: _TestFault | None, point: str) -> Path:
    _inject(fault, point, None, target)
    staging = Path(tempfile.mkdtemp(prefix=f".rc0-{kind}-staging-{target.name}-", dir=target.parent))
    entry = staging.lstat()
    if staging.parent != target.parent or stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise OSError(errno.EPERM, "exclusive staging directory failed validation")
    return staging


def _safe_remove_staging(staging: Path | None, parent: Path) -> None:
    if staging is None or staging.parent != parent or not staging.name.startswith(".rc0-"):
        return
    try:
        entry = staging.lstat()
        if stat.S_ISDIR(entry.st_mode) and not stat.S_ISLNK(entry.st_mode):
            shutil.rmtree(staging)
    except OSError:
        pass


def _atomic_publish_no_replace(staging: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace publication is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(_AT_FDCWD, os.fsencode(staging), _AT_FDCWD, os.fsencode(target), _RENAME_NOREPLACE)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
        raise OSError(errno.ENOTSUP, "filesystem lacks atomic no-replace directory publication")
    raise OSError(error, os.strerror(error))


def _write_bytes(path: Path, payload: bytes, fault: _TestFault | None, point: str, staging: Path) -> None:
    _inject(fault, point, staging, None)
    path.write_bytes(payload)


def _package_regular_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        entry = path.lstat()
        if stat.S_ISLNK(entry.st_mode):
            raise OSError(errno.EINVAL, "package staging contains a symlink")
        if stat.S_ISREG(entry.st_mode):
            result.append(path)
        elif not stat.S_ISDIR(entry.st_mode):
            raise OSError(errno.EINVAL, "package staging contains an unsupported entry")
    return result


def _checksum_payload(directory: Path) -> bytes:
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(directory).as_posix()}"
        for path in _package_regular_files(directory)
        if path.name != "checksums.sha256"
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parse_checksums(raw: bytes) -> dict[str, str]:
    declared: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not name or name in declared:
            raise OSError(errno.EINVAL, "package checksum syntax is invalid")
        declared[name] = digest
    return declared


def _sanitized_command(command: list[str]) -> str:
    names = [item for item in command if item.startswith("--")]
    return shlex.join([RUNNER_PATH, *names]) + "\n"


def _self_check_complete_package(
    staging: Path,
    audit: Mapping[str, Any],
    identity_name: str,
    identity_digest_field: str,
) -> None:
    if audit.get("status") not in {
        PACKAGE_STATUS_P_FAIL,
        PACKAGE_STATUS_P_PASS_R_FAIL,
        PACKAGE_STATUS_P_PASS_R_PASS,
        PACKAGE_STATUS_INSUFFICIENT,
    }:
        raise OSError(errno.EINVAL, "complete package contains an invalid terminal status")
    validate_status_mapping(audit.get("status"), audit.get("protocol_outcome"))
    if audit.get("formal_result") is not False or audit.get("stage_progression_allowed") is not False:
        raise OSError(errno.EINVAL, "complete package evidence boundary changed")
    required = {
        "audit.json",
        identity_name,
        "command.txt",
        "config.json",
        "plan.json",
        "protocol.md",
        "checksums.sha256",
        "generation/execution.json",
    }
    actual = {path.relative_to(staging).as_posix() for path in _package_regular_files(staging)}
    if not required.issubset(actual):
        raise OSError(errno.EINVAL, "complete package file set is incomplete")
    if json.loads((staging / "audit.json").read_bytes()) != audit:
        raise OSError(errno.EINVAL, "audit round-trip changed")
    declared = _parse_checksums((staging / "checksums.sha256").read_bytes())
    if set(declared) != actual - {"checksums.sha256"}:
        raise OSError(errno.EINVAL, "checksum set is incomplete")
    for relative_path, digest in declared.items():
        if sha256_file(staging / relative_path) != digest:
            raise OSError(errno.EINVAL, "package checksum self-check failed")
    if declared[identity_name] != audit[identity_digest_field]:
        raise OSError(errno.EINVAL, "run identity differs from audit")


def _fallback_candidates(output: Path, prefer_requested: bool) -> list[Path]:
    candidates = [output] if prefer_requested else []
    candidates.extend(output.with_name(f"{output.name}.invalid.{index:04d}") for index in range(1, _FALLBACK_LIMIT + 1))
    return candidates


def _minimal_invalid_audit(
    reason_code: str,
    failure_phase: str,
    exception_type: str,
    actual_package_path: Path | None,
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    audit = invalid_audit(reason_code, failure_phase, exception_type, context)
    audit.update(
        {
            "package_written": actual_package_path is not None,
            "actual_package_path": str(actual_package_path) if actual_package_path is not None else None,
        }
    )
    return audit


def _write_invalid_best_effort(
    output: Path,
    command: list[str],
    *,
    reason_code: str,
    failure_phase: str,
    exception_type: str,
    context: Mapping[str, Any] | None,
    prefer_requested: bool,
    fault: _TestFault | None,
) -> RunOutcome:
    staging: Path | None = None
    try:
        _validate_output_parent(output)
        for target in _fallback_candidates(output, prefer_requested):
            if _lstat(target) is not None:
                continue
            try:
                staging = _create_staging(target, "invalid", fault, "fallback_staging_mkdir")
                audit = _minimal_invalid_audit(reason_code, failure_phase, exception_type, target, context)
                _write_bytes(staging / "audit.json", canonical_json_bytes(audit) + b"\n", fault, "fallback_audit_write", staging)
                _write_bytes(staging / "command.txt", _sanitized_command(command).encode("utf-8"), fault, "fallback_command_write", staging)
                _inject(fault, "fallback_checksum_write", staging, target)
                (staging / "checksums.sha256").write_bytes(_checksum_payload(staging))
                _inject(fault, "fallback_publish_race", staging, target)
                _inject(fault, "fallback_publish", staging, target)
                _atomic_publish_no_replace(staging, target)
                staging = None
                return RunOutcome(audit=audit, actual_package_path=target, package_written=True, exit_code=2)
            except FileExistsError:
                _safe_remove_staging(staging, output.parent)
                staging = None
                continue
        raise OSError(errno.EEXIST, "no exclusive invalid-package target is available")
    except (OSError, RuntimeError, TypeError, ValueError) as fallback_error:
        _safe_remove_staging(staging, output.parent)
        audit = _minimal_invalid_audit(
            "INVALID_PACKAGE_WRITE_FAILED",
            "invalid_package_write",
            type(fallback_error).__name__,
            None,
            None,
        )
        return RunOutcome(audit=audit, actual_package_path=None, package_written=False, exit_code=2)


def _unexpected_reason(error: Exception) -> str:
    if isinstance(error, MemoryError) or (isinstance(error, RuntimeError) and "out of memory" in str(error).lower()):
        return "UNEXPECTED_MEMORY_FAILURE"
    if isinstance(error, subprocess.SubprocessError):
        return "UNEXPECTED_SUBPROCESS_FAILURE"
    if isinstance(error, TypeError):
        return "UNEXPECTED_TYPE_FAILURE"
    if isinstance(error, OSError):
        return "UNEXPECTED_IO_FAILURE"
    return "UNEXPECTED_RUNTIME_FAILURE"


def _context(
    preflight_result: Preflight | None,
    *,
    cpu_only_test_harness: bool,
    diagnostic_only: bool = False,
) -> dict[str, Any] | None:
    if preflight_result is None:
        return {"diagnostic_only": True} if diagnostic_only else None
    context = {
        "manifest_sha256": sha256_bytes(preflight_result.manifest_bytes),
        "source_head": preflight_result.source_state.head,
        "source_tree": preflight_result.source_state.tree,
        "config_sha256": sha256_bytes(preflight_result.config_bytes),
        "plan_sha256": sha256_bytes(preflight_result.plan_bytes),
        "cpu_only_test_harness": cpu_only_test_harness,
        "test_only_non_evidence": cpu_only_test_harness,
    }
    if diagnostic_only:
        context.update(
            {
                "diagnostic_only": True,
                "diagnostic_class": "DIAGNOSTIC_ONLY",
                "authorization_claimed": False,
                "gate_claimed": False,
            }
        )
    return context


def run(
    arguments: argparse.Namespace,
    command: list[str],
    *,
    repo_root: Path = REPO_ROOT,
    _test_backend: Any | None = None,
    _test_fault: _TestFault | None = None,
) -> RunOutcome:
    output = _lexical_absolute(arguments.output)
    # Step zero: reject all external or caller-selected production topology
    # before resolving or reading any execution-package value or manifest.
    if arguments.execution_package is not None:
        return _write_invalid_best_effort(
            output,
            command,
            reason_code="EXTERNAL_PRODUCTION_PACKAGE_FORBIDDEN",
            failure_phase="cli_mode",
            exception_type="InvalidExperiment",
            context=None,
            prefer_requested=_lstat(output) is None,
            fault=_test_fault,
        )
    if arguments.synthetic_fixture or arguments.backend is not None or arguments.device is not None or arguments.receipt is not None or not arguments.generate:
        return _write_invalid_best_effort(
            output,
            command,
            reason_code="CLI_MODE_FORBIDDEN",
            failure_phase="cli_mode",
            exception_type="InvalidExperiment",
            context=None,
            prefer_requested=_lstat(output) is None,
            fault=_test_fault,
        )

    diagnostic_only = arguments.diagnostic_bootstrap is not None
    if (arguments.manifest is None) == (arguments.diagnostic_bootstrap is None):
        return _write_invalid_best_effort(
            output,
            command,
            reason_code="RUN_IDENTITY_MODE_INVALID",
            failure_phase="cli_mode",
            exception_type="InvalidExperiment",
            context={"diagnostic_only": True} if diagnostic_only else None,
            prefer_requested=_lstat(output) is None,
            fault=_test_fault,
        )
    identity_path = arguments.diagnostic_bootstrap if diagnostic_only else arguments.manifest

    frozen: Preflight | None = None
    phase = "manifest_and_source_preflight"
    try:
        frozen = preflight(
            repo_root,
            identity_path,
            declared_source_commit=arguments.source_commit,
            diagnostic_only=diagnostic_only,
        )
        phase = "output_ownership"
        _assert_output_unclaimed(output)
    except InvalidExperiment as error:
        return _write_invalid_best_effort(
            output,
            command,
            reason_code=error.reason_code,
            failure_phase=phase,
            exception_type=type(error).__name__,
            context=_context(frozen, cpu_only_test_harness=_test_backend is not None, diagnostic_only=diagnostic_only),
            prefer_requested=_lstat(output) is None,
            fault=_test_fault,
        )
    except Exception as error:
        return _write_invalid_best_effort(
            output,
            command,
            reason_code=_unexpected_reason(error),
            failure_phase=phase,
            exception_type=type(error).__name__,
            context=_context(frozen, cpu_only_test_harness=_test_backend is not None, diagnostic_only=diagnostic_only),
            prefer_requested=_lstat(output) is None,
            fault=_test_fault,
        )

    staging: Path | None = None
    try:
        staging = _create_staging(output, "package", _test_fault, "main_staging_mkdir")
        phase = "generation"
        from sc_sstw_feasibility.rc0_generation import run_rc0_generation

        outcome = run_rc0_generation(frozen, staging / "generation", command, _test_backend=_test_backend)
        phase = "execution_validation"
        record, artifacts, evidence_mode = validate_execution_record(
            outcome.record_path,
            frozen,
            outcome.receipt,
            allow_cpu_test_harness=outcome.cpu_only_test_harness,
        )
        phase = "saved_mp4_level_p_then_r"
        evaluation = evaluate_execution(record, artifacts, frozen, evidence_mode)
        audit = evaluation_audit(
            frozen,
            record,
            sha256_file(outcome.record_path),
            evaluation,
            cpu_only_test_harness=outcome.cpu_only_test_harness,
        )
        diagnostic_package = is_diagnostic_preflight(frozen)
        identity_name = "diagnostic_bootstrap.json" if diagnostic_package else "authorization_manifest.json"
        identity_digest_field = "diagnostic_bootstrap_sha256" if diagnostic_package else "manifest_sha256"
        phase = "package_write"
        _write_bytes(staging / "audit.json", canonical_json_bytes(audit) + b"\n", _test_fault, "main_audit_write", staging)
        _write_bytes(staging / "command.txt", _sanitized_command(command).encode("utf-8"), _test_fault, "main_command_write", staging)
        _write_bytes(staging / identity_name, frozen.manifest_bytes, _test_fault, "main_manifest_write", staging)
        _write_bytes(staging / "config.json", frozen.config_bytes, _test_fault, "main_config_write", staging)
        _write_bytes(staging / "plan.json", frozen.plan_bytes, _test_fault, "main_plan_write", staging)
        _write_bytes(staging / "protocol.md", frozen.protocol_bytes, _test_fault, "main_protocol_write", staging)
        _inject(_test_fault, "main_checksum_write", staging, output)
        (staging / "checksums.sha256").write_bytes(_checksum_payload(staging))
        _inject(_test_fault, "main_checksum_self_check", staging, output)
        _self_check_complete_package(staging, audit, identity_name, identity_digest_field)
        _inject(_test_fault, "main_publish_race", staging, output)
        _inject(_test_fault, "main_publish", staging, output)
        _atomic_publish_no_replace(staging, output)
        staging = None
        exit_codes = {
            PACKAGE_STATUS_P_PASS_R_PASS: 0,
            PACKAGE_STATUS_P_FAIL: 3,
            PACKAGE_STATUS_P_PASS_R_FAIL: 3,
            PACKAGE_STATUS_INSUFFICIENT: 4,
        }
        return RunOutcome(audit=audit, actual_package_path=output, package_written=True, exit_code=exit_codes[audit["status"]])
    except InsufficientEvidence as error:
        reason_code = error.reason_code
        exception_type = type(error).__name__
    except InvalidExperiment as error:
        reason_code = error.reason_code
        exception_type = type(error).__name__
    except Exception as error:
        reason_code = _unexpected_reason(error)
        exception_type = type(error).__name__
    finally:
        _safe_remove_staging(staging, output.parent)
    return _write_invalid_best_effort(
        output,
        command,
        reason_code=reason_code,
        failure_phase=phase,
        exception_type=exception_type,
        context=_context(frozen, cpu_only_test_harness=_test_backend is not None, diagnostic_only=diagnostic_only),
        prefer_requested=False,
        fault=_test_fault,
    )


def _response(outcome: RunOutcome) -> dict[str, Any]:
    return {
        "status": outcome.audit["status"],
        "protocol_outcome": outcome.audit["protocol_outcome"],
        "reason_code": outcome.audit["reason_code"],
        "package_written": outcome.package_written,
        "actual_package_path": str(outcome.actual_package_path) if outcome.actual_package_path is not None else None,
    }


def main(
    argv: list[str] | None = None,
    *,
    _repo_root: Path = REPO_ROOT,
    _test_backend: Any | None = None,
    _test_fault: _TestFault | None = None,
) -> int:
    arguments = _parser().parse_args(argv)
    command = list(sys.argv if argv is None else [str(_repo_root / RUNNER_PATH), *argv])
    outcome = run(
        arguments,
        command,
        repo_root=_repo_root,
        _test_backend=_test_backend,
        _test_fault=_test_fault,
    )
    stream = sys.stdout if outcome.package_written else sys.stderr
    print(canonical_json_bytes(_response(outcome)).decode("utf-8"), file=stream)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
