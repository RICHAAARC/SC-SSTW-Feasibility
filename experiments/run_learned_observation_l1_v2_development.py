#!/usr/bin/env python3
"""Run the manifest-authorized, CPU-only Observation L1-v2 development gate."""

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
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_sstw_feasibility.learned_observation_l1_v2 import (  # noqa: E402
    FAILURE_STATUS,
    INVALID_STATUS,
    OUTPUT_SCHEMA,
    PROTOCOL_ID,
    SUCCESS_STATUS,
    InvalidExperiment,
    build_selected_prerequisite,
    canonical_json_bytes,
    evaluate_gate,
    preflight,
    read_source_state,
    sha256_bytes,
    valid_audit,
)


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_FALLBACK_LIMIT = 1000
_TestFault = Callable[[str, Path | None, Path | None], None]


@dataclass(frozen=True)
class _RunOutcome:
    audit: dict[str, Any]
    actual_package_path: Path | None
    package_written: bool


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
    staging = Path(tempfile.mkdtemp(prefix=f".g0-{kind}-staging-{target.name}-", dir=target.parent))
    entry = staging.lstat()
    if staging.parent != target.parent or stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
        raise OSError(errno.EPERM, "exclusive staging directory failed validation")
    return staging


def _safe_remove_staging(staging: Path | None, parent: Path) -> None:
    if staging is None or staging.parent != parent or not staging.name.startswith(".g0-"):
        return
    try:
        entry = staging.lstat()
        if stat.S_ISDIR(entry.st_mode) and not stat.S_ISLNK(entry.st_mode):
            shutil.rmtree(staging)
    except OSError:
        # A retained hidden staging directory is rejected by the RC1 consumer.
        pass


def _atomic_publish_no_replace(staging: Path, target: Path) -> None:
    """Atomically publish a directory without replacing any target.

    There is deliberately no check-then-rename fallback: filesystems without
    renameat2(RENAME_NOREPLACE) fail closed.
    """

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


def _checksum_payload(directory: Path) -> bytes:
    lines: list[str] = []
    for path in sorted(directory.iterdir()):
        entry = path.lstat()
        if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode):
            raise OSError(errno.EINVAL, "package staging contains a non-regular entry")
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parse_checksums(raw: bytes) -> dict[str, str]:
    declared: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not name or name in declared:
            raise OSError(errno.EINVAL, "package checksum syntax is invalid")
        declared[name] = digest
    return declared


def _self_check_complete_package(staging: Path, audit: dict[str, Any], prerequisite_artifacts: Any | None) -> None:
    if audit.get("valid_experiment") is not True or audit.get("status") not in {SUCCESS_STATUS, FAILURE_STATUS}:
        raise OSError(errno.EINVAL, "complete writer accepts only valid success or scientific-failure audits")
    expected = {"audit.json", "authorization_manifest.json", "command.txt", "config.json", "checksums.sha256"}
    if prerequisite_artifacts is not None:
        expected.update({"frozen_frontend.json", "readout.json"})
    actual = {path.name for path in staging.iterdir()}
    if actual != expected:
        raise OSError(errno.EINVAL, "package file set is incomplete")
    audit_bytes = (staging / "audit.json").read_bytes()
    if json.loads(audit_bytes) != audit:
        raise OSError(errno.EINVAL, "package audit round-trip changed")
    declared = _parse_checksums((staging / "checksums.sha256").read_bytes())
    if set(declared) != expected - {"checksums.sha256"}:
        raise OSError(errno.EINVAL, "package checksum set is incomplete")
    for name, digest in declared.items():
        if hashlib.sha256((staging / name).read_bytes()).hexdigest() != digest:
            raise OSError(errno.EINVAL, "package checksum self-check failed")
    if (
        declared["authorization_manifest.json"] != audit.get("manifest_identity", {}).get("sha256")
        or declared["config.json"] != audit.get("config_identity", {}).get("sha256")
    ):
        raise OSError(errno.EINVAL, "package manifest or config identity self-check failed")
    if audit.get("status") == SUCCESS_STATUS:
        if prerequisite_artifacts is None or audit.get("selected_candidate") not in {"A1", "A2"}:
            raise OSError(errno.EINVAL, "successful package lacks its frozen prerequisite")
        frontend = json.loads((staging / "frozen_frontend.json").read_bytes())
        readout = json.loads((staging / "readout.json").read_bytes())
        audit_sha = hashlib.sha256(audit_bytes).hexdigest()
        readout_sha = declared["readout.json"]
        if (
            frontend.get("selected_candidate") != audit["selected_candidate"]
            or readout.get("selected_candidate") != audit["selected_candidate"]
            or frontend.get("readout_sha256") != readout_sha
            or frontend.get("g0_identity", {}).get("audit_sha256") != audit_sha
            or frontend.get("g0_identity", {}).get("manifest_sha256") != audit["manifest_identity"]["sha256"]
            or readout.get("source_head") != audit["source_state"]["head"]
            or readout.get("source_tree") != audit["source_state"]["tree"]
            or readout.get("config_sha256") != audit["config_identity"]["sha256"]
        ):
            raise OSError(errno.EINVAL, "successful prerequisite identity self-check failed")
    else:
        if audit.get("selected_candidate") is not None:
            raise OSError(errno.EINVAL, "scientific-failure package names a selected candidate")
        if prerequisite_artifacts is not None or {"frozen_frontend.json", "readout.json"}.intersection(actual):
            raise OSError(errno.EINVAL, "non-success package contains prerequisite artifacts")


def _write_complete_package(
    output: Path,
    audit: dict[str, Any],
    command: list[str],
    *,
    manifest_bytes: bytes,
    config_bytes: bytes,
    prerequisite_artifacts: Any | None,
    fault: _TestFault | None,
) -> None:
    staging: Path | None = None
    try:
        staging = _create_staging(output, "package", fault, "main_staging_mkdir")
        audit_bytes = canonical_json_bytes(audit) + b"\n"
        if prerequisite_artifacts is not None and prerequisite_artifacts.audit_bytes != audit_bytes:
            raise InvalidExperiment("PREREQUISITE_EXPORT_INTEGRITY_FAILURE", "exported prerequisite does not bind the written G0 audit")
        _write_bytes(staging / "audit.json", audit_bytes, fault, "main_audit_write", staging)
        _write_bytes(staging / "command.txt", (shlex.join(command) + "\n").encode("utf-8"), fault, "main_command_write", staging)
        _write_bytes(staging / "authorization_manifest.json", manifest_bytes, fault, "main_manifest_write", staging)
        _write_bytes(staging / "config.json", config_bytes, fault, "main_config_write", staging)
        if prerequisite_artifacts is not None:
            _write_bytes(staging / "frozen_frontend.json", prerequisite_artifacts.frontend_bytes, fault, "main_frontend_write", staging)
            _write_bytes(staging / "readout.json", prerequisite_artifacts.readout_bytes, fault, "main_readout_write", staging)
        _inject(fault, "main_checksum_write", staging, output)
        (staging / "checksums.sha256").write_bytes(_checksum_payload(staging))
        _inject(fault, "main_checksum_self_check", staging, output)
        _self_check_complete_package(staging, audit, prerequisite_artifacts)
        _inject(fault, "main_publish_race", staging, output)
        _inject(fault, "main_publish", staging, output)
        _atomic_publish_no_replace(staging, output)
        staging = None
    finally:
        _safe_remove_staging(staging, output.parent)


def _minimal_invalid_audit(
    reason_code: str,
    failure_phase: str,
    exception_type: str,
    *,
    actual_package_path: Path | None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema_version": 1,
        "output_schema": OUTPUT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": INVALID_STATUS,
        "valid_experiment": False,
        "reason_code": reason_code,
        "reason_detail": "experiment invalidated before package completion",
        "failure_phase": failure_phase,
        "exception_type": exception_type,
        "formal_result": False,
        "stage_progression_allowed": False,
        "science_metrics_present": False,
        "fresh_held_out_read": False,
        "package_written": actual_package_path is not None,
        "actual_package_path": str(actual_package_path) if actual_package_path is not None else None,
    }
    if context:
        audit["integrity_context"] = context
    return audit


def _fallback_candidates(output: Path, prefer_requested: bool) -> list[Path]:
    candidates = [output] if prefer_requested else []
    candidates.extend(output.with_name(f"{output.name}.invalid.{index:04d}") for index in range(1, _FALLBACK_LIMIT + 1))
    return candidates


def _write_invalid_best_effort(
    output: Path,
    command: list[str],
    *,
    reason_code: str,
    failure_phase: str,
    exception_type: str,
    context: dict[str, Any] | None,
    prefer_requested: bool,
    fault: _TestFault | None,
) -> _RunOutcome:
    staging: Path | None = None
    try:
        _validate_output_parent(output)
        for target in _fallback_candidates(output, prefer_requested):
            if _lstat(target) is not None:
                continue
            try:
                staging = _create_staging(target, "invalid", fault, "fallback_staging_mkdir")
                audit = _minimal_invalid_audit(
                    reason_code,
                    failure_phase,
                    exception_type,
                    actual_package_path=target,
                    context=context,
                )
                _write_bytes(staging / "audit.json", canonical_json_bytes(audit) + b"\n", fault, "fallback_audit_write", staging)
                _write_bytes(staging / "command.txt", (shlex.join(command) + "\n").encode("utf-8"), fault, "fallback_command_write", staging)
                _inject(fault, "fallback_checksum_write", staging, target)
                (staging / "checksums.sha256").write_bytes(_checksum_payload(staging))
                _inject(fault, "fallback_publish_race", staging, target)
                _inject(fault, "fallback_publish", staging, target)
                _atomic_publish_no_replace(staging, target)
                staging = None
                return _RunOutcome(audit=audit, actual_package_path=target, package_written=True)
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
            actual_package_path=None,
            context=None,
        )
        return _RunOutcome(audit=audit, actual_package_path=None, package_written=False)


def _invalid_context(repo_root: Path, manifest: Path) -> dict[str, Any]:
    context: dict[str, Any] = {}
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
    _test_fault: _TestFault | None = None,
) -> _RunOutcome:
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
    output = _lexical_absolute(output)
    try:
        _assert_output_unclaimed(output)
    except InvalidExperiment as error:
        return _write_invalid_best_effort(
            output,
            invocation,
            reason_code=error.reason_code,
            failure_phase="output_precheck",
            exception_type=type(error).__name__,
            context=None,
            prefer_requested=False,
            fault=_test_fault,
        )
    except Exception as error:
        return _write_invalid_best_effort(
            output,
            invocation,
            reason_code="OUTPUT_PRECHECK_FAILED",
            failure_phase="output_precheck",
            exception_type=type(error).__name__,
            context=None,
            prefer_requested=False,
            fault=_test_fault,
        )

    try:
        _inject(_test_fault, "before_preflight", None, output)
        checked = preflight(repo_root, manifest, declared_source_commit=source_commit)
        _inject(_test_fault, "before_evaluate", None, output)
        candidates, selected = evaluate_gate(checked.features)
        audit = valid_audit(checked, candidates, selected)
        prerequisite_artifacts = build_selected_prerequisite(checked, audit, selected) if selected is not None else None
    except InvalidExperiment as error:
        return _write_invalid_best_effort(
            output,
            invocation,
            reason_code=error.reason_code,
            failure_phase="preflight_or_science",
            exception_type=type(error).__name__,
            context=_invalid_context(repo_root, manifest),
            prefer_requested=True,
            fault=_test_fault,
        )
    except Exception as error:
        return _write_invalid_best_effort(
            output,
            invocation,
            reason_code="UNEXPECTED_RUNTIME_FAILURE",
            failure_phase="preflight_or_science",
            exception_type=type(error).__name__,
            context=None,
            prefer_requested=True,
            fault=_test_fault,
        )

    try:
        _write_complete_package(
            output,
            audit,
            invocation,
            manifest_bytes=checked.manifest_bytes,
            config_bytes=checked.config_bytes,
            prerequisite_artifacts=prerequisite_artifacts,
            fault=_test_fault,
        )
    except Exception as error:
        return _write_invalid_best_effort(
            output,
            invocation,
            reason_code="PACKAGE_WRITE_FAILED",
            failure_phase="package_write",
            exception_type=type(error).__name__,
            context=None,
            prefer_requested=False,
            fault=_test_fault,
        )
    return _RunOutcome(audit=audit, actual_package_path=output, package_written=True)


def _response(outcome: _RunOutcome) -> dict[str, Any]:
    return {
        "actual_package_path": str(outcome.actual_package_path) if outcome.actual_package_path is not None else None,
        "package_written": outcome.package_written,
        "reason_code": outcome.audit["reason_code"],
        "status": outcome.audit["status"],
    }


def main(argv: list[str] | None = None, *, _test_fault: _TestFault | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    arguments = parser.parse_args(argv)
    outcome = run(
        arguments.manifest,
        arguments.output,
        source_commit=arguments.source_commit,
        command=sys.argv if argv is None else [str(ROOT / "experiments" / "run_learned_observation_l1_v2_development.py"), *argv],
        _test_fault=_test_fault,
    )
    print(canonical_json_bytes(_response(outcome)).decode("utf-8"))
    return 0 if outcome.audit["valid_experiment"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
