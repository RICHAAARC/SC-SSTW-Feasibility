from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from src.sc_sstw_feasibility.learned_observation_l1_v2 import (
    ABSOLUTE_THRESHOLDS,
    CONFIG_PATH as G0_CONFIG_PATH,
    OUTPUT_SCHEMA as G0_OUTPUT_SCHEMA,
    PERMITTED_INPUT_IDS,
    PROTOCOL_ID as G0_PROTOCOL_ID,
    REQUIRED_SOURCE_PATHS as G0_REQUIRED_SOURCE_PATHS,
    SUCCESS_STATUS as G0_SUCCESS_STATUS,
    transform,
)
from src.sc_sstw_feasibility.rc1_method_validation import (
    ARTIFACT_NAMES,
    CONFIG_PATH,
    CONFIG_RAW_SHA256,
    CONDITIONS,
    EXECUTION_SCHEMA,
    FORBIDDEN_CLAIM_TOKENS,
    MANIFEST_SCHEMA,
    NOTEBOOK_PATH,
    OUTPUT_SCHEMA,
    PASS_CONCLUSION,
    PLAN_PATH,
    PLAN_RAW_SHA256,
    PROTOCOL_ID,
    REQUIRED_SOURCE_PATHS,
    RUNNER_PATH,
    START_INDICES,
    STATUS_FAIL,
    STATUS_INVALID,
    STATUS_PASS,
    STATUS_PREREQUISITE,
    TEMPLATES,
    FrozenPrerequisite,
    InvalidExperiment,
    PrerequisiteNotMet,
    canonical_json_bytes,
    evaluate_execution,
    expected_matched_parameters,
    frontend_definition,
    preflight,
    schedule_a,
    schedule_b,
    schedule_preflight,
    sha256_bytes,
    sha256_file,
    validate_execution_record,
    validate_frozen_config_bytes,
    validate_frozen_plan_bytes,
    validate_manifest_schema,
    validate_prerequisite_package,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HEX64 = "a" * 64
HEX40 = "b" * 40


def _write_json(path: Path, value: object, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2 if pretty else None, sort_keys=False, allow_nan=False).encode("utf-8") + b"\n"
    path.write_bytes(payload)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _raw_features(schedule: tuple[tuple[float, float], ...]) -> np.ndarray:
    features = np.tile(np.linspace(-0.03, 0.03, 30), (13, 1))
    features[:, 0] = np.asarray(schedule)[:, 0]
    features[:, 1] = np.asarray(schedule)[:, 1]
    return features


def _synthetic_readout() -> np.ndarray:
    features = _raw_features(schedule_a())
    design = np.column_stack((transform(features, "A1"), np.ones(13)))
    return np.linalg.lstsq(design, np.asarray(schedule_a()), rcond=None)[0]


def _make_prerequisite(path: Path, *, status: str = G0_SUCCESS_STATUS, candidate: str = "A1") -> str:
    thresholds = dict(ABSOLUTE_THRESHOLDS)
    audit = {
        "protocol_id": G0_PROTOCOL_ID,
        "output_schema": G0_OUTPUT_SCHEMA,
        "valid_experiment": True,
        "status": status,
        "formal_result": False,
        "stage_progression_allowed": False,
        "selected_candidate": candidate,
        "source_state": {"head": HEX40, "tree": "c" * 40, "dirty": False},
        "source_file_sha256": {name: HEX64 for name in G0_REQUIRED_SOURCE_PATHS},
        "config_identity": {"path": G0_CONFIG_PATH, "sha256": "d" * 64},
        "input_sha256": {str(item): "e" * 64 for item in PERMITTED_INPUT_IDS},
        "candidates": [{"candidate": candidate, "development_gate_pass": True, "derived_envelope": thresholds}],
    }
    readout = {"schema_version": 1, "selected_candidate": candidate, "shape": [31, 2], "coefficients": _synthetic_readout().tolist()}
    _write_json(path / "audit.json", audit)
    _write_json(path / "readout.json", readout)
    audit_sha = sha256_file(path / "audit.json")
    readout_sha = sha256_file(path / "readout.json")
    frontend = {
        "schema_version": 1,
        "selected_candidate": candidate,
        "frontend_definition_sha256": sha256_bytes(canonical_json_bytes(frontend_definition(candidate))) if candidate in {"A1", "A2"} else HEX64,
        "readout_path": "readout.json",
        "readout_sha256": readout_sha,
        "thresholds": thresholds,
        "g0_audit_sha256": audit_sha,
        "source_head": HEX40,
        "source_tree": "c" * 40,
    }
    _write_json(path / "frozen_frontend.json", frontend)
    checksums = "".join(f"{sha256_file(path / name)}  {name}\n" for name in ("audit.json", "frozen_frontend.json", "readout.json"))
    (path / "checksums.sha256").write_text(checksums, encoding="utf-8")
    return sha256_file(path / "checksums.sha256")


def _make_repo(path: Path) -> tuple[str, str]:
    for relative in REQUIRED_SOURCE_PATHS:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    init = path / "src/sc_sstw_feasibility/__init__.py"
    init.write_text("", encoding="utf-8")
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "synthetic@example.invalid")
    _git(path, "config", "user.name", "Synthetic Fixture")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "synthetic RC1 source")
    return _git(path, "rev-parse", "HEAD"), _git(path, "rev-parse", "HEAD^{tree}")


def _make_manifest(path: Path, repo: Path, prerequisite: Path, checksums_sha: str, *, head: str, tree: str) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "manifest_schema": MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "expected_source": {"head": head, "tree": tree},
        "source_files": {relative: sha256_file(repo / relative) for relative in REQUIRED_SOURCE_PATHS},
        "config": {"path": CONFIG_PATH, "sha256": CONFIG_RAW_SHA256},
        "plan": {"path": PLAN_PATH, "sha256": PLAN_RAW_SHA256},
        "prerequisite": {"package_path": str(prerequisite), "checksums_sha256": checksums_sha},
        "command_schema": {
            "runner": RUNNER_PATH,
            "required_arguments": ["--manifest", "--output"],
            "optional_arguments": ["--source-commit", "--execution-package", "--generate", "--synthetic-fixture"],
        },
        "output_schema": OUTPUT_SCHEMA,
    }
    _write_json(path, manifest, pretty=True)
    return manifest


def _artifact_features(condition: str, group_index: int) -> np.ndarray:
    if condition == "A":
        return _raw_features(schedule_a())
    if condition == "B":
        return _raw_features(schedule_b())
    return np.random.default_rng(700 + group_index).normal(size=(13, 30))


def _make_execution(path: Path, frozen: object) -> Path:
    groups = []
    for group_index, plan_group in enumerate(frozen.plan["groups"]):
        parameters = expected_matched_parameters(frozen.config, plan_group)
        parameter_sha = sha256_bytes(canonical_json_bytes(parameters))
        latent_sha = hashlib.sha256(f"latent:{group_index}".encode()).hexdigest()
        conditions = []
        attempts = []
        for condition in CONDITIONS:
            condition_dir = path / plan_group["group_id"] / condition
            condition_dir.mkdir(parents=True)
            video = condition_dir / "saved.mp4"
            video.write_bytes(f"pure synthetic MP4 bytes {group_index} {condition}".encode())
            feature_payload = {
                "source": "pure_synthetic_saved_mp4_fixture",
                "video_sha256": sha256_file(video),
                "features": _artifact_features(condition, group_index).tolist(),
            }
            _write_json(condition_dir / "features.json", feature_payload)
            for name in ("stdout", "stderr", "config", "environment", "command", "integrity"):
                (condition_dir / f"{name}.txt").write_text(f"synthetic {name}\n", encoding="utf-8")
            files = {
                "video": video,
                "features": condition_dir / "features.json",
                "stdout": condition_dir / "stdout.txt",
                "stderr": condition_dir / "stderr.txt",
                "config": condition_dir / "config.txt",
                "environment": condition_dir / "environment.txt",
                "command": condition_dir / "command.txt",
                "integrity": condition_dir / "integrity.txt",
            }
            artifacts = {name: {"path": str(files[name].relative_to(path)), "sha256": sha256_file(files[name])} for name in ARTIFACT_NAMES}
            conditions.append({
                "condition": condition,
                "schedule_id": {"OFF": "NONE", "A": "A", "B": "B"}[condition],
                "carrier_enabled": condition != "OFF",
                "initial_latent_sha256": latent_sha,
                "matched_parameters_sha256": parameter_sha,
                "artifacts": artifacts,
            })
            attempts.append({"condition": condition, "attempt_index": 0, "outcome": "success", "matched_parameters_sha256": parameter_sha})
        groups.append({
            "group_id": plan_group["group_id"],
            "content_grammar": plan_group["content_grammar"],
            "prompt": plan_group["prompt"],
            "prompt_sha256": parameters["prompt_sha256"],
            "seed": plan_group["seed"],
            "matched_parameters": parameters,
            "conditions": conditions,
            "attempts": attempts,
        })
    record = {
        "schema_version": 1,
        "execution_schema": EXECUTION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": PLAN_RAW_SHA256,
        "prerequisite_identity": frozen.prerequisite.identity(),
        "synthetic_fixture": True,
        "groups": groups,
    }
    record_path = path / "execution.json"
    _write_json(record_path, record, pretty=True)
    return record_path


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    prerequisite = tmp_path / "prerequisite"
    manifest_path = tmp_path / "authorization.json"
    head, tree = _make_repo(repo)
    checksums_sha = _make_prerequisite(prerequisite)
    manifest = _make_manifest(manifest_path, repo, prerequisite, checksums_sha, head=head, tree=tree)
    frozen = preflight(repo, manifest_path, declared_source_commit=head)
    execution_path = _make_execution(tmp_path / "execution", frozen)
    return {"repo": repo, "prerequisite": prerequisite, "manifest_path": manifest_path, "manifest": manifest, "head": head, "tree": tree, "frozen": frozen, "execution_path": execution_path}


def _rewrite_record(path: Path, mutate) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record)
    _write_json(path, record, pretty=True)


def _reason(exc: pytest.ExceptionInfo[InvalidExperiment]) -> str:
    return exc.value.reason_code


def test_schedule_structure_cross_residual_and_heterogeneous_plan() -> None:
    a, b = schedule_a(), schedule_b()
    assert a[:3] == b[:3]
    assert sorted(a) == sorted(b)
    assert b[4] == a[5] and b[5] == a[4]
    assert a[6:] == b[6:]
    result = schedule_preflight()
    assert result["passed"] is True
    assert result["A_observation_by_B_template"] == pytest.approx(0.761311945935564, abs=1e-9)
    config = validate_frozen_config_bytes((REPO_ROOT / CONFIG_PATH).read_bytes())
    plan = validate_frozen_plan_bytes((REPO_ROOT / PLAN_PATH).read_bytes(), config)
    assert len({group["content_grammar"] for group in plan["groups"]}) == 2
    assert config["g0_baseline"]["independent_implementation_gate"] == "NOT_PASSED"


def test_valid_synthetic_execution_scans_every_case_without_averaging(environment: dict[str, object]) -> None:
    record, artifacts = validate_execution_record(environment["execution_path"], environment["frozen"], synthetic_fixture=True)
    groups, passed = evaluate_execution(record, environment["execution_path"], artifacts, environment["frozen"], synthetic_fixture=True)
    assert passed is True
    assert len(groups) == 2
    for group in groups:
        assert group["group_pass"] is True
        for condition in group["conditions"]:
            assert tuple(condition["templates"]) == TEMPLATES
            assert all([window["start"] for window in condition["templates"][template]] == list(START_INDICES) for template in TEMPLATES)
            assert condition["case_pass"] is True


def test_missing_prerequisite_is_not_invalid_and_precedes_generation_access(tmp_path: Path) -> None:
    with pytest.raises(PrerequisiteNotMet) as caught:
        validate_prerequisite_package(tmp_path / "absent", HEX64)
    assert caught.value.reason_code == "PREREQUISITE_PACKAGE_MISSING"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda audit: audit.update(selected_candidate="A3"), "PREREQUISITE_CANDIDATE_INVALID"),
        (lambda audit: audit.update(formal_result=True), "PREREQUISITE_EVIDENCE_BOUNDARY_INVALID"),
        (lambda audit: audit["candidates"][0]["derived_envelope"].update(max_residual=0.251), "PREREQUISITE_THRESHOLDS_INVALID"),
    ],
)
def test_forged_candidate_boundary_or_threshold_is_invalid(tmp_path: Path, mutation, reason: str) -> None:
    package = tmp_path / "prerequisite"
    checksums_sha = _make_prerequisite(package)
    audit = json.loads((package / "audit.json").read_text(encoding="utf-8"))
    mutation(audit)
    _write_json(package / "audit.json", audit)
    checksums = "".join(f"{sha256_file(package / name)}  {name}\n" for name in ("audit.json", "frozen_frontend.json", "readout.json"))
    (package / "checksums.sha256").write_text(checksums, encoding="utf-8")
    with pytest.raises(InvalidExperiment) as caught:
        validate_prerequisite_package(package, sha256_file(package / "checksums.sha256"))
    assert _reason(caught) == reason


def test_damaged_prerequisite_hash_is_invalid(tmp_path: Path) -> None:
    package = tmp_path / "prerequisite"
    checksums_sha = _make_prerequisite(package)
    (package / "readout.json").write_bytes(b"{}\n")
    with pytest.raises(InvalidExperiment) as caught:
        validate_prerequisite_package(package, checksums_sha)
    assert _reason(caught) == "PREREQUISITE_FILE_HASH_MISMATCH"


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("fake_declaration", "SOURCE_COMMIT_DECLARATION_MISMATCH"),
        ("head", "HEAD_MISMATCH"),
        ("tree", "TREE_MISMATCH"),
        ("source_code_hash", "SOURCE_FILE_HASH_MISMATCH"),
        ("protocol_hash", "SOURCE_FILE_HASH_MISMATCH"),
        ("notebook_hash", "SOURCE_FILE_HASH_MISMATCH"),
        ("config_identity", "CONFIG_IDENTITY_MISMATCH"),
    ],
)
def test_source_manifest_mismatches_fail_closed(environment: dict[str, object], kind: str, reason: str) -> None:
    manifest = copy.deepcopy(environment["manifest"])
    declared = environment["head"]
    if kind == "fake_declaration":
        declared = "f" * 40
    elif kind == "head":
        manifest["expected_source"]["head"] = "f" * 40
    elif kind == "tree":
        manifest["expected_source"]["tree"] = "f" * 40
    elif kind.endswith("_hash"):
        changed_path = {
            "source_code_hash": "src/sc_sstw_feasibility/rc1_method_validation.py",
            "protocol_hash": "protocols/rc1_method_validation.md",
            "notebook_hash": NOTEBOOK_PATH,
        }[kind]
        manifest["source_files"][changed_path] = "f" * 64
    else:
        manifest["config"]["sha256"] = "f" * 64
    _write_json(environment["manifest_path"], manifest)
    with pytest.raises(InvalidExperiment) as caught:
        preflight(environment["repo"], environment["manifest_path"], declared_source_commit=declared)
    assert _reason(caught) == reason


def test_dirty_source_fails_before_prerequisite_access(environment: dict[str, object]) -> None:
    (environment["repo"] / "untracked.txt").write_text("dirty", encoding="utf-8")
    shutil.rmtree(environment["prerequisite"])
    with pytest.raises(InvalidExperiment) as caught:
        preflight(environment["repo"], environment["manifest_path"])
    assert _reason(caught) == "DIRTY_WORKTREE"


def test_unreadable_git_fails_closed_before_prerequisite_access(tmp_path: Path, environment: dict[str, object]) -> None:
    nongit = tmp_path / "not-a-repository"
    nongit.mkdir()
    shutil.rmtree(environment["prerequisite"])
    with pytest.raises(InvalidExperiment) as caught:
        preflight(nongit, environment["manifest_path"])
    assert _reason(caught) == "GIT_STATE_UNREADABLE"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda r: r["groups"][0]["matched_parameters"].update(guidance_scale=6.0), "TRIPLET_PARAMETER_MISMATCH"),
        (lambda r: r["groups"][0]["conditions"][1].update(initial_latent_sha256="f" * 64), "TRIPLET_LATENT_MISMATCH"),
        (lambda r: r["groups"][0]["conditions"].pop(), "TRIPLET_CONDITION_SET_MISMATCH"),
        (lambda r: r["groups"][0]["conditions"].insert(2, copy.deepcopy(r["groups"][0]["conditions"][1])), "TRIPLET_CONDITION_SET_MISMATCH"),
        (lambda r: r["groups"][0]["conditions"][1].update(schedule_id="B"), "TRIPLET_CONDITION_LABEL_MISMATCH"),
        (lambda r: r["groups"][0].update(prompt="changed"), "TRIPLET_PROMPT_OR_SEED_MISMATCH"),
    ],
)
def test_triplet_integrity_mismatches_fail_before_artifact_reads(environment: dict[str, object], mutation, reason: str) -> None:
    mutation_record = environment["execution_path"]
    _rewrite_record(mutation_record, mutation)
    shutil.rmtree(mutation_record.parent / "orbital_glass")
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(mutation_record, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == reason


def test_forbidden_formal_id_path_is_rejected_before_read(environment: dict[str, object], tmp_path: Path) -> None:
    disguised = tmp_path / "disguised-41008-input.json"
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(disguised, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == "FORBIDDEN_FORMAL_PATH"
    with pytest.raises(InvalidExperiment) as caught:
        preflight(environment["repo"], tmp_path / "fake41007manifest.json")
    assert _reason(caught) == "FORBIDDEN_FORMAL_PATH"


def test_artifact_path_escape_is_rejected_before_external_read(environment: dict[str, object], tmp_path: Path) -> None:
    record_path = environment["execution_path"]
    external = tmp_path / "external-secret.bin"
    external.write_bytes(b"must not be opened")
    record = json.loads(record_path.read_text())
    record["groups"][0]["conditions"][0]["artifacts"]["video"] = {"path": str(external), "sha256": sha256_file(external)}
    _write_json(record_path, record)
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == "EXECUTION_ARTIFACT_PATH_INVALID"


def test_artifact_replacement_and_retry_parameter_change_are_invalid(environment: dict[str, object]) -> None:
    record_path = environment["execution_path"]
    record = json.loads(record_path.read_text())
    artifact = record["groups"][0]["conditions"][0]["artifacts"]["video"]
    (record_path.parent / artifact["path"]).write_bytes(b"replacement")
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == "TRIPLET_ARTIFACT_HASH_MISMATCH"

    artifact["sha256"] = sha256_file(record_path.parent / artifact["path"])
    record["groups"][0]["attempts"][0]["matched_parameters_sha256"] = "f" * 64
    _write_json(record_path, record)
    with pytest.raises(InvalidExperiment) as caught:
        validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    assert _reason(caught) == "TRIPLET_ATTEMPT_LOG_INVALID"


@pytest.mark.parametrize("bad_condition", ["A", "B", "OFF"])
def test_cross_template_wrong_window_or_off_false_positive_fails_entire_screen(environment: dict[str, object], bad_condition: str) -> None:
    record_path = environment["execution_path"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    target = next(item for item in record["groups"][0]["conditions"] if item["condition"] == bad_condition)
    feature_path = record_path.parent / target["artifacts"]["features"]["path"]
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    if bad_condition == "A":
        payload["features"] = _raw_features(schedule_b()).tolist()
    elif bad_condition == "B":
        payload["features"] = _raw_features(schedule_a()).tolist()
    else:
        payload["features"] = _raw_features(schedule_a()).tolist()
    _write_json(feature_path, payload)
    target["artifacts"]["features"]["sha256"] = sha256_file(feature_path)
    _write_json(record_path, record, pretty=True)
    validated, artifacts = validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    groups, passed = evaluate_execution(validated, record_path, artifacts, environment["frozen"], synthetic_fixture=True)
    assert passed is False
    assert groups[0]["group_pass"] is False
    assert groups[1]["group_pass"] is True


def test_fixed_time_wrong_window_pseudocorrelation_is_rejected(environment: dict[str, object]) -> None:
    record_path = environment["execution_path"]
    record = json.loads(record_path.read_text())
    target = next(item for item in record["groups"][0]["conditions"] if item["condition"] == "A")
    feature_path = record_path.parent / target["artifacts"]["features"]["path"]
    payload = json.loads(feature_path.read_text())
    shifted = np.roll(np.asarray(schedule_a()), 1, axis=0)
    payload["features"] = _raw_features(tuple(map(tuple, shifted))).tolist()
    _write_json(feature_path, payload)
    target["artifacts"]["features"]["sha256"] = sha256_file(feature_path)
    _write_json(record_path, record)
    validated, artifacts = validate_execution_record(record_path, environment["frozen"], synthetic_fixture=True)
    groups, passed = evaluate_execution(validated, record_path, artifacts, environment["frozen"], synthetic_fixture=True)
    assert passed is False
    a_case = next(item for item in groups[0]["conditions"] if item["condition"] == "A")
    assert a_case["templates"]["A"][1]["accepted"] is True
    assert a_case["case_pass"] is False


def test_manifest_forbids_candidate_schedule_and_threshold_injection(environment: dict[str, object]) -> None:
    for unexpected in ("selected_candidate", "schedule_override", "threshold_override", "extra_input"):
        manifest = copy.deepcopy(environment["manifest"])
        manifest[unexpected] = "forbidden"
        with pytest.raises(InvalidExperiment) as caught:
            validate_manifest_schema(manifest)
        assert _reason(caught) == "MANIFEST_SCHEMA_MISMATCH"


def test_cli_real_path_emits_pass_fail_invalid_and_prerequisite_packages(environment: dict[str, object], tmp_path: Path) -> None:
    runner = environment["repo"] / RUNNER_PATH
    base = [sys.executable, "-B", str(runner), "--manifest", str(environment["manifest_path"]), "--source-commit", environment["head"]]
    pass_output = tmp_path / "cli-pass"
    completed = subprocess.run([*base, "--output", str(pass_output), "--execution-package", str(environment["execution_path"]), "--synthetic-fixture"], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((pass_output / "audit.json").read_text())["status"] == STATUS_PASS

    missing_output = tmp_path / "cli-prerequisite"
    completed = subprocess.run([*base, "--output", str(missing_output)], text=True, capture_output=True)
    missing = json.loads((missing_output / "audit.json").read_text())
    assert completed.returncode == 2 and missing["status"] == STATUS_PREREQUISITE
    assert missing["generation_inputs_accessed"] is False and missing["science_metrics_present"] is False

    invalid_output = tmp_path / "cli-invalid"
    completed = subprocess.run([*base, "--output", str(invalid_output), "--execution-package", str(environment["execution_path"]), "--synthetic-fixture", "--source-commit", "f" * 40], text=True, capture_output=True)
    invalid = json.loads((invalid_output / "audit.json").read_text())
    assert completed.returncode == 2 and invalid["status"] == STATUS_INVALID
    assert invalid["science_metrics_present"] is False and "groups" not in invalid

    record = json.loads(environment["execution_path"].read_text())
    target = record["groups"][0]["conditions"][0]
    feature_path = environment["execution_path"].parent / target["artifacts"]["features"]["path"]
    payload = json.loads(feature_path.read_text())
    payload["features"] = _raw_features(schedule_a()).tolist()
    _write_json(feature_path, payload)
    target["artifacts"]["features"]["sha256"] = sha256_file(feature_path)
    _write_json(environment["execution_path"], record)
    fail_output = tmp_path / "cli-fail"
    completed = subprocess.run([*base, "--output", str(fail_output), "--execution-package", str(environment["execution_path"]), "--synthetic-fixture"], text=True, capture_output=True)
    failed = json.loads((fail_output / "audit.json").read_text())
    assert completed.returncode == 3 and failed["status"] == STATUS_FAIL

    passing = json.loads((pass_output / "audit.json").read_text())
    for audit in (passing, missing, invalid, failed):
        assert audit["formal_result"] is False and audit["stage_progression_allowed"] is False
    assert passing["conclusion"] == PASS_CONCLUSION
    assert not any(token in json.dumps(passing).lower() for token in FORBIDDEN_CLAIM_TOKENS)


def test_status_vocabulary_claim_ceiling_and_notebook_static_structure() -> None:
    assert {STATUS_PASS, STATUS_FAIL, STATUS_INVALID, STATUS_PREREQUISITE} == {
        "RC1_VALID_PASS", "RC1_VALID_FAIL", "INVALID_EXPERIMENT", "PREREQUISITE_NOT_MET"
    }
    assert "method" not in PASS_CONCLUSION.lower()
    notebook = json.loads((REPO_ROOT / NOTEBOOK_PATH).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = (
        "SC_SSTW_RC1_EXACT_REF", "git', 'checkout', '--detach'", "drive.mount", "torch.cuda.is_available",
        "--generate", "try:", "finally:", "make_archive", "shutil.copy2", "sha256(drive_copy)", "testzip()", "verification.json", "packaged = True",
    )
    assert all(token in source for token in required)
    assert "checkout exact ref" not in source.lower()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), NOTEBOOK_PATH, "exec")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text())
    assert tuple(config["forbidden_claims"]) == FORBIDDEN_CLAIM_TOKENS
