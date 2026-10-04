"""Verification gates for portable model and public release trees."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mentat_sys1.audit.provenance import sha256_file
from mentat_sys1.contracts import ProjectConfig

BASE_MODEL_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "decision_readout.json",
    "decision_readout.safetensors",
)
GENERATED_MODEL_FILES = (
    "model-manifest.json",
    "SHA256SUMS",
)
MIGRATED_MODEL_FILES = (*BASE_MODEL_FILES, *GENERATED_MODEL_FILES)
CALIBRATED_MODEL_FILES = (
    *BASE_MODEL_FILES,
    "calibration.json",
    *GENERATED_MODEL_FILES,
)
PUBLICATION_EVIDENCE_FILES = (
    "calibrated-model-receipt.json",
    "calibration-report.json",
    "contamination-report.json",
    "legacy-public-evaluation.json",
)
PUBLICATION_DOCUMENTS = {
    "README.md": "README.md",
    "MODEL_CARD.md": "docs/MODEL_CARD.md",
    "REPRODUCIBILITY.md": "docs/REPRODUCIBILITY.md",
    "JEVBENCH_SUBMISSION.md": "docs/JEVBENCH_SUBMISSION.md",
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
}
PUBLICATION_GENERATED_FILES = (
    "release-manifest.json",
    "SHA256SUMS",
)
CALIBRATION_ROWS = 1120
CALIBRATION_SHA256 = (
    "a116f4d228e25bac5df61b9e52191adfe4181a0852f6050d9040cee7f35b0a02"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".txt", ".toml", ".yaml", ".yml"})
FORBIDDEN_TEXT_PATTERNS = (
    re.compile(r"/data/"),
    re.compile(r"/Users/"),
    re.compile(r"/home/"),
    re.compile(r"file://", re.IGNORECASE),
    re.compile(
        r"https?://[^\s\"']*(?:internal|corp|bytedance|localhost|127\.0\.0\.1)",
        re.IGNORECASE,
    ),
)
SECRET_KEY_PATTERN = re.compile(
    r"^(?:access[_-]?token|api[_-]?key|password|secret)$",
    re.IGNORECASE,
)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _walk_json(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                isinstance(key, str)
                and SECRET_KEY_PATTERN.fullmatch(key)
                and item not in (None, "")
            ):
                raise ValueError("forbidden public content: embedded secret")
            _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            _walk_json(item)


def scan_public_tree(root: Path) -> dict[str, object]:
    base = Path(root)
    if not base.is_dir():
        raise ValueError(f"public tree is not a directory: {base}")
    scanned: list[str] = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "LICENSE",
            "NOTICE",
            "SHA256SUMS",
        }:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern.search(text):
                raise ValueError(
                    f"forbidden public content in {relative}: {pattern.pattern}"
                )
        if path.name != "NOTICE" and "imajev" in text.casefold():
            raise ValueError(
                f"forbidden public content in {relative}: historical identity"
            )
        if path.suffix.lower() == ".json":
            _walk_json(json.loads(text))
        scanned.append(relative)
    return {"files_scanned": len(scanned), "files": scanned}


def _verify_manifest_files(
    model_dir: Path,
    manifest: dict[str, Any],
    *,
    require_calibration: bool,
) -> None:
    records = manifest.get("files")
    if not isinstance(records, dict):
        raise ValueError("model manifest files must be an object")
    expected_names = set(BASE_MODEL_FILES)
    if require_calibration:
        expected_names.add("calibration.json")
    if set(records) != expected_names:
        raise ValueError("model manifest file set differs")
    for name in sorted(expected_names):
        record = records[name]
        if not isinstance(record, dict):
            raise ValueError(f"invalid model manifest record: {name}")
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"missing model file: {name}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"manifest SHA-256 mismatch: {name}")
        if record.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"manifest size mismatch: {name}")


def _verify_checksums(model_dir: Path, *, require_calibration: bool) -> None:
    checksum_path = model_dir / "SHA256SUMS"
    observed: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError("invalid SHA256SUMS line")
        digest, name = parts
        if name in observed or "/" in name or "\\" in name:
            raise ValueError("invalid SHA256SUMS file name")
        observed[name] = digest
    expected = set(BASE_MODEL_FILES) | {"model-manifest.json"}
    if require_calibration:
        expected.add("calibration.json")
    if set(observed) != expected:
        raise ValueError("SHA256SUMS file set differs")
    for name, digest in observed.items():
        if digest != sha256_file(model_dir / name):
            raise ValueError(f"SHA256SUMS mismatch: {name}")


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"calibration {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"calibration {name} must be finite")
    return result


def validate_calibration(
    *,
    config: ProjectConfig,
    calibration_path: Path,
) -> dict[str, object]:
    payload = _load_json_object(calibration_path)
    if (
        payload.get("schema_version") != 1
        or payload.get("method") != "scalar_temperature"
    ):
        raise ValueError("unsupported calibration schema")
    temperature = _finite_number(
        payload.get("temperature"),
        name="temperature",
    )
    if temperature <= 0:
        raise ValueError("calibration temperature must be positive")
    split = payload.get("split")
    if not isinstance(split, dict) or split != {
        "rows": CALIBRATION_ROWS,
        "sha256": CALIBRATION_SHA256,
    }:
        raise ValueError("calibration split identity differs")
    model = payload.get("model")
    if not isinstance(model, dict) or model != {
        "adapter_sha256": config.release.adapter_sha256,
        "readout_sha256": config.release.readout_sha256,
    }:
        raise ValueError("calibration model identity differs")
    if payload.get("predictions_unchanged") is not True:
        raise ValueError("calibration changed predictions")
    before = payload.get("before")
    after = payload.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("calibration metrics are missing")
    before_nll = _finite_number(before.get("nll"), name="before NLL")
    after_nll = _finite_number(after.get("nll"), name="after NLL")
    if after_nll > before_nll + 1e-12:
        raise ValueError("calibration regressed NLL")
    digest = sha256_file(calibration_path)
    return {
        "temperature": temperature,
        "calibration_sha256": digest,
        "calibration_version": f"scalar_temperature-v1:{digest[:12]}",
    }


def verify_migrated_model(
    *,
    config: ProjectConfig,
    model_dir: Path,
    require_calibration: bool = False,
) -> dict[str, object]:
    root = Path(model_dir)
    required_files = (
        CALIBRATED_MODEL_FILES if require_calibration else MIGRATED_MODEL_FILES
    )
    missing = [name for name in required_files if not (root / name).is_file()]
    if missing:
        raise ValueError(f"missing model file: {', '.join(missing)}")

    manifest = _load_json_object(root / "model-manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported model manifest schema")
    if manifest.get("model_id") != config.project_id:
        raise ValueError("model manifest identity differs")
    if manifest.get("base_model") != config.base_model.model_dump(mode="json"):
        raise ValueError("model manifest base identity differs")
    _verify_manifest_files(
        root,
        manifest,
        require_calibration=require_calibration,
    )
    _verify_checksums(root, require_calibration=require_calibration)

    adapter_config = _load_json_object(root / "adapter_config.json")
    if (
        adapter_config.get("base_model_name_or_path") != config.base_model.repository
        or adapter_config.get("revision") != config.base_model.revision
    ):
        raise ValueError("portable adapter base identity differs")

    readout = _load_json_object(root / "decision_readout.json")
    codes = readout.get("codes")
    if (
        readout.get("version") != 1
        or not isinstance(codes, list)
        or len(codes) != config.runtime.readout_codes
    ):
        raise ValueError("decision readout schema differs")

    adapter_sha = sha256_file(root / "adapter_model.safetensors")
    readout_sha = sha256_file(root / "decision_readout.safetensors")
    if adapter_sha != config.release.adapter_sha256:
        raise ValueError("registered adapter identity differs")
    if readout_sha != config.release.readout_sha256:
        raise ValueError("registered readout identity differs")
    calibration = (
        validate_calibration(
            config=config,
            calibration_path=root / "calibration.json",
        )
        if require_calibration
        else {}
    )
    scan_public_tree(root)
    return {
        "schema_version": 1,
        "model_id": config.project_id,
        "adapter_sha256": adapter_sha,
        "readout_sha256": readout_sha,
        "readout_codes": len(codes),
        "base_model": config.base_model.model_dump(mode="json"),
        **calibration,
    }


def prediction_digest(records: Sequence[Mapping[str, object]]) -> str:
    selected: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for record in records:
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in identifiers:
            raise ValueError("prediction records require unique task IDs")
        identifiers.add(task_id)
        selected.append(
            {
                key: record.get(key)
                for key in (
                    "task_id",
                    "status",
                    "valid",
                    "strict_valid",
                    "correct",
                    "predicted",
                    "probs",
                    "schema_error",
                )
            }
        )
    encoded = json.dumps(
        sorted(selected, key=lambda row: str(row["task_id"])),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assess_legacy_public(
    *,
    config: ProjectConfig,
    tiers: Mapping[str, Mapping[str, int]],
    invalid: int,
    severe_failures: int,
    second_process_match: bool,
) -> dict[str, object]:
    if set(tiers) != {"easy", "original", "hard"}:
        raise ValueError("legacy-public tiers differ")
    expected_totals = {"easy": 48, "original": 72, "hard": 111}
    for name, expected_total in expected_totals.items():
        if tiers[name].get("total") != expected_total:
            raise ValueError("legacy-public totals differ")
    correct = sum(tiers[name].get("correct", -1) for name in expected_totals)
    total = sum(expected_totals.values())
    if (
        correct != config.release.legacy_public_correct
        or total != config.release.legacy_public_total
        or tiers["easy"].get("correct")
        != config.release.legacy_public_easy_correct
    ):
        raise ValueError("legacy-public result differs")
    if invalid != 0 or invalid > config.release.maximum_invalid:
        raise ValueError("legacy-public invalid responses exceed the gate")
    if severe_failures != 0 or (
        severe_failures > config.release.maximum_severe_failures
    ):
        raise ValueError("legacy-public severe failures exceed the gate")
    if second_process_match is not True:
        raise ValueError("legacy-public second-process reload differs")
    return {
        "schema_version": 1,
        "passed": True,
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "tiers": {name: dict(tiers[name]) for name in sorted(tiers)},
        "invalid": invalid,
        "severe_failures": severe_failures,
        "second_process_match": second_process_match,
    }


def verify_publication_evidence(
    *,
    config: ProjectConfig,
    model_dir: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    from mentat_sys1.audit.contamination import REGISTERED_BENCHMARK_REVISION

    model_root = Path(model_dir)
    evidence_root = Path(evidence_dir)
    missing = [
        name
        for name in PUBLICATION_EVIDENCE_FILES
        if not (evidence_root / name).is_file()
    ]
    if missing:
        raise ValueError(f"missing publication evidence: {', '.join(missing)}")

    model = verify_migrated_model(
        config=config,
        model_dir=model_root,
        require_calibration=True,
    )
    calibration_path = evidence_root / "calibration-report.json"
    calibration = validate_calibration(
        config=config,
        calibration_path=calibration_path,
    )
    if sha256_file(calibration_path) != sha256_file(model_root / "calibration.json"):
        raise ValueError("publication calibration report differs from model")

    receipt = _load_json_object(evidence_root / "calibrated-model-receipt.json")
    expected_receipt = {
        "schema_version": 1,
        "status": "calibrated_model_assembled",
        "manifest_sha256": sha256_file(model_root / "model-manifest.json"),
        "checksums_sha256": sha256_file(model_root / "SHA256SUMS"),
        "calibration_sha256": calibration["calibration_sha256"],
        "calibration_version": calibration["calibration_version"],
        "temperature": calibration["temperature"],
        "model": model,
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise ValueError(f"calibrated model receipt differs: {key}")

    contamination = _load_json_object(evidence_root / "contamination-report.json")
    if (
        contamination.get("schema_version") != 1
        or contamination.get("status") != "clean"
        or contamination.get("benchmark_revision") != REGISTERED_BENCHMARK_REVISION
        or contamination.get("train_rows") != 87386
        or contamination.get("benchmark_rows") != 231
    ):
        raise ValueError("contamination evidence identity differs")
    matches = contamination.get("matches")
    expected_match_keys = {
        "exact_ids",
        "derivative_ids",
        "source_groups",
        "image_hashes",
        "word_8grams",
    }
    if (
        not isinstance(matches, dict)
        or set(matches) != expected_match_keys
        or any(matches[key] != [] for key in expected_match_keys)
    ):
        raise ValueError("unresolved contamination remains")
    reviewed = contamination.get("resolved_word_8grams")
    if not isinstance(reviewed, list):
        raise ValueError("reviewed contamination evidence is missing")
    reviewed_occurrences = 0
    for item in reviewed:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("ngram_sha256"), str)
            or SHA256_PATTERN.fullmatch(item["ngram_sha256"]) is None
            or not isinstance(item.get("occurrences"), int)
            or item["occurrences"] < 1
            or not isinstance(item.get("review"), str)
            or not item["review"]
        ):
            raise ValueError("reviewed contamination evidence is invalid")
        reviewed_occurrences += item["occurrences"]

    legacy = _load_json_object(evidence_root / "legacy-public-evaluation.json")
    if (
        legacy.get("schema_version") != 1
        or legacy.get("passed") is not True
        or legacy.get("benchmark_revision") != REGISTERED_BENCHMARK_REVISION
        or legacy.get("model_manifest_sha256")
        != sha256_file(model_root / "model-manifest.json")
        or legacy.get("calibration_sha256")
        != sha256_file(model_root / "calibration.json")
    ):
        raise ValueError("legacy-public evidence identity differs")
    tiers = legacy.get("tiers")
    if not isinstance(tiers, dict):
        raise ValueError("legacy-public tiers are missing")
    assessed = assess_legacy_public(
        config=config,
        tiers=tiers,
        invalid=legacy.get("invalid", -1),
        severe_failures=legacy.get("severe_failures", -1),
        second_process_match=legacy.get("second_process_match", False),
    )
    for key in ("correct", "total", "accuracy"):
        if legacy.get(key) != assessed[key]:
            raise ValueError(f"legacy-public evidence differs: {key}")
    digest = legacy.get("prediction_digest")
    runs = legacy.get("runs")
    if (
        not isinstance(digest, str)
        or SHA256_PATTERN.fullmatch(digest) is None
        or not isinstance(runs, dict)
        or set(runs) != {"reload-1", "reload-2"}
    ):
        raise ValueError("legacy-public reload evidence differs")
    for run in runs.values():
        if (
            not isinstance(run, dict)
            or run.get("rows") != config.release.legacy_public_total
            or run.get("prediction_digest") != digest
        ):
            raise ValueError("legacy-public reload evidence differs")

    return {
        "model": model,
        "evidence_sha256": {
            name: sha256_file(evidence_root / name)
            for name in PUBLICATION_EVIDENCE_FILES
        },
        "reviewed_word_8grams": len(reviewed),
        "reviewed_word_8gram_occurrences": reviewed_occurrences,
        "legacy_public": assessed,
    }


def _verify_publication_checksums(
    *,
    bundle_root: Path,
    expected_files: set[str],
) -> None:
    observed: dict[str, str] = {}
    for line in (bundle_root / "SHA256SUMS").read_text(
        encoding="utf-8"
    ).splitlines():
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            raise ValueError("invalid publication SHA256SUMS line")
        digest, name = parts
        path = Path(name)
        if (
            SHA256_PATTERN.fullmatch(digest) is None
            or name in observed
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("invalid publication SHA256SUMS entry")
        observed[name] = digest
    if set(observed) != expected_files:
        raise ValueError("publication SHA256SUMS file set differs")
    for name, digest in observed.items():
        if digest != sha256_file(bundle_root / name):
            raise ValueError(f"publication SHA256SUMS mismatch: {name}")


def verify_publication_bundle(
    *,
    config: ProjectConfig,
    bundle_root: Path,
) -> dict[str, object]:
    root = Path(bundle_root)
    if not root.is_dir():
        raise ValueError(f"publication bundle is not a directory: {root}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("publication bundle must not contain symlinks")

    expected_payload = {
        *(f"model/{name}" for name in CALIBRATED_MODEL_FILES),
        *(f"evidence/{name}" for name in PUBLICATION_EVIDENCE_FILES),
        *PUBLICATION_DOCUMENTS,
    }
    expected_all = expected_payload | set(PUBLICATION_GENERATED_FILES)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != expected_all:
        raise ValueError("publication bundle file set differs")

    evidence = verify_publication_evidence(
        config=config,
        model_dir=root / "model",
        evidence_dir=root / "evidence",
    )
    manifest = _load_json_object(root / "release-manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("release_type") != "publication_bundle"
        or manifest.get("model_id") != config.project_id
        or manifest.get("base_model") != config.base_model.model_dump(mode="json")
        or manifest.get("model") != evidence
    ):
        raise ValueError("publication manifest identity differs")
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != expected_payload:
        raise ValueError("publication manifest file set differs")
    for name, record in records.items():
        path = root / name
        if (
            not isinstance(record, dict)
            or record.get("sha256") != sha256_file(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"publication manifest record differs: {name}")

    _verify_publication_checksums(
        bundle_root=root,
        expected_files=expected_all - {"SHA256SUMS"},
    )
    scan = scan_public_tree(root)
    return {
        "files": len(expected_all),
        "model": evidence["model"],
        "evidence": {
            key: value for key, value in evidence.items() if key != "model"
        },
        "public_scan": scan,
    }
