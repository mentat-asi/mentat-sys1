"""Portable, byte-preserving checkpoint migration."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from mentat_sys1.audit.provenance import sha256_file, write_immutable_json
from mentat_sys1.contracts import ProjectConfig
from mentat_sys1.release.verify import (
    CALIBRATED_MODEL_FILES,
    PUBLICATION_DOCUMENTS,
    PUBLICATION_EVIDENCE_FILES,
    verify_publication_bundle,
    verify_publication_evidence,
)

SOURCE_MODEL_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "decision_readout.json",
    "decision_readout.safetensors",
)
PUBLICATION_MODEL_FILES = CALIBRATED_MODEL_FILES


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _validate_readout(path: Path, expected_codes: int) -> dict[str, int]:
    payload = _load_json_object(path)
    if payload.get("version") != 1:
        raise ValueError("decision readout schema version must be 1")
    codes = payload.get("codes")
    if not isinstance(codes, list) or len(codes) != expected_codes:
        raise ValueError(f"decision readout must contain {expected_codes} codes")
    names: set[str] = set()
    token_ids: set[int] = set()
    for item in codes:
        if not isinstance(item, dict):
            raise ValueError("decision readout code must be an object")
        name = item.get("code")
        token_id = item.get("token_id")
        if not isinstance(name, str) or not name:
            raise ValueError("decision readout code name must be non-empty")
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError("decision readout token ID must be non-negative")
        if name in names or token_id in token_ids:
            raise ValueError("decision readout codes and token IDs must be unique")
        names.add(name)
        token_ids.add(token_id)
    return {"version": 1, "codes": len(codes)}


def _copy_immutable(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"conflicting immutable file: {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _write_immutable_text(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"conflicting immutable text: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _file_record(path: Path) -> dict[str, object]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def migrate_checkpoint(
    *,
    config: ProjectConfig,
    source_model_dir: Path,
    artifact_root: Path,
) -> dict[str, object]:
    source = Path(source_model_dir)
    missing = [name for name in SOURCE_MODEL_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"missing source model file: {', '.join(missing)}")

    source_adapter_sha = sha256_file(source / "adapter_model.safetensors")
    source_readout_sha = sha256_file(source / "decision_readout.safetensors")
    if source_adapter_sha != config.release.adapter_sha256:
        raise ValueError("adapter tensor SHA-256 differs from the registered model")
    if source_readout_sha != config.release.readout_sha256:
        raise ValueError("readout tensor SHA-256 differs from the registered model")
    readout = _validate_readout(
        source / "decision_readout.json",
        config.runtime.readout_codes,
    )

    source_adapter_config = _load_json_object(source / "adapter_config.json")
    portable_adapter_config = dict(source_adapter_config)
    portable_adapter_config["base_model_name_or_path"] = config.base_model.repository
    portable_adapter_config["revision"] = config.base_model.revision

    model_dir = Path(artifact_root) / config.paths.model
    evidence_dir = Path(artifact_root) / config.paths.evidence
    _copy_immutable(
        source / "adapter_model.safetensors",
        model_dir / "adapter_model.safetensors",
    )
    _copy_immutable(
        source / "decision_readout.json",
        model_dir / "decision_readout.json",
    )
    _copy_immutable(
        source / "decision_readout.safetensors",
        model_dir / "decision_readout.safetensors",
    )
    write_immutable_json(model_dir / "adapter_config.json", portable_adapter_config)

    runtime_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "decision_readout.json",
        "decision_readout.safetensors",
    )
    manifest = {
        "schema_version": 1,
        "model_id": config.project_id,
        "base_model": config.base_model.model_dump(mode="json"),
        "adapter_config": {
            "source_sha256": sha256_file(source / "adapter_config.json"),
            "portable_sha256": sha256_file(model_dir / "adapter_config.json"),
        },
        "readout": readout,
        "files": {name: _file_record(model_dir / name) for name in runtime_files},
    }
    write_immutable_json(model_dir / "model-manifest.json", manifest)

    checksummed_files = (*runtime_files, "model-manifest.json")
    checksums = "".join(
        f"{sha256_file(model_dir / name)}  {name}\n"
        for name in sorted(checksummed_files)
    )
    _write_immutable_text(model_dir / "SHA256SUMS", checksums)

    receipt = {
        "schema_version": 1,
        "status": "checkpoint_migrated",
        "source": {
            "adapter_config_sha256": sha256_file(source / "adapter_config.json"),
            "adapter_sha256": source_adapter_sha,
            "readout_json_sha256": sha256_file(source / "decision_readout.json"),
            "readout_sha256": source_readout_sha,
        },
        "portable": {
            "adapter_config_sha256": sha256_file(model_dir / "adapter_config.json"),
            "adapter_sha256": sha256_file(model_dir / "adapter_model.safetensors"),
            "readout_json_sha256": sha256_file(model_dir / "decision_readout.json"),
            "readout_sha256": sha256_file(model_dir / "decision_readout.safetensors"),
            "manifest_sha256": sha256_file(model_dir / "model-manifest.json"),
            "checksums_sha256": sha256_file(model_dir / "SHA256SUMS"),
        },
    }
    write_immutable_json(evidence_dir / "migration-receipt.json", receipt)
    return receipt


def assemble_calibrated_model(
    *,
    config: ProjectConfig,
    source_model_dir: Path,
    calibration_report: Path,
    artifact_root: Path,
) -> dict[str, object]:
    from mentat_sys1.release.verify import (
        validate_calibration,
        verify_migrated_model,
    )

    source = Path(source_model_dir)
    report_path = Path(calibration_report)
    verify_migrated_model(
        config=config,
        model_dir=source,
        require_calibration=False,
    )
    calibration = validate_calibration(
        config=config,
        calibration_path=report_path,
    )
    calibration_payload = _load_json_object(report_path)

    model_dir = Path(artifact_root) / config.paths.model
    evidence_dir = Path(artifact_root) / config.paths.evidence
    if source.resolve() == model_dir.resolve():
        raise ValueError("calibrated model must use a new artifact root")

    for name in SOURCE_MODEL_FILES:
        _copy_immutable(source / name, model_dir / name)
    _copy_immutable(report_path, model_dir / "calibration.json")
    _copy_immutable(report_path, evidence_dir / "calibration-report.json")

    source_manifest = _load_json_object(source / "model-manifest.json")
    runtime_files = (*SOURCE_MODEL_FILES, "calibration.json")
    calibration_manifest = {
        "method": calibration_payload["method"],
        "calibration_sha256": calibration["calibration_sha256"],
        "calibration_version": calibration["calibration_version"],
    }
    if calibration_payload["schema_version"] == 1:
        calibration_manifest["temperature"] = calibration["temperature"]
    else:
        calibration_manifest["schema_version"] = 2
        calibration_manifest["pooled_temperature"] = calibration["temperature"]
        calibration_manifest["temperature_by_type"] = calibration[
            "temperature_by_type"
        ]
    manifest = {
        "schema_version": 1,
        "model_id": config.project_id,
        "base_model": config.base_model.model_dump(mode="json"),
        "adapter_config": source_manifest["adapter_config"],
        "readout": source_manifest["readout"],
        "calibration": calibration_manifest,
        "files": {name: _file_record(model_dir / name) for name in runtime_files},
    }
    write_immutable_json(model_dir / "model-manifest.json", manifest)

    checksummed_files = (*runtime_files, "model-manifest.json")
    checksums = "".join(
        f"{sha256_file(model_dir / name)}  {name}\n"
        for name in sorted(checksummed_files)
    )
    _write_immutable_text(model_dir / "SHA256SUMS", checksums)
    verified = verify_migrated_model(
        config=config,
        model_dir=model_dir,
        require_calibration=True,
    )

    receipt = {
        "schema_version": 1,
        "status": "calibrated_model_assembled",
        "source_manifest_sha256": sha256_file(source / "model-manifest.json"),
        "manifest_sha256": sha256_file(model_dir / "model-manifest.json"),
        "checksums_sha256": sha256_file(model_dir / "SHA256SUMS"),
        "calibration_sha256": calibration["calibration_sha256"],
        "calibration_version": calibration["calibration_version"],
        "temperature": calibration["temperature"],
        "model": verified,
    }
    if "temperature_by_type" in calibration:
        receipt["temperature_by_type"] = calibration["temperature_by_type"]
    write_immutable_json(
        evidence_dir / "calibrated-model-receipt.json",
        receipt,
    )
    return receipt


def build_publication_bundle(
    *,
    config: ProjectConfig,
    source_artifact_root: Path,
    project_root: Path,
    bundle_root: Path,
) -> dict[str, object]:
    source = Path(source_artifact_root)
    project = Path(project_root)
    destination = Path(bundle_root)
    if destination.exists():
        raise ValueError(f"publication bundle already exists: {destination}")

    source_model = source / config.paths.model
    source_evidence = source / config.paths.evidence
    model = verify_publication_evidence(
        config=config,
        model_dir=source_model,
        evidence_dir=source_evidence,
    )
    missing_documents = [
        source_name
        for source_name in PUBLICATION_DOCUMENTS.values()
        if not (project / source_name).is_file()
    ]
    if missing_documents:
        raise ValueError(
            "missing publication document: " + ", ".join(missing_documents)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        for name in PUBLICATION_MODEL_FILES:
            _copy_immutable(source_model / name, temporary / "model" / name)
        for name in PUBLICATION_EVIDENCE_FILES:
            _copy_immutable(source_evidence / name, temporary / "evidence" / name)
        for output_name, source_name in PUBLICATION_DOCUMENTS.items():
            _copy_immutable(project / source_name, temporary / output_name)

        payload_files = sorted(
            path
            for path in temporary.rglob("*")
            if path.is_file()
        )
        manifest = {
            "schema_version": 1,
            "release_type": "publication_bundle",
            "model_id": config.project_id,
            "base_model": config.base_model.model_dump(mode="json"),
            "model": model,
            "files": {
                path.relative_to(temporary).as_posix(): _file_record(path)
                for path in payload_files
            },
        }
        write_immutable_json(temporary / "release-manifest.json", manifest)
        checksummed_files = sorted(
            path
            for path in temporary.rglob("*")
            if path.is_file() and path != temporary / "SHA256SUMS"
        )
        checksums = "".join(
            f"{sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
            for path in checksummed_files
        )
        _write_immutable_text(temporary / "SHA256SUMS", checksums)
        verified = verify_publication_bundle(
            config=config,
            bundle_root=temporary,
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "schema_version": 1,
        "status": "publication_bundle_assembled",
        "manifest_sha256": sha256_file(destination / "release-manifest.json"),
        "checksums_sha256": sha256_file(destination / "SHA256SUMS"),
        **verified,
    }
