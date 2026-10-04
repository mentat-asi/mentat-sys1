import json
from pathlib import Path

import pytest

from mentat_sys1.audit.provenance import sha256_file
from mentat_sys1.contracts import ProjectConfig
from mentat_sys1.release.package import (
    SOURCE_MODEL_FILES,
    assemble_calibrated_model,
    migrate_checkpoint,
)
from mentat_sys1.release.verify import (
    CALIBRATION_SHA256,
    verify_migrated_model,
)

ROOT = Path(__file__).parents[2]


def _source_checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "/data/private/Qwen3.5-4B",
                "revision": None,
                "r": 64,
                "lora_alpha": 128,
                "target_modules": ["q_proj", "v_proj"],
            }
        ),
        encoding="utf-8",
    )
    (root / "adapter_model.safetensors").write_bytes(b"adapter-tensors")
    (root / "decision_readout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "codes": [
                    {"code": f"C{index}", "token_id": index} for index in range(256)
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "decision_readout.safetensors").write_bytes(b"readout-tensors")
    return root


def _config_for(source: Path) -> ProjectConfig:
    raw = json.loads(
        (ROOT / "configs/mentat-sys1-v0.1.json").read_text(encoding="utf-8")
    )
    raw["release"]["adapter_sha256"] = sha256_file(source / "adapter_model.safetensors")
    raw["release"]["readout_sha256"] = sha256_file(
        source / "decision_readout.safetensors"
    )
    return ProjectConfig.model_validate(raw)


def _calibration_report(config: ProjectConfig) -> dict[str, object]:
    return {
        "schema_version": 1,
        "method": "scalar_temperature",
        "temperature": 0.97,
        "optimizer": "bounded_log_temperature",
        "objective": "negative_log_likelihood",
        "split": {"rows": 1120, "sha256": CALIBRATION_SHA256},
        "model": {
            "adapter_sha256": config.release.adapter_sha256,
            "readout_sha256": config.release.readout_sha256,
        },
        "before": {"ece": 0.03, "nll": 0.4, "brier": 0.2},
        "after": {"ece": 0.02, "nll": 0.39, "brier": 0.19},
        "predictions_unchanged": True,
        "scoring": {"rotation_mode": "rot4", "model_passes": 4204},
    }


def test_checkpoint_migration_rewrites_only_portable_metadata(
    tmp_path: Path,
) -> None:
    source = _source_checkpoint(tmp_path / "source")
    original_files = {
        path.name: path.read_bytes() for path in source.iterdir() if path.is_file()
    }
    config = _config_for(source)

    receipt = migrate_checkpoint(
        config=config,
        source_model_dir=source,
        artifact_root=tmp_path / "artifact",
    )

    model = tmp_path / "artifact" / "model"
    portable_config = json.loads(
        (model / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert portable_config["base_model_name_or_path"] == "Qwen/Qwen3.5-4B"
    assert portable_config["revision"] == config.base_model.revision
    assert portable_config["target_modules"] == ["q_proj", "v_proj"]
    assert (model / "adapter_model.safetensors").read_bytes() == (
        original_files["adapter_model.safetensors"]
    )
    assert (model / "decision_readout.json").read_bytes() == (
        original_files["decision_readout.json"]
    )
    assert (model / "decision_readout.safetensors").read_bytes() == (
        original_files["decision_readout.safetensors"]
    )
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original_files
    assert (
        receipt["source"]["adapter_config_sha256"]
        != (receipt["portable"]["adapter_config_sha256"])
    )
    assert (
        receipt["source"]["adapter_sha256"] == (receipt["portable"]["adapter_sha256"])
    )
    assert receipt["portable"]["readout_sha256"] == (config.release.readout_sha256)
    assert (tmp_path / "artifact/evidence/migration-receipt.json").is_file()
    assert (model / "model-manifest.json").is_file()
    assert (model / "SHA256SUMS").is_file()


def test_checkpoint_migration_rejects_unregistered_tensors(
    tmp_path: Path,
) -> None:
    source = _source_checkpoint(tmp_path / "source")
    config = _config_for(source)
    (source / "adapter_model.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="adapter tensor SHA-256"):
        migrate_checkpoint(
            config=config,
            source_model_dir=source,
            artifact_root=tmp_path / "artifact",
        )

    assert not (tmp_path / "artifact/model").exists()


def test_calibrated_model_assembly_preserves_runtime_files_and_binds_report(
    tmp_path: Path,
) -> None:
    source = _source_checkpoint(tmp_path / "source")
    config = _config_for(source)
    staging_root = tmp_path / "staging"
    migrate_checkpoint(
        config=config,
        source_model_dir=source,
        artifact_root=staging_root,
    )
    calibration_report = staging_root / "evidence/calibration-report.json"
    calibration_report.write_text(
        json.dumps(_calibration_report(config), sort_keys=True),
        encoding="utf-8",
    )
    before = {
        name: (staging_root / "model" / name).read_bytes()
        for name in SOURCE_MODEL_FILES
    }

    receipt = assemble_calibrated_model(
        config=config,
        source_model_dir=staging_root / "model",
        calibration_report=calibration_report,
        artifact_root=tmp_path / "final",
    )

    final_model = tmp_path / "final/model"
    assert {
        name: (final_model / name).read_bytes() for name in SOURCE_MODEL_FILES
    } == before
    assert (final_model / "calibration.json").read_bytes() == (
        calibration_report.read_bytes()
    )
    assert (tmp_path / "final/evidence/calibration-report.json").read_bytes() == (
        calibration_report.read_bytes()
    )
    verified = verify_migrated_model(
        config=config,
        model_dir=final_model,
        require_calibration=True,
    )
    assert verified["temperature"] == pytest.approx(0.97)
    assert receipt["status"] == "calibrated_model_assembled"
    assert receipt["calibration_sha256"] == sha256_file(
        final_model / "calibration.json"
    )
