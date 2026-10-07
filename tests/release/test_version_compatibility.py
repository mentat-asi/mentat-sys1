from __future__ import annotations

import json
from pathlib import Path

import pytest

from mentat_sys1.audit.provenance import sha256_file
from mentat_sys1.contracts import ProjectConfig, load_config
from mentat_sys1.release.package import (
    assemble_calibrated_model,
    migrate_checkpoint,
)
from mentat_sys1.release.verify import verify_migrated_model

ROOT = Path(__file__).parents[2]


def _source_model(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "/data/private/Qwen3.5-4B",
                "revision": None,
                "r": 64,
                "lora_alpha": 128,
            }
        ),
        encoding="utf-8",
    )
    (source / "adapter_model.safetensors").write_bytes(b"v02-adapter")
    (source / "decision_readout.json").write_text(
        json.dumps(
            {
                "version": 1,
                "codes": [
                    {"code": f"C{index}", "token_id": index}
                    for index in range(256)
                ],
            }
        ),
        encoding="utf-8",
    )
    (source / "decision_readout.safetensors").write_bytes(b"v02-readout")
    return source


def _v02_config(source: Path) -> ProjectConfig:
    payload = json.loads(
        (ROOT / "configs/mentat-sys1-v0.2.json").read_text(encoding="utf-8")
    )
    payload["release"]["adapter_sha256"] = sha256_file(
        source / "adapter_model.safetensors"
    )
    payload["release"]["readout_sha256"] = sha256_file(
        source / "decision_readout.safetensors"
    )
    return ProjectConfig.model_validate(payload)


def _write_v02_calibration(path: Path, config: ProjectConfig) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "method": "temperature_by_type",
                "pooled_temperature": 1.5758800892767102,
                "temperature_by_type": {
                    "choice": 1.609674650339169,
                    "noul": 1.3462360767794004,
                    "score": 1.6476480715733943,
                },
                "split": {"rows": 1250, "sha256": "a" * 64},
                "model": {
                    "adapter_sha256": config.release.adapter_sha256,
                    "readout_sha256": config.release.readout_sha256,
                },
                "before": {"nll": 1.0416192707769858},
                "after": {"nll": 0.9668147145225859},
                "type_fits": {
                    "choice": {
                        "rows": 375,
                        "source": "shrinkage",
                        "temperature": 1.609674650339169,
                    },
                    "noul": {
                        "rows": 375,
                        "source": "shrinkage",
                        "temperature": 1.3462360767794004,
                    },
                    "score": {
                        "rows": 500,
                        "source": "shrinkage",
                        "temperature": 1.6476480715733943,
                    },
                },
                "prediction_digest": {
                    "before": "b" * 64,
                    "after": "b" * 64,
                },
                "predictions_unchanged": True,
            }
        ),
        encoding="utf-8",
    )


def test_v02_calibrated_model_round_trips_and_rejects_v01_config(
    tmp_path: Path,
) -> None:
    source = _source_model(tmp_path)
    config = _v02_config(source)
    migrated = tmp_path / "migrated"
    migrate_checkpoint(
        config=config,
        source_model_dir=source,
        artifact_root=migrated,
    )
    calibration = tmp_path / "calibration.json"
    _write_v02_calibration(calibration, config)
    final = tmp_path / "final"

    receipt = assemble_calibrated_model(
        config=config,
        source_model_dir=migrated / "model",
        calibration_report=calibration,
        artifact_root=final,
    )
    verified = verify_migrated_model(
        config=config,
        model_dir=final / "model",
        require_calibration=True,
    )

    assert receipt["calibration_version"].startswith("temperature_by_type-v2:")
    assert verified["temperature"] == pytest.approx(1.5758800892767102)
    assert verified["temperature_by_type"] == {
        "choice": pytest.approx(1.609674650339169),
        "noul": pytest.approx(1.3462360767794004),
        "score": pytest.approx(1.6476480715733943),
    }
    with pytest.raises(ValueError, match="model manifest identity differs"):
        verify_migrated_model(
            config=load_config(ROOT / "configs/mentat-sys1-v0.1.json"),
            model_dir=final / "model",
            require_calibration=True,
        )
