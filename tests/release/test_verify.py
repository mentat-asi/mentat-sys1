import json
from pathlib import Path

import pytest

from mentat_sys1.audit.provenance import sha256_file
from mentat_sys1.contracts import ProjectConfig
from mentat_sys1.release.package import migrate_checkpoint
from mentat_sys1.release.verify import (
    scan_public_tree,
    verify_migrated_model,
)

ROOT = Path(__file__).parents[2]


def _migrated_fixture(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    source = tmp_path / "source"
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
    (source / "adapter_model.safetensors").write_bytes(b"adapter")
    (source / "decision_readout.json").write_text(
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
    (source / "decision_readout.safetensors").write_bytes(b"readout")
    raw = json.loads(
        (ROOT / "configs/mentat-sys1-v0.1.json").read_text(encoding="utf-8")
    )
    raw["release"]["adapter_sha256"] = sha256_file(source / "adapter_model.safetensors")
    raw["release"]["readout_sha256"] = sha256_file(
        source / "decision_readout.safetensors"
    )
    config = ProjectConfig.model_validate(raw)
    artifact_root = tmp_path / "artifact"
    migrate_checkpoint(
        config=config,
        source_model_dir=source,
        artifact_root=artifact_root,
    )
    return config, artifact_root


def test_migrated_model_verifies_registered_hashes(tmp_path: Path) -> None:
    config, artifact_root = _migrated_fixture(tmp_path)

    receipt = verify_migrated_model(
        config=config,
        model_dir=artifact_root / "model",
    )

    assert receipt["adapter_sha256"] == config.release.adapter_sha256
    assert receipt["readout_sha256"] == config.release.readout_sha256
    assert receipt["readout_codes"] == 256


def test_migrated_model_rejects_tampering(tmp_path: Path) -> None:
    config, artifact_root = _migrated_fixture(tmp_path)
    model = artifact_root / "model"
    (model / "decision_readout.safetensors").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="manifest SHA-256"):
        verify_migrated_model(config=config, model_dir=model)


@pytest.mark.parametrize(
    "payload",
    [
        {"path": "/data/private/model"},
        {"path": "/Users/private/model"},
        {"uri": "file:///tmp/model"},
        {"endpoint": "http://internal.service/model"},
        {"project": "imajev_phase3"},
        {"access_token": "secret-value"},
    ],
)
def test_public_tree_rejects_private_content(
    tmp_path: Path,
    payload: dict[str, str],
) -> None:
    (tmp_path / "release.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden public content"):
        scan_public_tree(tmp_path)


def test_notice_may_record_historical_source_name(tmp_path: Path) -> None:
    (tmp_path / "NOTICE").write_text(
        "Adapted from Imajev at a pinned public revision.\n",
        encoding="utf-8",
    )

    assert scan_public_tree(tmp_path)["files_scanned"] == 1
