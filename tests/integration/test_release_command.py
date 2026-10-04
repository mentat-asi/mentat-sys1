import json
from pathlib import Path

import pytest

from mentat_sys1.audit.provenance import sha256_file
from mentat_sys1.cli import main

ROOT = Path(__file__).parents[2]


def test_release_command_migrates_checkpoint(
    tmp_path: Path,
    capsys: object,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "/data/private/base",
                "revision": None,
                "r": 64,
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

    config = json.loads(
        (ROOT / "configs/mentat-sys1-v0.1.json").read_text(encoding="utf-8")
    )
    config["release"]["adapter_sha256"] = sha256_file(
        source / "adapter_model.safetensors"
    )
    config["release"]["readout_sha256"] = sha256_file(
        source / "decision_readout.safetensors"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_root = tmp_path / "artifact"

    exit_code = main(
        [
            "release",
            "--config",
            str(config_path),
            "--artifact-root",
            str(artifact_root),
            "--source-model-dir",
            str(source),
        ]
    )

    assert exit_code == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "checkpoint_migrated"
    assert (artifact_root / "model/model-manifest.json").is_file()


def test_release_command_builds_publication_bundle(
    tmp_path: Path,
    capsys: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "final-artifact"
    project = tmp_path / "project"
    bundle = tmp_path / "publication"
    observed: dict[str, Path] = {}

    def fake_build_publication_bundle(
        *,
        config: object,
        source_artifact_root: Path,
        project_root: Path,
        bundle_root: Path,
    ) -> dict[str, object]:
        del config
        observed.update(
            {
                "source": source_artifact_root,
                "project": project_root,
                "bundle": bundle_root,
            }
        )
        return {"status": "publication_bundle_assembled"}

    monkeypatch.setattr(
        "mentat_sys1.cli.build_publication_bundle",
        fake_build_publication_bundle,
    )

    exit_code = main(
        [
            "release",
            "--config",
            str(ROOT / "configs/mentat-sys1-v0.1.json"),
            "--artifact-root",
            str(bundle),
            "--publication-source-root",
            str(source),
            "--project-root",
            str(project),
        ]
    )

    assert exit_code == 0
    assert observed == {"source": source, "project": project, "bundle": bundle}
    assert json.loads(capsys.readouterr().out) == {
        "status": "publication_bundle_assembled"
    }
