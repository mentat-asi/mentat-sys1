import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mentat_sys1.contracts import ProjectConfig, load_config

ROOT = Path(__file__).parents[2]


def _raw_config() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs/mentat-sys1-v0.1.json").read_text(encoding="utf-8")
    )


def test_project_config_freezes_registered_identity_and_runtime() -> None:
    config = ProjectConfig.model_validate(_raw_config())

    assert config.project_id == "mentat-sys1-v0.1"
    assert config.base_model.repository == "Qwen/Qwen3.5-4B"
    assert config.base_model.revision == ("851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    assert config.runtime.readout_codes == 256
    assert config.runtime.max_length == 16384
    assert config.release.adapter_sha256 == (
        "c72687243aaa8c38f984091247619c242f3f3443a1f3d35c9c3526c4dfde2d83"
    )

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(
            {
                **_raw_config(),
                "runtime": {
                    **_raw_config()["runtime"],
                    "max_length": 8192,
                },
            }
        )


def test_load_config_rejects_absolute_artifact_paths(tmp_path: Path) -> None:
    payload = _raw_config()
    payload["paths"] = {
        **payload["paths"],
        "model": "/data/private/model",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="relative"):
        load_config(path)


def test_project_config_is_frozen() -> None:
    config = ProjectConfig.model_validate(_raw_config())

    with pytest.raises(ValidationError):
        config.project_id = "changed"
