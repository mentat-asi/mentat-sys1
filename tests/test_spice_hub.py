"""Verify Hub loading boundaries without network or GPU model allocation."""
from pathlib import Path

import pytest

from mentat_spice.hub import BASE_MODEL_ID, BASE_REVISION, MODEL_FILES, resolve_pretrained
from mentat_spice.model import DecisionModel


def make_package(tmp_path):
    package = tmp_path / "package"
    adapter = package / "model"
    adapter.mkdir(parents=True)
    for name in MODEL_FILES:
        (adapter / name).write_bytes(b"test fixture")
    return package


def test_hub_downloads_only_adapter_files_and_pins_backbone(tmp_path, monkeypatch):
    package = make_package(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(base if kwargs["repo_id"] == BASE_MODEL_ID else package)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert resolve_pretrained("owner/model", revision="adapter-sha", cache_dir="cache", local_files_only=True) == (str(base), str(package / "model"))
    assert calls[0]["revision"] == "adapter-sha"
    assert calls[0]["allow_patterns"] == [f"model/{name}" for name in MODEL_FILES]
    assert calls[1]["repo_id"] == BASE_MODEL_ID
    assert calls[1]["revision"] == BASE_REVISION
    assert all(call["local_files_only"] and call["cache_dir"] == "cache" for call in calls)


def test_local_paths_skip_network_and_missing_files_fail_before_base_download(tmp_path, monkeypatch):
    package = make_package(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    def no_network(**kwargs):
        pytest.fail("Local model loading attempted a network download")
    monkeypatch.setattr("huggingface_hub.snapshot_download", no_network)
    assert resolve_pretrained(str(package), base_model=base) == (str(base), str(package / "model"))
    (package / "model/decision_readout.json").unlink()
    with pytest.raises(ValueError, match="decision_readout.json"):
        resolve_pretrained(str(package), base_model=base)


def test_from_pretrained_constructs_the_reference_runtime(tmp_path):
    package = make_package(tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    class FakeModel(DecisionModel):
        def __init__(self, base_model, adapter_directory, *, device):
            self.paths = (base_model, adapter_directory, device)
    model = FakeModel.from_pretrained(str(package), base_model=base, device="cuda:5")
    assert model.paths == (str(base), str(package / "model"), "cuda:5")
    with pytest.raises(ValueError, match="runtime"):
        FakeModel.from_pretrained(str(package), runtime="invalid")
