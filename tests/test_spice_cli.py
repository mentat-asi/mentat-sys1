"""CPU checks of the new HTTP/CLI boundary, without loading model weights."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mentat_spice import cli


class RecordingModel:
    model_id = "mentat-Spice-4B"

    def __init__(self):
        self.calls = []

    def predict(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        if payload.get("invalid"):
            raise ValueError("Invalid decision contract")
        return {"model": self.model_id, "answers": {"ok": True}}


def test_http_passes_request_once_and_reports_contract_errors():
    model = RecordingModel()
    client = TestClient(cli.create_app(model))
    payload = {"state": "hello", "questions": {"q": {"type": "noul"}}}
    assert client.get("/health").json()["model"] == model.model_id
    assert client.post("/v1/systemone", json=payload).json()["answers"] == {"ok": True}
    assert model.calls == [(payload, {})]
    assert client.post("/v1/systemone", json={"invalid": True}).status_code == 422


@pytest.mark.parametrize("body", [b"{", b"null", b"[]", b'"text"', b"\xff"])
def test_http_rejects_invalid_json_without_prediction(body):
    model = RecordingModel()
    client = TestClient(cli.create_app(model))
    assert client.post("/v1/systemone", content=body).status_code == 422
    assert not model.calls


def test_http_rejects_local_image_paths_and_oversize_body():
    model = RecordingModel()
    client = TestClient(cli.create_app(model))
    assert client.post("/v1/systemone", json={"images": ["/tmp/private.png"]}).status_code == 422
    assert client.post("/v1/systemone", content=b"x" * 1_048_577).status_code == 413
    assert not model.calls


def test_predict_preserves_embedded_images_unless_cli_images_given(tmp_path, monkeypatch, capsys):
    model = RecordingModel()
    monkeypatch.setattr(cli, "load_model", lambda args: model)
    payload = {"state": "test", "images": ["image.png"], "questions": {}}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload))
    cli.main(["predict", "--base-model", "base", "--model-package", "adapter", "--input", str(path)])
    assert model.calls == [(payload, {})]
    assert json.loads(capsys.readouterr().out)["model"] == model.model_id


@pytest.mark.parametrize("mode,module,class_name", [
    ("native", "model", "DecisionModel"),
    ("accelerated", "accelerated", "AcceleratedDecisionModel"),
    ("turbo", "turbo", "TurboDecisionModel"),
])
def test_runtime_dispatch_does_not_load_another_mode(mode, module, class_name, monkeypatch):
    import sys
    observed = []
    def constructor(*args, **kwargs):
        observed.append((args, kwargs))
        return "loaded"
    monkeypatch.setitem(sys.modules, f"mentat_spice.{module}", SimpleNamespace(**{class_name: constructor}))
    args = SimpleNamespace(runtime=mode, base_model="base", model_package="package", device="cuda:2")
    assert cli.load_model(args) == "loaded"
    assert observed == [(("base", "package/model"), {"device": "cuda:2"})]


def test_demo_automatically_resolves_hf_model(monkeypatch, capsys):
    from mentat_spice.model import DecisionModel
    model = RecordingModel()
    loads = []
    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return model
    monkeypatch.setattr(DecisionModel, "from_pretrained", load)
    cli.main(["predict", "--demo", "--offline", "--cache-dir", "cache"])
    assert model.calls == [(cli.DEMO_REQUEST, {})]
    assert loads[0][0] == ("yunqu/mentat-Spice-4B",)
    assert loads[0][1]["runtime"] == "native"
    assert loads[0][1]["local_files_only"] is True
    assert loads[0][1]["cache_dir"] == "cache"
    assert json.loads(capsys.readouterr().out)["answers"] == {"ok": True}


def test_invalid_input_fails_before_loading_weights(tmp_path, monkeypatch):
    path = tmp_path / "invalid.json"
    path.write_text("[]")
    def unexpected(args):
        pytest.fail("Invalid request loaded model weights")
    monkeypatch.setattr(cli, "load_model", unexpected)
    with pytest.raises(SystemExit):
        cli.main(["predict", "--input", str(path)])
