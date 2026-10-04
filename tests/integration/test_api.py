from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from mentat_sys1.cli import main
from mentat_sys1.inference.api import create_app
from mentat_sys1.inference.backend import PortableBackend
from mentat_sys1.inference.contracts import DecisionRequest
from mentat_sys1.inference.scoring import candidates, result_from_logits

ROOT = Path(__file__).parents[2]


class FakeBackend:
    model_id = "mentat-sys1-v0.1"
    identity = {
        "name": "mentat-sys1-v0.1",
        "base_model": "Qwen/Qwen3.5-4B",
        "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "adapter_sha256": "a" * 64,
        "readout_sha256": "b" * 64,
        "rotations": 4,
        "calibration": "frozen",
    }

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[DecisionRequest] = []

    def score(
        self,
        images: list[Any],
        request: DecisionRequest,
    ) -> tuple[list[Any], dict[str, object]]:
        if self.fail:
            raise RuntimeError("/data/private/model failed")
        self.requests.append(request)
        results = []
        for field in request.fields:
            field_candidates = candidates(field)
            logits = [0.0] * len(field_candidates)
            logits[0] = 4.0
            results.append(result_from_logits(field_candidates, logits))
        return results, {
            "input_tokens": 12,
            "rotations": 4,
            "images": len(images),
        }


def _request() -> dict[str, object]:
    return {
        "model": "mentat-sys1-v0.1",
        "state": "A precedes B.",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Which comes first?",
                "criteria": {"a": "A", "b": "B"},
            }
        },
    }


def test_service_identity_and_choice_contract() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(backend))

    assert client.get("/health").json() == {
        "status": "ready",
        "model": "mentat-sys1-v0.1",
    }
    models = client.get("/v1/models").json()["models"]
    assert models == [backend.identity]
    response = client.post("/v1/systemone", json=_request())

    assert response.status_code == 200
    assert response.json()["model"] == "mentat-sys1-v0.1"
    assert response.json()["answers"]["q"]["choice"] == "a"
    assert response.json()["usage"]["rotations"] == 4
    assert len(backend.requests) == 1


def test_oversized_or_malformed_request_is_typed_4xx() -> None:
    client = TestClient(create_app(FakeBackend(), maximum_request_bytes=128))
    oversized = client.post("/v1/systemone", json={"state": "x" * 512})
    assert oversized.status_code == 413
    assert set(oversized.json()) == {"code", "message"}

    client = TestClient(create_app(FakeBackend()))
    mismatch = client.post(
        "/v1/systemone",
        json={**_request(), "model": "another-model"},
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["code"] == "invalid_request"


def test_backend_failure_does_not_expose_exception_or_local_path() -> None:
    client = TestClient(
        create_app(FakeBackend(fail=True)),
        raise_server_exceptions=False,
    )

    response = client.post("/v1/systemone", json=_request())

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "Model scoring failed.",
    }
    assert "/data/" not in response.text


def test_serve_command_loads_final_model_before_starting_uvicorn(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    observed: dict[str, object] = {}

    def fake_load(cls: type[PortableBackend], **kwargs: object) -> FakeBackend:
        del cls
        observed["load"] = kwargs
        return backend

    def fake_create_app(loaded: FakeBackend) -> object:
        assert loaded is backend
        return "app"

    def fake_run(app: object, *, host: str, port: int) -> None:
        observed["uvicorn"] = {"app": app, "host": host, "port": port}

    monkeypatch.setattr(PortableBackend, "load", classmethod(fake_load))
    monkeypatch.setattr(
        "mentat_sys1.inference.api.create_app",
        fake_create_app,
    )
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))

    result = main(
        [
            "serve",
            "--config",
            str(ROOT / "configs/mentat-sys1-v0.1.json"),
            "--artifact-root",
            str(tmp_path),
            "--base-model-path",
            str(tmp_path / "base"),
            "--model-dir",
            str(tmp_path / "final/model"),
            "--device",
            "cuda:0",
            "--host",
            "127.0.0.1",
            "--port",
            "28101",
        ]
    )

    assert result == 0
    load = observed["load"]
    assert isinstance(load, dict)
    assert load["config"].project_id == "mentat-sys1-v0.1"
    assert {key: value for key, value in load.items() if key != "config"} == {
        "model_dir": tmp_path / "final/model",
        "base_model_path": tmp_path / "base",
        "device": "cuda:0",
    }
    assert observed["uvicorn"] == {
        "app": "app",
        "host": "127.0.0.1",
        "port": 28101,
    }
