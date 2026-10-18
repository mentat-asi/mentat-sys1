import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from mentat_sys1.v03 import loader
from mentat_sys1.v03.api import create_app


def test_package_rejects_tampering_before_import(tmp_path, monkeypatch):
    files = {
        "model/adapter_model.safetensors": b"adapter",
        "model/decision_readout.safetensors": b"readout",
        "infer.py": b"raise RuntimeError('must not execute during verification')",
        "inference_config.json": b"{}",
        "runtime/src/mentat_jev_v1/inference/backend.py": b"# inference",
    }
    digests = {}
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digests[name] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        loader, "ADAPTER_SHA256", digests["model/adapter_model.safetensors"]
    )
    monkeypatch.setattr(
        loader, "READOUT_SHA256", digests["model/decision_readout.safetensors"]
    )
    (tmp_path / "release-manifest.json").write_text(
        json.dumps({"model_id": loader.MODEL_ID})
    )
    (tmp_path / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in digests.items())
    )
    loader.verify_package(tmp_path)
    (tmp_path / "infer.py").write_text("# changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        loader.verify_package(tmp_path)


def test_serving_serializes_graph_buffers_and_validates_identity():
    class Impl:
        active = 0
        maximum = 0
        lock = threading.Lock()

        def predict(self, payload):
            with self.lock:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            time.sleep(0.005)
            with self.lock:
                self.active -= 1
            return {"model": loader.MODEL_ID, "answers": payload.get("questions", {})}

    impl = Impl()
    model = loader.Model(impl, "accelerated")
    payload = {"model": loader.MODEL_ID, "questions": {"route": {"type": "choice"}}}
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(model.predict, [payload] * 16))
    assert impl.maximum == 1
    assert all(result["answers"] == payload["questions"] for result in results)
    with TestClient(create_app(model)) as client:
        assert client.get("/health").json()["runtime"] == "accelerated"
        assert client.post("/v1/systemone", json=payload).status_code == 200
        assert client.post("/v1/systemone", json={"model": "other"}).status_code == 422


def test_graph_native_process_isolation(monkeypatch):
    monkeypatch.setattr(loader, "_accelerated_process", True)
    model = loader.Model(object(), "native")
    with pytest.raises(RuntimeError, match="separate processes"):
        model.predict({})


def test_prefix_cut_requires_exact_tokens_and_remaining_suffix():
    from mentat_sys1.v03.runtime import PrefixOptimized

    class Ids:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    def item(values):
        return {"input_ids": Ids(values), "tokens": len(values)}

    model = object.__new__(PrefixOptimized)
    model.share_prefix = True
    assert model.cut([item([1] * 200 + [2, 3]), item([1] * 200 + [4, 5])]) == 192
    assert model.cut([item([1] * 127 + [2]), item([1] * 127 + [3])]) == 0
    assert model.cut([item([1] * 128), item([1] * 128 + [2])]) == 0
    model.share_prefix = False
    assert model.cut([item([1] * 200 + [2, 3]), item([1] * 200 + [4, 5])]) == 0
