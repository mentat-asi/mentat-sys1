"""Load the frozen public model package without machine-specific paths."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import Any

MODEL_ID = "mentat-sys1-v0.3"
ADAPTER_SHA256 = "a42f0690754634a0cb796fe5e82b87d6f55eede438a4409d9d3af3b2bc8d75c4"
READOUT_SHA256 = "057da9f1ef8f5e57be9710b8fbfb3c541152f93f310cbdcc08fe6b778f3e90be"
_accelerated_process = False


def verify_package(package: Path) -> dict[str, Any]:
    """Verify every packaged file before importing the supplied inference code."""
    package = package.resolve()
    manifest = json.loads((package / "release-manifest.json").read_text())
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("Expected the mentat-sys1-v0.3 model package")
    sums = {}
    for line in (package / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = package / name
        if not path.resolve().is_relative_to(package) or path.is_symlink():
            raise ValueError("Invalid model package path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Model package checksum mismatch: {name}")
        sums[name] = digest
    for name, expected in {
        "model/adapter_model.safetensors": ADAPTER_SHA256,
        "model/decision_readout.safetensors": READOUT_SHA256,
    }.items():
        if sums.get(name) != expected:
            raise ValueError(f"Incorrect v0.3 model weights: {name}")
    for name in [
        "infer.py",
        "inference_config.json",
        "runtime/src/mentat_jev_v1/inference/backend.py",
    ]:
        if name not in sums:
            raise ValueError(f"Missing checksummed inference file: {name}")
    return manifest


class Model:
    """Serialize predictions that reuse CUDA graph buffers."""

    def __init__(self, impl: Any, runtime: str):
        self.impl = impl
        self.runtime = runtime
        self.lock = threading.RLock()

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("model", MODEL_ID) != MODEL_ID:
            raise ValueError(f"Expected model {MODEL_ID}")
        if self.runtime == "native" and _accelerated_process:
            raise RuntimeError(
                "Run native and accelerated models in separate processes"
            )
        with self.lock:
            return self.impl.predict(payload)

    def warm(self, payloads: list[dict[str, Any]]) -> None:
        for payload in payloads:
            self.predict(payload)


def load_model(
    *,
    model_package: str | Path,
    base_model: str | Path,
    runtime: str = "accelerated",
    device: str = "cuda:0",
    threads: int = 4,
    cuda_graphs: bool = True,
    max_graphs: int = 8,
    max_graph_tokens: int = 4096,
) -> Model:
    """Use the downloaded model package and the separately downloaded Qwen base.

    The accelerated implementation installs process-local Qwen kernel hooks.
    Use a separate worker process for each runtime mode.
    """
    global _accelerated_process
    if runtime not in {"native", "accelerated"}:
        raise ValueError("runtime must be native or accelerated")
    if threads < 1 or max_graphs < 0 or max_graph_tokens < 1:
        raise ValueError("Invalid thread or graph limits")
    if runtime == "native" and _accelerated_process:
        raise RuntimeError("Run native and accelerated models in separate processes")
    package = Path(model_package).expanduser().resolve()
    verify_package(package)
    import torch

    torch.set_num_threads(threads)
    if runtime == "accelerated" and not device.startswith("cuda"):
        raise ValueError("The accelerated runtime requires an NVIDIA CUDA GPU")
    spec = importlib.util.spec_from_file_location(
        "mentat_public_v03", package / "infer.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the model package inference entry point")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    native = module.Model(str(base_model), device)
    if runtime == "native":
        return Model(native, runtime)
    # Fail explicitly when optional kernels are missing; never advertise a silent
    # native fallback as the measured accelerated implementation.
    try:
        import fla  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            'Install the accelerated dependencies: pip install ".[accelerate]"'
        ) from error
    from .fusion import install
    from .runtime import PrefixOptimized

    merged = native.engine.model.merge_and_unload(safe_merge=True)
    native.engine.model = merged
    merged.get_base_model = lambda: merged
    install()
    _accelerated_process = True
    impl = PrefixOptimized(
        native,
        mode="graph" if cuda_graphs else "batch",
        fast_kernels=True,
        fast_prepare=True,
        share_prefix=True,
    )
    impl.max_graphs = max_graphs
    impl.max_graph_tokens = max_graph_tokens
    return Model(impl, runtime)
