from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mentat_sys1.inference.backend import (
    PortableBackend,
    configure_torch_numerics,
    verify_model_directory,
)
from mentat_sys1.inference.contracts import DecisionRequest


class FakeEngine:
    load_seconds = 0.25

    def __init__(self) -> None:
        self.calls: list[tuple[list[object], str, list[str]]] = []
        self._logits = [
            [4.0, 1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0, 4.0],
            [0.0, -1.0, 4.0, 1.0],
            [-1.0, 4.0, 1.0, 0.0],
        ]

    def labels(self, count: int, n_images: int) -> list[str]:
        assert count == 4
        assert n_images == 0
        return ["A", "B", "C", "D"]

    def score_prompt(
        self,
        images: list[object],
        prompt: str,
        labels: list[str],
    ) -> tuple[list[float], dict[str, object]]:
        self.calls.append((images, prompt, labels))
        return self._logits[len(self.calls) - 1], {
            "input_tokens": 32,
            "forward_seconds": 0.01,
        }


class TypedEngine:
    load_seconds = 0.0

    def labels(self, count: int, n_images: int) -> list[str]:
        assert count == 3
        assert n_images == 0
        return ["A", "B", "C"]

    def score_prompt(
        self,
        images: list[object],
        prompt: str,
        labels: list[str],
    ) -> tuple[list[float], dict[str, object]]:
        del images, prompt, labels
        return [3.0, 1.0, 0.0], {"input_tokens": 8}


def test_model_directory_requires_every_bound_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing model file"):
        verify_model_directory(tmp_path)


def test_stable_torch_numerics_are_applied_and_verified() -> None:
    matmul = SimpleNamespace(allow_bf16_reduced_precision_reduction=True)
    torch_module = SimpleNamespace(
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul))
    )

    observed = configure_torch_numerics(torch_module)

    assert observed == {"allow_bf16_reduced_precision_reduction": False}
    assert matmul.allow_bf16_reduced_precision_reduction is False


def test_portable_backend_scores_four_rotations_by_candidate_identity() -> None:
    engine = FakeEngine()
    backend = PortableBackend(
        engine=engine,
        identity={
            "name": "mentat-sys1-v0.1",
            "base_model": "Qwen/Qwen3.5-4B",
        },
        rotations=4,
    )
    request = DecisionRequest.model_validate(
        {
            "request_id": "rot4",
            "state": {},
            "fields": [
                {
                    "id": "decision",
                    "type": "choice",
                    "question": "Pick one.",
                    "options": [
                        {"value": "a"},
                        {"value": "b"},
                        {"value": "c"},
                    ],
                }
            ],
        }
    )

    results, usage = backend.score([], request)

    assert results[0].value == "a"
    assert len(engine.calls) == 4
    assert usage["rotations"] == 4
    assert usage["language_model_passes"] == 4
    assert usage["input_tokens"] == 32
    assert "A: a\nB: b\nC: c\nD: unknown" in engine.calls[0][1]
    rotated = engine.calls[1][1]
    assert (
        rotated.index("A: b")
        < rotated.index("B: c")
        < rotated.index("C: unknown")
        < rotated.index("D: a")
    )


def test_portable_backend_applies_frozen_temperature_to_rot4_scores() -> None:
    backend = PortableBackend(
        engine=FakeEngine(),
        identity={
            "name": "mentat-sys1-v0.1",
            "base_model": "Qwen/Qwen3.5-4B",
        },
        rotations=4,
        temperature=2.0,
        calibration_version="scalar_temperature-v1:abc123",
    )
    request = DecisionRequest.model_validate(
        {
            "request_id": "calibrated",
            "state": {},
            "fields": [
                {
                    "id": "decision",
                    "type": "choice",
                    "question": "Pick one.",
                    "options": [
                        {"value": "a"},
                        {"value": "b"},
                        {"value": "c"},
                    ],
                }
            ],
        }
    )

    results, _ = backend.score([], request)

    assert results[0].value == "a"
    assert results[0].score_semantics == "calibrated_normalized_scores"
    assert results[0].calibration_version == "scalar_temperature-v1:abc123"
    assert max(results[0].scores.values()) < 0.8


def test_backend_uses_configured_identity_and_routes_typed_temperature() -> None:
    backend = PortableBackend(
        engine=TypedEngine(),
        identity={"name": "mentat-sys1-v0.2"},
        temperature_by_type={
            "choice": 0.5,
            "noul": 2.0,
            "score": 4.0,
        },
        calibration_version="temperature_by_type-v2:abc123",
    )
    request = DecisionRequest.model_validate(
        {
            "request_id": "typed-calibration",
            "state": {},
            "fields": [
                {
                    "id": "choice",
                    "type": "choice",
                    "question": "Pick one.",
                    "options": [{"value": "a"}, {"value": "b"}],
                },
                {
                    "id": "boolean",
                    "type": "boolean",
                    "question": "Is this true?",
                },
                {
                    "id": "ordinal",
                    "type": "ordinal",
                    "question": "How much?",
                    "levels": [
                        {"value": 1, "description": "low"},
                        {"value": 2, "description": "high"},
                    ],
                },
            ],
        }
    )

    results, _ = backend.score([], request)

    confidences = [max(result.scores.values()) for result in results]
    assert backend.model_id == "mentat-sys1-v0.2"
    assert confidences[0] > confidences[1] > confidences[2]
    assert {result.calibration_version for result in results} == {
        "temperature_by_type-v2:abc123"
    }


def test_backend_load_requires_calibration_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[bool] = []

    def fake_verify(
        model_dir: Path,
        *,
        config: object,
        require_calibration: bool,
    ) -> dict[str, object]:
        del model_dir, config
        calls.append(require_calibration)
        return {
            "adapter_sha256": "a" * 64,
            "readout_sha256": "b" * 64,
            "readout_codes": 256,
            "temperature": 0.97,
            "temperature_by_type": {
                "choice": 0.97,
                "noul": 0.97,
                "score": 0.97,
            },
            "calibration_sha256": "c" * 64,
            "calibration_version": "scalar_temperature-v1:cccccccccccc",
        }

    fake_engine = SimpleNamespace(load_seconds=0.1)

    def fake_engine_load(**kwargs: Any) -> object:
        del kwargs
        return fake_engine

    monkeypatch.setattr(
        "mentat_sys1.inference.backend.verify_model_directory",
        fake_verify,
    )
    monkeypatch.setattr(
        "mentat_sys1.inference.backend.TorchDecisionEngine.load",
        fake_engine_load,
    )
    config = SimpleNamespace(
        project_id="mentat-sys1-v0.1",
        base_model=SimpleNamespace(
            repository="Qwen/Qwen3.5-4B",
            revision="revision",
        ),
        runtime=SimpleNamespace(max_length=16384),
    )

    backend = PortableBackend.load(
        config=config,  # type: ignore[arg-type]
        model_dir=tmp_path,
        base_model_path=tmp_path,
    )

    assert calls == [True]
    assert backend.rotations == 1
    assert backend.identity["rotations"] == 1
    assert backend.identity["calibration"] == "frozen"
    assert backend.identity["calibration_version"] == (
        "scalar_temperature-v1:cccccccccccc"
    )
