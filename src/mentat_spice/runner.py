"""Native request runner, extracted without model loading or training dependencies.

Adapted from the frozen Mentat backend. Scoring and typed normalization are
preserved; calibrated-runtime fitting and release/training loaders are excluded.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias
import math
from .contracts import DecisionRequest, DecisionResult
from .scoring import compile_question, cyclic_offsets, rotate, result_from_logits, combine_rotations
MODEL_ID = "mentat-jev-v1"
CalibrationType: TypeAlias = Literal["choice", "noul", "score"]
CALIBRATION_TYPES = ("choice", "noul", "score")
DEFAULT_ROTATIONS = 1
class PortableBackend:
    model_id = MODEL_ID

    def __init__(
            self,
            *,
            engine: DecisionEngine,
            identity: dict[str, object],
            rotations: int = DEFAULT_ROTATIONS,
            temperature: float = 1.0,
            temperature_by_type: Mapping[str, float] | None = None,
            calibration_version: str | None = None,
        ) -> None:
            if identity.get("name") != self.model_id:
                raise ValueError("backend identity name differs from model ID")
            if rotations < 1:
                raise ValueError("rotations must be positive")
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
            if calibration_version is None and temperature != 1.0:
                raise ValueError("non-unit temperature requires a calibration version")
            if temperature_by_type is None:
                typed_temperatures = {
                    kind: temperature for kind in CALIBRATION_TYPES
                }
            else:
                if set(temperature_by_type) != set(CALIBRATION_TYPES):
                    raise ValueError(
                        "typed temperatures must cover choice, noul, score"
                    )
                typed_temperatures = {}
                for kind in CALIBRATION_TYPES:
                    value = temperature_by_type[kind]
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(
                            "typed temperatures must be finite and positive"
                        )
                    typed_temperatures[kind] = float(value)
                if calibration_version is None:
                    raise ValueError(
                        "typed temperatures require a calibration version"
                    )
            self.engine = engine
            self.identity = dict(identity)
            self.rotations = rotations
            self.temperature = temperature
            self.temperature_by_type = typed_temperatures
            self.calibration_version = calibration_version

    def score(
            self,
            images: list[Any],
            request: DecisionRequest,
        ) -> tuple[list[DecisionResult], dict[str, object]]:
            def numeric_metadata(
                metadata: dict[str, object],
                key: str,
            ) -> float:
                value = metadata.get(key, 0.0)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"engine metadata {key} must be numeric")
                number = float(value)
                if not math.isfinite(number) or number < 0:
                    raise ValueError(f"engine metadata {key} must be finite")
                return number
    
            results: list[DecisionResult] = []
            forward_seconds = 0.0
            preprocess_seconds = 0.0
            input_tokens = 0
            model_passes = 0
            for field in request.fields:
                runtime_types: dict[str, CalibrationType] = {
                    "choice": "choice",
                    "boolean": "noul",
                    "ordinal": "score",
                }
                runtime_type = runtime_types[field.type]
                temperature = self.temperature_by_type[runtime_type]
                header, choices, texts = compile_question(field, request.state)
                labels = self.engine.labels(len(choices), len(images))
                passes: list[tuple[int, Sequence[float], Sequence[int] | None]] = []
                for offset in cyclic_offsets(len(choices), self.rotations):
                    prompt = header + "\n".join(
                        f"{label}: {text}"
                        for label, text in zip(
                            labels,
                            rotate(texts, offset),
                            strict=True,
                        )
                    )
                    logits, metadata = self.engine.score_prompt(images, prompt, labels)
                    raw_token_ids = metadata.get("choice_token_ids")
                    token_ids = (
                        list(raw_token_ids)
                        if isinstance(raw_token_ids, list)
                        and all(
                            isinstance(value, int) and not isinstance(value, bool)
                            for value in raw_token_ids
                        )
                        else None
                    )
                    passes.append((offset, logits, token_ids))
                    preprocess_seconds += numeric_metadata(
                        metadata,
                        "preprocess_seconds",
                    )
                    forward_seconds += numeric_metadata(metadata, "forward_seconds")
                    input_tokens = max(
                        input_tokens,
                        int(numeric_metadata(metadata, "input_tokens")),
                    )
                    model_passes += 1
                if len(passes) == 1:
                    _, single_logits, single_token_ids = passes[0]
                    results.append(
                        result_from_logits(
                            choices,
                            single_logits,
                            token_ids=single_token_ids,
                            temperature=temperature,
                            calibration_version=self.calibration_version,
                        )
                    )
                else:
                    results.append(
                        combine_rotations(
                            choices,
                            [(offset, logits) for offset, logits, _ in passes],
                            temperature=temperature,
                            calibration_version=self.calibration_version,
                        )
                    )
            return results, {
                "preprocess_ms": round(preprocess_seconds * 1000.0, 3),
                "questions_ms": round(forward_seconds * 1000.0, 3),
                "input_tokens": input_tokens,
                "rotations": self.rotations,
                "language_model_passes": model_passes,
            }
