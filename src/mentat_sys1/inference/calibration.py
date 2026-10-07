"""Deterministic scalar-temperature calibration."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np

from mentat_sys1.audit.provenance import sha256_file, write_immutable_json
from mentat_sys1.contracts import ProjectConfig
from mentat_sys1.inference.contracts import (
    UNKNOWN,
    DecisionRequest,
    DecisionResult,
)

LogitRowInput: TypeAlias = Sequence[float] | np.ndarray[Any, Any]  # noqa: UP040
LogitsInput: TypeAlias = (  # noqa: UP040
    Sequence[LogitRowInput] | np.ndarray[Any, Any]
)
CALIBRATION_ROWS = 1120
CALIBRATION_SHA256 = (
    "a116f4d228e25bac5df61b9e52191adfe4181a0852f6050d9040cee7f35b0a02"
)
CalibrationType: TypeAlias = Literal["choice", "noul", "score"]  # noqa: UP040
CALIBRATION_TYPES: tuple[CalibrationType, ...] = ("choice", "noul", "score")


class CalibrationBackend(Protocol):
    def score(
        self,
        images: list[Any],
        request: DecisionRequest,
    ) -> tuple[list[DecisionResult], dict[str, object]]: ...


@dataclass(frozen=True)
class CalibrationMetrics:
    ece: float
    nll: float
    brier: float


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    before: CalibrationMetrics
    after: CalibrationMetrics


@dataclass(frozen=True)
class CalibrationParameters:
    pooled_temperature: float
    temperature_by_type: dict[CalibrationType, float]
    version: str


def _validated_rows(
    logits: LogitsInput,
    targets: Sequence[int] | np.ndarray[Any, Any],
) -> tuple[tuple[np.ndarray[Any, Any], ...], tuple[int, ...]]:
    if isinstance(logits, np.ndarray):
        if logits.ndim != 2:
            raise ValueError("logits must be a two-dimensional array")
        rows = tuple(np.asarray(row, dtype=np.float64) for row in logits)
    else:
        rows = tuple(np.asarray(row, dtype=np.float64) for row in logits)
    raw_targets = np.asarray(targets)
    if raw_targets.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    target_values = tuple(int(value) for value in raw_targets.tolist())
    if not rows or len(rows) != len(target_values):
        raise ValueError("expected one target per non-empty logit row")
    for row, target in zip(rows, target_values, strict=True):
        if row.ndim != 1 or len(row) < 2:
            raise ValueError("each logit row must contain at least two candidates")
        if not bool(np.isfinite(row).all()):
            raise ValueError("logits must be finite")
        if not 0 <= target < len(row):
            raise ValueError("target index is outside its logit row")
    return rows, target_values


def _probabilities(
    row: np.ndarray[Any, Any],
    temperature: float,
) -> np.ndarray[Any, Any]:
    scaled = row / temperature
    shifted = scaled - np.max(scaled)
    weights = np.exp(shifted)
    return weights / np.sum(weights, dtype=np.float64)


def metrics(
    logits: LogitsInput,
    targets: Sequence[int] | np.ndarray[Any, Any],
    *,
    temperature: float,
    bins: int = 15,
) -> CalibrationMetrics:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if bins < 2:
        raise ValueError("ECE bins must be at least two")
    rows, target_values = _validated_rows(logits, targets)

    nll_total = 0.0
    brier_total = 0.0
    confidences: list[float] = []
    correct: list[float] = []
    for row, target in zip(rows, target_values, strict=True):
        probabilities = _probabilities(row, temperature)
        nll_total -= math.log(float(probabilities[target]))
        one_hot = np.zeros(len(row), dtype=np.float64)
        one_hot[target] = 1.0
        brier_total += float(np.square(probabilities - one_hot).sum())
        prediction = int(np.argmax(row))
        confidences.append(float(np.max(probabilities)))
        correct.append(float(prediction == target))

    ece = 0.0
    sample_count = len(rows)
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item, confidence in enumerate(confidences)
            if (confidence >= lower if index == 0 else confidence > lower)
            and confidence <= upper
        ]
        if members:
            bin_accuracy = math.fsum(correct[item] for item in members) / len(
                members
            )
            bin_confidence = math.fsum(
                confidences[item] for item in members
            ) / len(members)
            ece += len(members) / sample_count * abs(
                bin_accuracy - bin_confidence
            )
    return CalibrationMetrics(
        ece=ece,
        nll=nll_total / sample_count,
        brier=brier_total / sample_count,
    )


def fit_temperature(
    logits: LogitsInput,
    targets: Sequence[int] | np.ndarray[Any, Any],
) -> TemperatureFit:
    rows, target_values = _validated_rows(logits, targets)
    before = metrics(rows, target_values, temperature=1.0)

    lower = -8.0
    upper = 8.0
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)

    def objective(log_temperature: float) -> float:
        return metrics(
            rows,
            target_values,
            temperature=math.exp(log_temperature),
        ).nll

    left_value = objective(left)
    right_value = objective(right)
    for _ in range(128):
        if left_value <= right_value:
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        else:
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)

    candidate = math.exp((lower + upper) / 2.0)
    after = metrics(rows, target_values, temperature=candidate)
    if after.nll > before.nll:
        candidate = 1.0
        after = before
    return TemperatureFit(
        temperature=candidate,
        before=before,
        after=after,
    )


def _validated_temperature(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def load_temperature_calibration(
    payload: Mapping[str, object],
    *,
    digest: str,
) -> CalibrationParameters:
    method = payload.get("method")
    schema_version = payload.get("schema_version")
    if schema_version == 1 and method == "scalar_temperature":
        temperature = _validated_temperature(
            payload.get("temperature"),
            name="calibration temperature",
        )
        return CalibrationParameters(
            pooled_temperature=temperature,
            temperature_by_type={kind: temperature for kind in CALIBRATION_TYPES},
            version=f"scalar_temperature-v1:{digest[:12]}",
        )
    if schema_version != 2 or method != "temperature_by_type":
        raise ValueError("unsupported calibration schema")
    pooled = _validated_temperature(
        payload.get("pooled_temperature"),
        name="pooled calibration temperature",
    )
    raw_temperatures = payload.get("temperature_by_type")
    if not isinstance(raw_temperatures, dict) or set(raw_temperatures) != set(
        CALIBRATION_TYPES
    ):
        raise ValueError("calibration temperatures must cover choice, noul, score")
    temperatures: dict[CalibrationType, float] = {}
    for kind in CALIBRATION_TYPES:
        temperatures[kind] = _validated_temperature(
            raw_temperatures[kind],
            name=f"{kind} calibration temperature",
        )
    return CalibrationParameters(
        pooled_temperature=pooled,
        temperature_by_type=temperatures,
        version=f"temperature_by_type-v2:{digest[:12]}",
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"calibration row {line_number} must be an object"
                )
            rows.append(payload)
    return rows


def _target_key(target: object) -> str:
    if target is None:
        return UNKNOWN
    if isinstance(target, bool):
        return str(target).lower()
    if isinstance(target, (str, int)):
        return str(target)
    raise ValueError("calibration target has an unsupported type")


def calibrate_dataset(
    *,
    config: ProjectConfig,
    calibration_data: Path,
    artifact_root: Path,
    backend: CalibrationBackend | None = None,
    model_dir: Path | None = None,
    base_model_path: Path | None = None,
    device: str = "cuda",
    expected_rows: int = CALIBRATION_ROWS,
    expected_sha256: str = CALIBRATION_SHA256,
) -> dict[str, Any]:
    calibration_path = Path(calibration_data)
    observed_sha256 = sha256_file(calibration_path)
    if observed_sha256 != expected_sha256:
        raise ValueError("calibration split SHA-256 differs")
    rows = _load_rows(calibration_path)
    if len(rows) != expected_rows:
        raise ValueError(
            f"calibration split must contain {expected_rows} rows"
        )
    identifiers = [row.get("id") for row in rows]
    if (
        any(
            not isinstance(identifier, str) or not identifier
            for identifier in identifiers
        )
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError("calibration row IDs must be non-empty and unique")

    if backend is None:
        if model_dir is None or base_model_path is None:
            raise ValueError(
                "model_dir and base_model_path are required to score calibration"
            )
        from mentat_sys1.inference.backend import PortableBackend

        backend = PortableBackend.load(
            config=config,
            model_dir=model_dir,
            base_model_path=base_model_path,
            device=device,
            require_calibration=False,
        )

    logits: list[list[float]] = []
    targets: list[int] = []
    model_passes = 0
    rotation_counts: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        if row.get("partition", "calibration") != "calibration":
            raise ValueError(
                f"calibration row {row_number} has the wrong partition"
            )
        images = row.get("images", [])
        if images != []:
            raise ValueError("registered calibration split must be text-only")
        request = DecisionRequest.model_validate(row.get("request"))
        if len(request.fields) != 1:
            raise ValueError("calibration rows must contain exactly one field")
        results, usage = backend.score([], request)
        if len(results) != 1:
            raise ValueError("calibration backend returned the wrong result count")
        keys = list(results[0].raw_logits)
        target_key = _target_key(row.get("target"))
        if target_key not in results[0].raw_logits:
            raise ValueError(
                f"calibration target is absent from row {row_number} candidates"
            )
        logits.append([results[0].raw_logits[key] for key in keys])
        targets.append(keys.index(target_key))
        raw_passes = usage.get("language_model_passes", 0)
        if isinstance(raw_passes, bool) or not isinstance(raw_passes, int):
            raise ValueError("calibration backend returned invalid pass metadata")
        model_passes += raw_passes
        raw_rotations = usage.get("rotations", 0)
        if (
            isinstance(raw_rotations, bool)
            or not isinstance(raw_rotations, int)
            or raw_rotations < 1
        ):
            raise ValueError(
                "calibration backend returned invalid rotation metadata"
            )
        rotation_counts.add(raw_rotations)

    if len(rotation_counts) != 1:
        raise ValueError("calibration backend changed rotation count")
    rotations = next(iter(rotation_counts))

    fit = fit_temperature(logits, targets)
    predictions_before = [
        int(np.argmax(row))
        for row in logits
    ]
    predictions_after = [
        int(np.argmax(np.asarray(row, dtype=np.float64) / fit.temperature))
        for row in logits
    ]
    predictions_unchanged = predictions_before == predictions_after
    if not predictions_unchanged:
        raise ValueError("scalar calibration changed predictions")
    if fit.after.nll > fit.before.nll + 1e-12:
        raise ValueError("scalar calibration regressed NLL")

    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "scalar_temperature",
        "temperature": fit.temperature,
        "optimizer": "bounded_log_temperature",
        "objective": "negative_log_likelihood",
        "split": {
            "rows": len(rows),
            "sha256": observed_sha256,
        },
        "model": {
            "adapter_sha256": config.release.adapter_sha256,
            "readout_sha256": config.release.readout_sha256,
        },
        "before": asdict(fit.before),
        "after": asdict(fit.after),
        "predictions_unchanged": predictions_unchanged,
        "scoring": {
            "rotation_mode": "single" if rotations == 1 else f"rot{rotations}",
            "rotations": rotations,
            "model_passes": model_passes,
        },
    }
    output = Path(artifact_root) / config.paths.evidence / "calibration-report.json"
    write_immutable_json(output, report)
    return report
