"""Strict typed decision contracts.

Adapted from Imajev revision 8d4554e18b621fd2c144098876c3cee623ef28f3.
Modified for the standalone mentat-sys1-v0.1 namespace and release contract.
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

MAX_STATE_BYTES = 131072
MAX_INTERNAL_FIELDS = 64
UNKNOWN = "__unknown__"
Text = Annotated[StrictStr, Field(min_length=1, max_length=2000)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Option(StrictModel):
    value: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    description: Text | None = None

    @model_validator(mode="after")
    def reject_reserved_value(self) -> Self:
        if self.value == UNKNOWN:
            raise ValueError(f"{UNKNOWN} is reserved")
        return self


class Level(StrictModel):
    value: StrictInt
    description: Text


class CommonField(StrictModel):
    id: Annotated[StrictStr, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
    question: Text


class ChoiceField(CommonField):
    type: Literal["choice"]
    options: Annotated[list[Option], Field(min_length=2, max_length=255)]

    @model_validator(mode="after")
    def require_unique_values(self) -> Self:
        if len({option.value for option in self.options}) != len(self.options):
            raise ValueError("choice values must be unique")
        return self


class BooleanField(CommonField):
    type: Literal["boolean"]
    yes_description: Text | None = None
    no_description: Text | None = None


class OrdinalField(CommonField):
    type: Literal["ordinal"]
    levels: Annotated[list[Level], Field(min_length=2, max_length=10)]

    @model_validator(mode="after")
    def require_ordered_unique_levels(self) -> Self:
        values = [level.value for level in self.levels]
        if values != sorted(set(values)):
            raise ValueError("ordinal levels must be unique and ascending")
        return self


DecisionField: TypeAlias = Annotated[  # noqa: UP040
    ChoiceField | BooleanField | OrdinalField,
    Field(discriminator="type"),
]


def _validate_json_value(value: object, depth: int = 0) -> None:
    if depth > 8:
        raise ValueError("state nesting exceeds 8")
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("state keys must be strings")
        for item in value.values():
            _validate_json_value(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json_value(item, depth + 1)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("state must contain JSON values")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("state numbers must be finite")


class DecisionRequest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    state: dict[str, Any] | StrictStr = Field(default_factory=dict)
    fields: Annotated[list[DecisionField], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if len({field.id for field in self.fields}) != len(self.fields):
            raise ValueError("field IDs must be unique")
        _validate_json_value(self.state)
        encoded = json.dumps(
            self.state,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > MAX_STATE_BYTES:
            raise ValueError(f"state exceeds {MAX_STATE_BYTES} bytes")
        return self


ResultValue: TypeAlias = StrictStr | StrictInt | bool | None  # noqa: UP040


class DecisionResult(StrictModel):
    status: Literal["answered", "abstained"]
    value: ResultValue
    scores: dict[str, float]
    raw_logits: dict[str, float]
    score_semantics: Literal[
        "uncalibrated_normalized_scores",
        "calibrated_normalized_scores",
    ] = "uncalibrated_normalized_scores"
    calibration_version: StrictStr | None = None
    reason: Literal["insufficient_evidence"] | None = None

    @field_validator("scores", "raw_logits")
    @classmethod
    def require_finite_values(cls, values: dict[str, float]) -> dict[str, float]:
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("scores and logits must be finite")
        return values

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        calibrated = self.score_semantics == "calibrated_normalized_scores"
        if calibrated != bool(self.calibration_version):
            raise ValueError("calibrated scores require a calibration version")
        if (
            not self.scores
            or set(self.scores) != set(self.raw_logits)
            or UNKNOWN not in self.scores
        ):
            raise ValueError("score and logit keys must match and include unknown")
        if any(value < 0.0 or value > 1.0 for value in self.scores.values()):
            raise ValueError("normalized scores must be in [0, 1]")
        if not math.isclose(math.fsum(self.scores.values()), 1.0, abs_tol=1e-6):
            raise ValueError("scores must sum to one")
        if self.status == "abstained":
            if self.value is not None or self.reason is None:
                raise ValueError("abstention requires a null value and reason")
        elif self.value is None or self.reason is not None:
            raise ValueError("answered result requires a value and no reason")
        return self
