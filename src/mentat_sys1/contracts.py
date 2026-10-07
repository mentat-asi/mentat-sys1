"""Frozen public runtime configuration for mentat-sys1-v0.1."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ModelId = Literal["mentat-sys1-v0.1", "mentat-sys1-v0.2"]
SUPPORTED_MODEL_IDS: frozenset[str] = frozenset(
    {"mentat-sys1-v0.1", "mentat-sys1-v0.2"}
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BaseModelIdentity(FrozenModel):
    repository: Literal["Qwen/Qwen3.5-4B"]
    revision: Literal["851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"]


class RelativePaths(FrozenModel):
    data: Path
    model: Path
    evidence: Path

    @field_validator("data", "model", "evidence")
    @classmethod
    def require_relative_path(cls, value: Path) -> Path:
        if value.is_absolute():
            raise ValueError("artifact paths must be relative")
        if not value.parts or ".." in value.parts:
            raise ValueError("artifact paths must stay within the artifact root")
        return value


class RuntimeConfig(FrozenModel):
    readout_codes: Literal[256]
    max_length: Literal[16384]


class ReleaseGate(FrozenModel):
    adapter_sha256: str
    readout_sha256: str
    legacy_public_correct: Annotated[int, Field(ge=0, le=231)]
    legacy_public_total: Literal[231]
    legacy_public_easy_correct: Annotated[int, Field(ge=0, le=48)]
    maximum_invalid: Literal[0]
    maximum_severe_failures: Literal[0]

    @field_validator("adapter_sha256", "readout_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("release hashes must be lowercase SHA-256")
        return value


class ProjectConfig(FrozenModel):
    schema_version: Literal[1]
    project_id: ModelId
    base_model: BaseModelIdentity
    paths: RelativePaths
    runtime: RuntimeConfig
    release: ReleaseGate


def load_config(path: Path) -> ProjectConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectConfig.model_validate(payload)
