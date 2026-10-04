"""Portable PyTorch inference for the frozen mentat-sys1-v0.1 checkpoint.

Adapted from Imajev revision 8d4554e18b621fd2c144098876c3cee623ef28f3.
Modified to remove source-tree imports and bind loading to the portable model
manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from mentat_sys1 import MODEL_ID
from mentat_sys1.audit.provenance import sha256_file
from mentat_sys1.contracts import ProjectConfig
from mentat_sys1.inference.contracts import DecisionRequest, DecisionResult
from mentat_sys1.inference.scoring import (
    combine_rotations,
    compile_question,
    cyclic_offsets,
    result_from_logits,
    rotate,
    verified_label_ids,
)
from mentat_sys1.release.verify import verify_migrated_model

BASE_MODEL_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "decision_readout.json",
    "decision_readout.safetensors",
)
GENERATED_MODEL_FILES = (
    "model-manifest.json",
    "SHA256SUMS",
)
MODEL_FILES = (*BASE_MODEL_FILES, *GENERATED_MODEL_FILES)
CALIBRATED_MODEL_FILES = (
    *BASE_MODEL_FILES,
    "calibration.json",
    *GENERATED_MODEL_FILES,
)
STABLE_TORCH_NUMERICS = {
    "allow_bf16_reduced_precision_reduction": False,
}
DEFAULT_ROTATIONS = 1


class DecisionEngine(Protocol):
    load_seconds: float

    def labels(self, count: int, n_images: int) -> list[str]: ...

    def score_prompt(
        self,
        images: list[object],
        prompt: str,
        labels: list[str],
    ) -> tuple[list[float], dict[str, object]]: ...


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _verify_manifest_structure(
    model_dir: Path,
    *,
    require_calibration: bool,
) -> dict[str, Any]:
    manifest = _load_json_object(model_dir / "model-manifest.json")
    records = manifest.get("files")
    expected = set(BASE_MODEL_FILES)
    if require_calibration:
        expected.add("calibration.json")
    if manifest.get("schema_version") != 1 or not isinstance(records, dict):
        raise ValueError("invalid model manifest")
    if set(records) != expected:
        raise ValueError("model manifest file set differs")
    for name in sorted(expected):
        record = records[name]
        path = model_dir / name
        if not isinstance(record, dict):
            raise ValueError(f"invalid model manifest record: {name}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"manifest SHA-256 mismatch: {name}")
        if record.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"manifest size mismatch: {name}")
    return manifest


def verify_model_directory(
    model_dir: Path,
    *,
    config: ProjectConfig | None = None,
    require_calibration: bool = True,
) -> dict[str, object]:
    """Verify portable files before importing heavyweight model libraries."""

    root = Path(model_dir)
    required_files = (
        CALIBRATED_MODEL_FILES if require_calibration else MODEL_FILES
    )
    missing = [name for name in required_files if not (root / name).is_file()]
    if missing:
        raise ValueError(f"missing model file: {', '.join(missing)}")
    manifest = _verify_manifest_structure(
        root,
        require_calibration=require_calibration,
    )
    if config is not None:
        return verify_migrated_model(
            config=config,
            model_dir=root,
            require_calibration=require_calibration,
        )
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("model manifest identity differs")
    calibration: dict[str, object] = {}
    if require_calibration:
        payload = _load_json_object(root / "calibration.json")
        temperature = payload.get("temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0
        ):
            raise ValueError("calibration temperature must be finite and positive")
        digest = sha256_file(root / "calibration.json")
        calibration = {
            "temperature": float(temperature),
            "calibration_sha256": digest,
            "calibration_version": f"scalar_temperature-v1:{digest[:12]}",
        }
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "base_model": manifest.get("base_model"),
        **calibration,
    }


def configure_torch_numerics(torch_module: Any) -> dict[str, bool]:
    matmul = torch_module.backends.cuda.matmul
    matmul.allow_bf16_reduced_precision_reduction = False
    observed = {
        "allow_bf16_reduced_precision_reduction": bool(
            matmul.allow_bf16_reduced_precision_reduction
        ),
    }
    if observed != STABLE_TORCH_NUMERICS:
        raise RuntimeError(f"failed to apply stable torch numerics: {observed}")
    return observed


class TorchDecisionEngine:
    """Exact PyTorch decision-position scorer used by the registered model."""

    def __init__(
        self,
        *,
        torch_module: Any,
        processor: Any,
        model: Any,
        readout: Any,
        codebook: tuple[tuple[str, int], ...],
        device: str,
        max_length: int,
        load_seconds: float,
    ) -> None:
        self._torch = torch_module
        self.processor = processor
        self.model = model
        self.readout = readout
        self.codebook = codebook
        self.device = device
        self.max_length = max_length
        self.load_seconds = load_seconds

    @classmethod
    def load(
        cls,
        *,
        base_model_path: Path,
        model_dir: Path,
        device: str = "cuda",
        max_length: int = 16384,
    ) -> TorchDecisionEngine:
        if not Path(base_model_path).is_dir():
            raise ValueError("local base model snapshot is missing")
        if max_length < 1:
            raise ValueError("max_length must be positive")

        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import (
            AutoProcessor,
            Qwen3_5ForConditionalGeneration,
        )

        configure_torch_numerics(torch)
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        started = perf_counter()
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            str(base_model_path),
            local_files_only=True,
        )
        base = cast(
            Any,
            Qwen3_5ForConditionalGeneration.from_pretrained(
                str(base_model_path),
                local_files_only=True,
                dtype=dtype,
            ),
        ).to(device)
        model = PeftModel.from_pretrained(
            base,
            str(model_dir),
            local_files_only=True,
        ).eval()

        readout_manifest = _load_json_object(
            Path(model_dir) / "decision_readout.json"
        )
        raw_codes = readout_manifest.get("codes")
        if readout_manifest.get("version") != 1 or not isinstance(raw_codes, list):
            raise ValueError("invalid decision readout manifest")
        codebook: list[tuple[str, int]] = []
        for row in raw_codes:
            if not isinstance(row, dict):
                raise ValueError("invalid decision readout code")
            code, token_id = row.get("code"), row.get("token_id")
            if not isinstance(code, str) or not isinstance(token_id, int):
                raise ValueError("invalid decision readout binding")
            codebook.append((code, token_id))
        if len(codebook) != 256 or len(set(codebook)) != len(codebook):
            raise ValueError("decision readout must contain 256 unique bindings")

        tensors = load_file(
            str(Path(model_dir) / "decision_readout.safetensors"),
            device=str(device),
        )
        if set(tensors) != {"weight"}:
            raise ValueError("decision readout tensor file must contain only weight")
        weight = tensors["weight"].float()
        language = cast(Any, model.get_base_model())
        hidden_size = int(language.lm_head.weight.shape[1])
        if tuple(weight.shape) != (len(codebook), hidden_size):
            raise ValueError("decision readout tensor shape differs")
        if not bool(torch.isfinite(weight).all().item()):
            raise ValueError("decision readout tensor must be finite")
        readout = torch.nn.Linear(
            hidden_size,
            len(codebook),
            bias=False,
            device=device,
            dtype=torch.float32,
        )
        readout.weight.data.copy_(weight)
        readout.weight.requires_grad_(False)
        engine = cls(
            torch_module=torch,
            processor=processor,
            model=model,
            readout=readout.eval(),
            codebook=tuple(codebook),
            device=device,
            max_length=max_length,
            load_seconds=perf_counter() - started,
        )
        engine._verify_codebook(0)
        return engine

    def _render(self, prompt: str, n_images: int) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image"} for _ in range(n_images)),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            rendered = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except ValueError as error:
            if "does not have a chat template" not in str(error):
                raise
            rendered = self.processor.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        if not isinstance(rendered, str) or not rendered.endswith(
            "<think>\n\n</think>\n\n"
        ):
            raise ValueError("unexpected Qwen non-thinking template boundary")
        return rendered

    def _verify_codebook(self, n_images: int) -> None:
        rendered = self._render("", n_images)
        labels = [code for code, _ in self.codebook]
        observed = verified_label_ids(self.processor.tokenizer, rendered, labels)
        expected = [token_id for _, token_id in self.codebook]
        if observed != expected:
            raise ValueError(
                "decision readout code/token binding does not match this tokenizer"
            )

    def labels(self, count: int, n_images: int) -> list[str]:
        if not 1 <= count <= len(self.codebook):
            raise ValueError(
                f"{count} candidates exceed the {len(self.codebook)}-code readout"
            )
        labels = [code for code, _ in self.codebook[:count]]
        rendered = self._render("", n_images)
        observed = verified_label_ids(self.processor.tokenizer, rendered, labels)
        expected = [token_id for _, token_id in self.codebook[:count]]
        if observed != expected:
            raise ValueError(
                "decision readout code/token binding does not match this tokenizer"
            )
        return labels

    def _prepare(
        self,
        images: list[object],
        prompt: str,
        labels: list[str],
    ) -> tuple[str, dict[str, Any], list[int]]:
        if len(images) > 2:
            raise ValueError("inference supports at most two images")
        rendered = self._render(prompt, len(images))
        token_ids = verified_label_ids(
            self.processor.tokenizer,
            rendered,
            labels,
        )
        expected = [token_id for _, token_id in self.codebook[: len(labels)]]
        if token_ids != expected:
            raise ValueError(
                "decision readout code/token binding does not match this tokenizer"
            )
        inputs = self.processor(
            text=[rendered],
            images=images or None,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        if int(input_ids.shape[-1]) > self.max_length:
            raise ValueError(
                f"processed request exceeds the {self.max_length}-token limit"
            )
        suffix = self.processor.tokenizer.encode(
            "</think>\n\n",
            add_special_tokens=False,
        )
        if input_ids[0, -len(suffix) :].tolist() != suffix:
            raise ValueError("processed decision-position suffix mismatch")
        if images:
            grids = inputs.get("image_grid_thw")
            if grids is None or len(grids) != len(images):
                raise ValueError("processor returned an invalid image grid")
            if any(
                int(grid[0] * grid[1] * grid[2] // 4) > 2048 for grid in grids
            ):
                raise ValueError("image exceeds the 2048-visual-token limit")
        return rendered, dict(inputs), token_ids

    def score_prompt(
        self,
        images: list[object],
        prompt: str,
        labels: list[str],
    ) -> tuple[list[float], dict[str, object]]:
        started = perf_counter()
        rendered, inputs, token_ids = self._prepare(images, prompt, labels)
        preprocess_seconds = perf_counter() - started
        moved = {key: value.to(self.device) for key, value in inputs.items()}
        base = self.model.get_base_model()
        forward_started = perf_counter()
        with self._torch.no_grad():
            hidden = base.model(**moved).last_hidden_state[0, -1]
            logits = self.readout(hidden.float())[: len(labels)]
        values = [float(value) for value in logits.detach().cpu().tolist()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate logits must be finite")
        return values, {
            "preprocess_seconds": preprocess_seconds,
            "forward_seconds": perf_counter() - forward_started,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "choice_token_ids": token_ids,
            "readout_indices": list(range(len(labels))),
            "template_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "logit_precision": "float32_candidate_head",
        }


class PortableBackend:
    model_id = MODEL_ID

    def __init__(
        self,
        *,
        engine: DecisionEngine,
        identity: dict[str, object],
        rotations: int = DEFAULT_ROTATIONS,
        temperature: float = 1.0,
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
        self.engine = engine
        self.identity = dict(identity)
        self.rotations = rotations
        self.temperature = temperature
        self.calibration_version = calibration_version

    @classmethod
    def load(
        cls,
        *,
        config: ProjectConfig,
        model_dir: Path,
        base_model_path: Path,
        device: str = "cuda",
        require_calibration: bool = True,
    ) -> PortableBackend:
        verified = verify_model_directory(
            model_dir,
            config=config,
            require_calibration=require_calibration,
        )
        engine = TorchDecisionEngine.load(
            base_model_path=base_model_path,
            model_dir=model_dir,
            device=device,
            max_length=config.runtime.max_length,
        )
        calibration_version = verified.get("calibration_version")
        temperature = verified.get("temperature", 1.0)
        if not isinstance(temperature, float):
            raise ValueError("verified calibration temperature is missing")
        if calibration_version is not None and not isinstance(
            calibration_version,
            str,
        ):
            raise ValueError("verified calibration version is invalid")
        identity: dict[str, object] = {
            "name": config.project_id,
            "base_model": config.base_model.repository,
            "base_revision": config.base_model.revision,
            "adapter_sha256": verified["adapter_sha256"],
            "readout_sha256": verified["readout_sha256"],
            "readout_codes": verified["readout_codes"],
            "rotations": DEFAULT_ROTATIONS,
            "calibration": "frozen" if require_calibration else "pending",
        }
        if require_calibration:
            identity["calibration_sha256"] = verified["calibration_sha256"]
            identity["calibration_version"] = calibration_version
        return cls(
            engine=engine,
            identity=identity,
            rotations=DEFAULT_ROTATIONS,
            temperature=temperature,
            calibration_version=calibration_version,
        )

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
                        temperature=self.temperature,
                        calibration_version=self.calibration_version,
                    )
                )
            else:
                results.append(
                    combine_rotations(
                        choices,
                        [(offset, logits) for offset, logits, _ in passes],
                        temperature=self.temperature,
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
