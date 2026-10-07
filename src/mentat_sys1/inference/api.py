"""FastAPI surface for the standalone System One contract.

Adapted from Imajev revision 8d4554e18b621fd2c144098876c3cee623ef28f3.
Modified to remove UI and repository-relative behavior, enforce model identity,
and return path-safe typed errors.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import re
import threading
from time import perf_counter
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from mentat_sys1.inference.contracts import DecisionRequest, DecisionResult
from mentat_sys1.inference.scoring import (
    to_decision_request,
    to_systemone_response,
)

MAXIMUM_REQUEST_BYTES = 32 * 1024 * 1024
MAXIMUM_IMAGE_BYTES = 20 * 1024 * 1024
MAXIMUM_IMAGE_PIXELS = 20_000_000
MAXIMUM_IMAGES = 2
IMAGE_DATA_URL = re.compile(
    r"data:image/(?P<format>png|jpeg|jpg|webp);base64,"
    r"(?P<payload>[A-Za-z0-9+/=]+)",
    re.IGNORECASE,
)
log = logging.getLogger("mentat_sys1.inference.api")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class Backend(Protocol):
    model_id: str
    identity: dict[str, object]

    def score(
        self,
        images: list[Any],
        request: DecisionRequest,
    ) -> tuple[list[DecisionResult], dict[str, object]]: ...


class RequestError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(code=code, message=message)
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _decode_image(data_url: str) -> tuple[Any, dict[str, object]]:
    match = IMAGE_DATA_URL.fullmatch(data_url.strip())
    if match is None:
        raise RequestError(422, "invalid_image", "Image must be a base64 data URL.")
    try:
        blob = base64.b64decode(match.group("payload"), validate=True)
    except (binascii.Error, ValueError) as error:
        raise RequestError(
            422,
            "invalid_image",
            "Image data is not valid base64.",
        ) from error
    if len(blob) > MAXIMUM_IMAGE_BYTES:
        raise RequestError(413, "image_too_large", "Image exceeds 20 MiB.")
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(blob))
        image.load()
    except (ImportError, OSError, ValueError) as error:
        raise RequestError(
            422,
            "invalid_image",
            "Image could not be decoded.",
        ) from error
    if image.width * image.height > MAXIMUM_IMAGE_PIXELS:
        raise RequestError(
            413,
            "image_too_large",
            "Image exceeds 20 million decoded pixels.",
        )
    converted = image.convert("RGB")
    return converted, {
        "width": converted.width,
        "height": converted.height,
        "size_bytes": len(blob),
    }


def _extract_images(
    payload: dict[str, Any],
) -> tuple[list[Any], list[dict[str, object]]]:
    urls: list[str] = []
    explicit = payload.pop("images", [])
    if not isinstance(explicit, list) or not all(
        isinstance(item, str) for item in explicit
    ):
        raise RequestError(
            422,
            "invalid_image",
            "The images field must be a list of data URLs.",
        )
    urls.extend(explicit)

    def walk(value: object) -> object:
        if isinstance(value, str) and "data:image/" in value:
            def replace(match: re.Match[str]) -> str:
                urls.append(match.group(0))
                return f"[image {len(urls)}]"

            return IMAGE_DATA_URL.sub(replace, value)
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value

    if "state" in payload:
        payload["state"] = walk(payload["state"])
    if len(urls) > MAXIMUM_IMAGES:
        raise RequestError(
            422,
            "invalid_image",
            f"A request supports at most {MAXIMUM_IMAGES} images.",
        )
    decoded = [_decode_image(url) for url in urls]
    return (
        [image for image, _ in decoded],
        [metadata for _, metadata in decoded],
    )


def create_app(
    backend: Backend,
    *,
    maximum_request_bytes: int = MAXIMUM_REQUEST_BYTES,
) -> FastAPI:
    if maximum_request_bytes < 1:
        raise ValueError("maximum_request_bytes must be positive")
    if backend.identity.get("name") != backend.model_id:
        raise ValueError("backend identity name differs from model ID")

    app = FastAPI(
        title="mentat-sys1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "model": backend.model_id}

    @app.get("/v1/models")
    def models() -> dict[str, list[dict[str, object]]]:
        return {"models": [dict(backend.identity)]}

    @app.post("/v1/systemone")
    async def systemone(http_request: Request) -> JSONResponse:
        body = await http_request.body()
        if len(body) > maximum_request_bytes:
            return _error(413, "request_too_large", "Request body is too large.")
        if (
            http_request.headers.get("content-type", "")
            .split(";", maxsplit=1)[0]
            .strip()
            .lower()
            != "application/json"
        ):
            return _error(
                415,
                "unsupported_media_type",
                "Content-Type must be application/json.",
            )
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            images, image_metadata = _extract_images(payload)
            decision_request = to_decision_request(
                payload,
                expected_model=backend.model_id,
            )
        except RequestError as error:
            return _error(error.status_code, error.code, error.message)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            return _error(422, "invalid_request", str(error))

        started = perf_counter()

        def run() -> tuple[list[DecisionResult], dict[str, object]]:
            with lock:
                return backend.score(images, decision_request)

        try:
            results, usage = await run_in_threadpool(run)
        except Exception:
            log.exception("model scoring failed")
            return _error(500, "internal_error", "Model scoring failed.")
        response = to_systemone_response(
            decision_request,
            results,
            model=backend.model_id,
        )
        response["usage"] = {
            **usage,
            "total_ms": round((perf_counter() - started) * 1000.0, 3),
            "image_metadata": image_metadata,
        }
        return JSONResponse(content=response)

    return app
