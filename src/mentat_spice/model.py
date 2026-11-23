"""Minimal state/questions interface around the frozen native decision contract."""
from __future__ import annotations

import io
from pathlib import Path
from threading import Lock
from time import perf_counter

from PIL import Image

from .runner import PortableBackend
from .scoring import to_decision_request, to_systemone_response


class DecisionModel:
    model_id = "mentat-Spice-4B"
    max_images = 4
    max_image_pixels = 1_638_400

    def __init__(self, base_model, adapter_directory, *, device="cuda:0"):
        from .engine import NativeEngine

        self._initialize(NativeEngine(base_model, adapter_directory, device=device))

    def _initialize(self, engine):
        self.engine = engine
        self.backend = PortableBackend(
            engine=engine, identity={"name": "mentat-jev-v1"},
            rotations=1, temperature=1.0,
        )
        self._lock = Lock()

    def _images(self, raw):
        if not isinstance(raw, list) or len(raw) > self.max_images:
            raise ValueError("images must be a list of at most four images")
        result = []
        for value in raw:
            if isinstance(value, Image.Image):
                image = value
            elif isinstance(value, (str, Path)):
                with Image.open(value) as source:
                    source.load()
                    image = source.copy()
            elif isinstance(value, bytes):
                with Image.open(io.BytesIO(value)) as source:
                    source.load()
                    image = source.copy()
            else:
                raise TypeError("Use a PIL image, a local image path, or encoded bytes")
            if image.width * image.height > self.max_image_pixels:
                raise ValueError("Image exceeds 1,638,400 pixels; resize explicitly before calling")
            result.append(image.convert("RGB"))
        return result

    def predict(self, payload, *, images=None):
        started = perf_counter()
        if not isinstance(payload, dict):
            raise ValueError("Request must be an object")
        body = dict(payload)
        if images is not None and "images" in body:
            raise ValueError("Specify images either in the payload or as an argument")
        raw_images = body.pop("images", []) if images is None else images
        requested = body.get("model", self.model_id)
        if requested != self.model_id:
            raise ValueError("Requested model differs from the loaded model")
        body["model"] = "mentat-jev-v1"
        request = to_decision_request(body, expected_model="mentat-jev-v1")
        decoded = self._images(raw_images)
        with self._lock:
            results, usage = self.backend.score(decoded, request)
        response = to_systemone_response(request, results, model=self.model_id)
        response["usage"] = {
            **usage, "images": len(decoded), "generated_answer_tokens": 0,
        }
        response["timing"] = {"total_ms": 1000 * (perf_counter() - started)}
        return response
