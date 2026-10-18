"""HTTP adapter for the public v0.3 prediction interface."""

from typing import Any

from fastapi import FastAPI, HTTPException

from .loader import MODEL_ID, Model


def create_app(model: Model) -> FastAPI:
    app = FastAPI(title=MODEL_ID)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "model": MODEL_ID, "runtime": model.runtime}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"models": [{"name": MODEL_ID, "runtime": model.runtime}]}

    @app.post("/v1/systemone")
    def predict(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return model.predict(payload)
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
