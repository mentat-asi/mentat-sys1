"""Predict and serve Spice adapters without changing legacy model commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEMO_REQUEST = {
    "state": "A delivery address is missing its street number.",
    "questions": {"complete": {
        "type": "noul", "instructions": "The delivery address is complete."
    }},
}


def load_model(args):
    if not args.model_package or not args.base_model:
        from .model import DecisionModel
        return DecisionModel.from_pretrained(
            args.model_package or args.model, runtime=args.runtime,
            device=args.device, base_model=args.base_model,
            revision=args.revision, cache_dir=args.cache_dir,
            local_files_only=args.offline,
        )
    if args.runtime == "turbo":
        from .turbo import TurboDecisionModel as model_class
    elif args.runtime == "accelerated":
        from .accelerated import AcceleratedDecisionModel as model_class
    else:
        from .model import DecisionModel as model_class
    return model_class(
        args.base_model, str(Path(args.model_package) / "model"), device=args.device
    )


def create_app(model):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="mentat-Spice decision API")

    @app.get("/health")
    async def health():
        return {"status": "ready", "model": model.model_id}

    # Resolve the locally imported type before FastAPI inspects annotations.
    async def systemone(request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 1_048_576:
                return JSONResponse({"error": "Request exceeds 1 MiB"}, status_code=413)
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("Request must be an object")
            if "images" in payload:
                raise ValueError("Use the Python interface for image input")
            return await run_in_threadpool(model.predict, payload)
        except (ValueError, TypeError, UnicodeError) as error:
            return JSONResponse({"error": str(error)}, status_code=422)

    systemone.__annotations__["request"] = Request
    app.post("/v1/systemone")(systemone)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("predict", "serve"):
        sub = commands.add_parser(command)
        sub.add_argument("--base-model", help="Existing local backbone; otherwise download the pinned base")
        sub.add_argument("--model-package", help="Existing local adapter package; otherwise download from HF")
        sub.add_argument("--model", default="yunqu/mentat-Spice-4B")
        sub.add_argument("--revision", help="Adapter Hub revision (the backbone is always pinned)")
        sub.add_argument("--cache-dir", help="Hugging Face cache directory")
        sub.add_argument("--offline", action="store_true", help="Only use existing local/cached model files")
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--runtime", choices=("native", "accelerated", "turbo"), default="native")
        if command == "predict":
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument("--input", type=Path)
            source.add_argument("--demo", action="store_true", help="Run a built-in address-completeness decision")
            sub.add_argument("--image", action="append", default=[])
        else:
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.command == "predict":
        try:
            payload = DEMO_REQUEST if args.demo else json.loads(args.input.read_text())
            if not isinstance(payload, dict):
                raise ValueError("Request must be an object")
        except (OSError, ValueError) as error:
            parser.error(str(error))
    model = load_model(args)
    if args.command == "predict":
        result = model.predict(payload, **({"images": args.image} if args.image else {}))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    else:
        import uvicorn
        uvicorn.run(create_app(model), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
