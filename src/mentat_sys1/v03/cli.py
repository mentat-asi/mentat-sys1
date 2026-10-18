"""v0.3 CLI; help and checksum verification do not import CUDA libraries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loader import load_model, verify_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mentat-sys1-v03")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--model-package", type=Path, required=True)
    for command in ["predict", "serve"]:
        child = commands.add_parser(command)
        child.add_argument("--model-package", type=Path, required=True)
        child.add_argument("--base-model", type=Path, required=True)
        child.add_argument(
            "--runtime", choices=["native", "accelerated"], default="accelerated"
        )
        child.add_argument("--device", default="cuda:0")
        child.add_argument("--threads", type=int, default=4)
        child.add_argument("--no-cuda-graphs", action="store_true")
        child.add_argument("--max-graphs", type=int, default=8)
        child.add_argument("--max-graph-tokens", type=int, default=4096)
        if command == "predict":
            child.add_argument("--input", type=Path, required=True)
        else:
            child.add_argument("--host", default="127.0.0.1")
            child.add_argument("--port", type=int, default=8765)
            child.add_argument(
                "--warmup-request", type=Path, action="append", default=[]
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        verify_package(args.model_package)
        print(json.dumps({"verified": True, "model": "mentat-sys1-v0.3"}))
        return 0
    model = load_model(
        model_package=args.model_package,
        base_model=args.base_model,
        runtime=args.runtime,
        device=args.device,
        threads=args.threads,
        cuda_graphs=not args.no_cuda_graphs,
        max_graphs=args.max_graphs,
        max_graph_tokens=args.max_graph_tokens,
    )
    if args.command == "predict":
        payload = json.loads(args.input.read_text())
        print(
            json.dumps(
                model.predict(payload), ensure_ascii=False, indent=2, allow_nan=False
            )
        )
    else:
        import uvicorn

        from .api import create_app

        model.warm([json.loads(path.read_text()) for path in args.warmup_request])
        uvicorn.run(create_app(model), host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
