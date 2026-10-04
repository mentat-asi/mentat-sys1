from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mentat_sys1.contracts import load_config
from mentat_sys1.release.package import (
    build_publication_bundle,
    migrate_checkpoint,
)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mentat-sys1")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    _add_runtime_arguments(audit)
    audit.add_argument("--train-data", required=True)
    audit.add_argument("--benchmark-dir", required=True)
    audit.add_argument("--ngram-review", required=True)
    serve = commands.add_parser("serve")
    _add_runtime_arguments(serve)
    serve.add_argument("--base-model-path", required=True)
    serve.add_argument("--model-dir", required=True)
    serve.add_argument("--device", default="cuda")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    calibrate = commands.add_parser("calibrate")
    _add_runtime_arguments(calibrate)
    calibrate.add_argument("--base-model-path", required=True)
    calibrate.add_argument("--calibration-data", required=True)
    calibrate.add_argument("--device", default="cuda")
    release = commands.add_parser("release")
    _add_runtime_arguments(release)
    release_source = release.add_mutually_exclusive_group(required=True)
    release_source.add_argument("--source-model-dir")
    release_source.add_argument("--publication-source-root")
    release.add_argument("--project-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "release":
        config = load_config(Path(args.config))
        if args.publication_source_root is not None:
            if args.project_root is None:
                parser.error("--project-root is required for a publication bundle")
            receipt = build_publication_bundle(
                config=config,
                source_artifact_root=Path(args.publication_source_root),
                project_root=Path(args.project_root),
                bundle_root=Path(args.artifact_root),
            )
        else:
            receipt = migrate_checkpoint(
                config=config,
                source_model_dir=Path(args.source_model_dir),
                artifact_root=Path(args.artifact_root),
            )
        print(json.dumps(receipt, sort_keys=True))
    elif args.command == "calibrate":
        from mentat_sys1.inference.calibration import calibrate_dataset

        config = load_config(Path(args.config))
        receipt = calibrate_dataset(
            config=config,
            calibration_data=Path(args.calibration_data),
            artifact_root=Path(args.artifact_root),
            model_dir=Path(args.artifact_root) / config.paths.model,
            base_model_path=Path(args.base_model_path),
            device=args.device,
        )
        print(json.dumps(receipt, sort_keys=True))
    elif args.command == "serve":
        import uvicorn

        from mentat_sys1.inference.api import create_app
        from mentat_sys1.inference.backend import PortableBackend

        backend = PortableBackend.load(
            config=load_config(Path(args.config)),
            model_dir=Path(args.model_dir),
            base_model_path=Path(args.base_model_path),
            device=args.device,
        )
        uvicorn.run(
            create_app(backend),
            host=args.host,
            port=args.port,
        )
    elif args.command == "audit":
        from mentat_sys1.audit.contamination import (
            REGISTERED_BENCHMARK_REVISION,
            REGISTERED_BENCHMARKS,
            REGISTERED_TRAIN,
            audit_jsonl_files,
            load_registered_ngram_review,
        )

        config = load_config(Path(args.config))
        benchmark_dir = Path(args.benchmark_dir)
        review_path = Path(args.ngram_review)
        report = audit_jsonl_files(
            train_path=Path(args.train_data),
            benchmark_paths={
                name: benchmark_dir / f"{name}.jsonl"
                for name in REGISTERED_BENCHMARKS
            },
            output_path=(
                Path(args.artifact_root)
                / config.paths.evidence
                / "contamination-report.json"
            ),
            benchmark_revision=REGISTERED_BENCHMARK_REVISION,
            expected_train=REGISTERED_TRAIN,
            expected_benchmarks=REGISTERED_BENCHMARKS,
            reviewed_ngram_allowlist=load_registered_ngram_review(review_path),
            review_path=review_path,
        )
        print(json.dumps(report, sort_keys=True))
    return 0
