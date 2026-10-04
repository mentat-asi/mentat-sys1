from __future__ import annotations

import json
from pathlib import Path

import pytest

from mentat_sys1.audit.contamination import REGISTERED_BENCHMARK_REVISION
from mentat_sys1.audit.provenance import sha256_file
from mentat_sys1.contracts import ProjectConfig
from mentat_sys1.release.package import (
    PUBLICATION_DOCUMENTS,
    PUBLICATION_EVIDENCE_FILES,
    PUBLICATION_MODEL_FILES,
    assemble_calibrated_model,
    build_publication_bundle,
    migrate_checkpoint,
)
from mentat_sys1.release.verify import (
    CALIBRATION_SHA256,
    verify_publication_bundle,
)

ROOT = Path(__file__).parents[2]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_checkpoint(root: Path) -> Path:
    root.mkdir()
    _write_json(
        root / "adapter_config.json",
        {
            "base_model_name_or_path": "/data/private/Qwen3.5-4B",
            "revision": None,
            "r": 64,
            "lora_alpha": 128,
        },
    )
    (root / "adapter_model.safetensors").write_bytes(b"adapter-tensors")
    _write_json(
        root / "decision_readout.json",
        {
            "version": 1,
            "codes": [
                {"code": f"C{index}", "token_id": index} for index in range(256)
            ],
        },
    )
    (root / "decision_readout.safetensors").write_bytes(b"readout-tensors")
    return root


def _config_for(source: Path) -> ProjectConfig:
    raw = json.loads(
        (ROOT / "configs/mentat-sys1-v0.1.json").read_text(encoding="utf-8")
    )
    raw["release"]["adapter_sha256"] = sha256_file(source / "adapter_model.safetensors")
    raw["release"]["readout_sha256"] = sha256_file(
        source / "decision_readout.safetensors"
    )
    return ProjectConfig.model_validate(raw)


def _calibration_report(config: ProjectConfig) -> dict[str, object]:
    return {
        "schema_version": 1,
        "method": "scalar_temperature",
        "temperature": 0.97,
        "optimizer": "bounded_log_temperature",
        "objective": "negative_log_likelihood",
        "split": {"rows": 1120, "sha256": CALIBRATION_SHA256},
        "model": {
            "adapter_sha256": config.release.adapter_sha256,
            "readout_sha256": config.release.readout_sha256,
        },
        "before": {"ece": 0.03, "nll": 0.4, "brier": 0.2},
        "after": {"ece": 0.02, "nll": 0.39, "brier": 0.19},
        "predictions_unchanged": True,
        "scoring": {"rotation_mode": "rot4", "model_passes": 4204},
    }


def _publication_source(tmp_path: Path) -> tuple[ProjectConfig, Path]:
    checkpoint = _source_checkpoint(tmp_path / "source-checkpoint")
    config = _config_for(checkpoint)
    staging = tmp_path / "staging"
    migrate_checkpoint(
        config=config,
        source_model_dir=checkpoint,
        artifact_root=staging,
    )
    calibration_path = staging / "evidence/calibration-report.json"
    _write_json(calibration_path, _calibration_report(config))
    final = tmp_path / "final"
    assemble_calibrated_model(
        config=config,
        source_model_dir=staging / "model",
        calibration_report=calibration_path,
        artifact_root=final,
    )
    _write_json(
        final / "evidence/contamination-report.json",
        {
            "schema_version": 1,
            "status": "clean",
            "benchmark_revision": REGISTERED_BENCHMARK_REVISION,
            "train_rows": 87386,
            "benchmark_rows": 231,
            "matches": {
                "exact_ids": [],
                "derivative_ids": [],
                "source_groups": [],
                "image_hashes": [],
                "word_8grams": [],
            },
            "resolved_word_8grams": [
                {
                    "ngram_sha256": "a" * 64,
                    "occurrences": 2,
                    "review": "generic_question_template",
                }
            ],
        },
    )
    manifest_sha = sha256_file(final / "model/model-manifest.json")
    calibration_sha = sha256_file(final / "model/calibration.json")
    _write_json(
        final / "evidence/legacy-public-evaluation.json",
        {
            "schema_version": 1,
            "passed": True,
            "benchmark_revision": REGISTERED_BENCHMARK_REVISION,
            "model_manifest_sha256": manifest_sha,
            "calibration_sha256": calibration_sha,
            "correct": 200,
            "total": 231,
            "accuracy": 200 / 231,
            "tiers": {
                "easy": {"correct": 48, "total": 48},
                "original": {"correct": 66, "total": 72},
                "hard": {"correct": 86, "total": 111},
            },
            "invalid": 0,
            "severe_failures": 0,
            "second_process_match": True,
            "prediction_digest": "b" * 64,
            "runs": {
                "reload-1": {"rows": 231, "prediction_digest": "b" * 64},
                "reload-2": {"rows": 231, "prediction_digest": "b" * 64},
            },
        },
    )
    return config, final


def _project_docs(root: Path) -> Path:
    (root / "docs").mkdir(parents=True)
    (root / "README.md").write_text("# mentat-sys1-v0.1\n", encoding="utf-8")
    (root / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    (root / "NOTICE").write_text(
        "Historical attribution: Imajev, Apache-2.0.\n",
        encoding="utf-8",
    )
    for name in (
        "MODEL_CARD.md",
        "REPRODUCIBILITY.md",
        "JEVBENCH_SUBMISSION.md",
    ):
        (root / "docs" / name).write_text(
            f"# {name}\n\npublication_pending\n",
            encoding="utf-8",
        )
    return root


def test_publication_bundle_is_allowlisted_and_self_verifying(
    tmp_path: Path,
) -> None:
    config, source = _publication_source(tmp_path)
    project = _project_docs(tmp_path / "project")
    _write_json(
        source / "evaluation/raw-response.json",
        {"private_path": "/data/private/raw"},
    )
    _write_json(source / "evidence/not-for-publication.json", {"extra": True})
    bundle = tmp_path / "publication"

    receipt = build_publication_bundle(
        config=config,
        source_artifact_root=source,
        project_root=project,
        bundle_root=bundle,
    )

    expected = {
        *(f"model/{name}" for name in PUBLICATION_MODEL_FILES),
        *(f"evidence/{name}" for name in PUBLICATION_EVIDENCE_FILES),
        *PUBLICATION_DOCUMENTS,
        "release-manifest.json",
        "SHA256SUMS",
    }
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    assert "evaluation/raw-response.json" not in actual
    assert "evidence/not-for-publication.json" not in actual
    assert receipt["status"] == "publication_bundle_assembled"
    verified = verify_publication_bundle(config=config, bundle_root=bundle)
    assert verified["files"] == len(expected)
    assert verified["model"]["temperature"] == pytest.approx(0.97)


def test_publication_bundle_fails_closed_without_clean_evidence(
    tmp_path: Path,
) -> None:
    config, source = _publication_source(tmp_path)
    project = _project_docs(tmp_path / "project")
    contamination = source / "evidence/contamination-report.json"
    payload = json.loads(contamination.read_text(encoding="utf-8"))
    payload["matches"]["word_8grams"] = [{"ngram_sha256": "c" * 64}]
    _write_json(contamination, payload)
    bundle = tmp_path / "publication"

    with pytest.raises(ValueError, match="unresolved contamination"):
        build_publication_bundle(
            config=config,
            source_artifact_root=source,
            project_root=project,
            bundle_root=bundle,
        )

    assert not bundle.exists()
