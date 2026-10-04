from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mentat_sys1.audit.provenance import sha256_file
from mentat_sys1.cli import build_parser
from mentat_sys1.contracts import load_config
from mentat_sys1.inference.calibration import calibrate_dataset
from mentat_sys1.inference.contracts import DecisionRequest
from mentat_sys1.inference.scoring import candidates, result_from_logits

ROOT = Path(__file__).parents[2]


class FakeBackend:
    def score(
        self,
        images: list[Any],
        request: DecisionRequest,
    ) -> tuple[list[Any], dict[str, object]]:
        assert images == []
        field = request.fields[0]
        logits = [3.0] + [0.0] * (len(candidates(field)) - 1)
        return [result_from_logits(candidates(field), logits)], {
            "language_model_passes": 1,
            "rotations": 1,
        }


def _row(identifier: str, target: str) -> dict[str, object]:
    return {
        "id": identifier,
        "images": [],
        "request": {
            "schema_version": "1.0",
            "request_id": identifier,
            "state": {},
            "fields": [
                {
                    "id": "decision",
                    "type": "choice",
                    "question": "Pick one.",
                    "options": [
                        {"value": "a"},
                        {"value": "b"},
                    ],
                }
            ],
        },
        "target": target,
    }


def test_calibration_command_requires_explicit_model_and_data_paths() -> None:
    args = build_parser().parse_args(
        [
            "calibrate",
            "--config",
            "config.json",
            "--artifact-root",
            "artifacts",
            "--base-model-path",
            "base",
            "--calibration-data",
            "calibration.jsonl",
        ]
    )

    assert args.base_model_path == "base"
    assert args.calibration_data == "calibration.jsonl"
    assert args.device == "cuda"


def test_calibration_dataset_writes_bound_immutable_report(tmp_path: Path) -> None:
    calibration_data = tmp_path / "calibration.jsonl"
    rows = [_row("one", "a"), _row("two", "b"), _row("three", "a")]
    calibration_data.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"

    report = calibrate_dataset(
        config=load_config(ROOT / "configs/mentat-sys1-v0.1.json"),
        calibration_data=calibration_data,
        artifact_root=artifact_root,
        backend=FakeBackend(),
        expected_rows=3,
        expected_sha256=sha256_file(calibration_data),
    )

    assert report["split"] == {
        "rows": 3,
        "sha256": sha256_file(calibration_data),
    }
    assert report["model"]["adapter_sha256"].startswith("c7268724")
    assert report["predictions_unchanged"] is True
    assert report["after"]["nll"] <= report["before"]["nll"]
    assert report["scoring"] == {
        "rotation_mode": "single",
        "rotations": 1,
        "model_passes": 3,
    }
    written = json.loads(
        (artifact_root / "evidence/calibration-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert written == report
