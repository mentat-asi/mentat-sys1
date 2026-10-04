from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mentat_sys1.audit.contamination import (
    REGISTERED_BENCHMARK_REVISION,
    REGISTERED_TRAIN,
    audit_jsonl_files,
    load_registered_ngram_review,
)
from mentat_sys1.audit.provenance import sha256_file
from mentat_sys1.cli import main

ROOT = Path(__file__).parents[2]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_file_audit_binds_input_hashes_and_writes_immutable_report(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    easy = tmp_path / "benchmark/easy.jsonl"
    _write_jsonl(
        train,
        [{"id": "train-1", "request": {"state": "short train example"}}],
    )
    _write_jsonl(
        easy,
        [{"id": "easy-1", "state": "separate public benchmark example"}],
    )
    output = tmp_path / "evidence/contamination-report.json"
    review = tmp_path / "review.json"
    review.write_text("{}\n", encoding="utf-8")

    report = audit_jsonl_files(
        train_path=train,
        benchmark_paths={"easy": easy},
        output_path=output,
        benchmark_revision="abc123",
        expected_train=(1, sha256_file(train)),
        expected_benchmarks={"easy": (1, sha256_file(easy))},
        review_path=review,
    )

    assert report["status"] == "clean"
    assert report["benchmark_revision"] == "abc123"
    assert report["inputs"]["train"] == {
        "file": "train.jsonl",
        "rows": 1,
        "sha256": sha256_file(train),
    }
    assert report["inputs"]["ngram_review"]["sha256"] == sha256_file(review)
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_file_audit_rejects_unregistered_input_before_writing(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    easy = tmp_path / "easy.jsonl"
    _write_jsonl(train, [{"id": "train-1", "state": "train"}])
    _write_jsonl(easy, [{"id": "easy-1", "state": "benchmark"}])
    output = tmp_path / "report.json"

    with pytest.raises(ValueError, match="train data identity differs"):
        audit_jsonl_files(
            train_path=train,
            benchmark_paths={"easy": easy},
            output_path=output,
            benchmark_revision="abc123",
            expected_train=(2, "0" * 64),
            expected_benchmarks={"easy": (1, sha256_file(easy))},
        )

    assert not output.exists()


def test_registered_ngram_review_is_bound_to_frozen_inputs() -> None:
    review = load_registered_ngram_review(
        ROOT / "data/jevbench-public-8gram-review.json"
    )

    assert len(review) == 24
    assert all(len(digest) == 64 for digest in review)
    assert REGISTERED_TRAIN[0] == 87386
    assert REGISTERED_BENCHMARK_REVISION.startswith("2fa63fa")


def test_audit_command_uses_registered_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: Any,
) -> None:
    observed: dict[str, object] = {}

    def fake_load_review(path: Path) -> dict[str, str]:
        observed["review_path"] = path
        return {"a" * 64: "reviewed"}

    def fake_audit(**kwargs: object) -> dict[str, object]:
        observed["audit"] = kwargs
        return {"status": "clean"}

    monkeypatch.setattr(
        "mentat_sys1.audit.contamination.load_registered_ngram_review",
        fake_load_review,
    )
    monkeypatch.setattr(
        "mentat_sys1.audit.contamination.audit_jsonl_files",
        fake_audit,
    )

    result = main(
        [
            "audit",
            "--config",
            str(ROOT / "configs/mentat-sys1-v0.1.json"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--train-data",
            str(tmp_path / "train.jsonl"),
            "--benchmark-dir",
            str(tmp_path / "benchmark"),
            "--ngram-review",
            str(tmp_path / "review.json"),
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {"status": "clean"}
    audit = observed["audit"]
    assert isinstance(audit, dict)
    assert audit["expected_train"] == REGISTERED_TRAIN
    assert audit["benchmark_revision"] == REGISTERED_BENCHMARK_REVISION
    assert audit["output_path"] == (
        tmp_path / "artifacts/evidence/contamination-report.json"
    )
