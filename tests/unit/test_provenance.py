import json
from pathlib import Path

import pytest

from mentat_sys1.audit.provenance import (
    hash_tree,
    sha256_file,
    write_immutable_json,
    write_immutable_jsonl,
)


def test_sha256_file_and_tree_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "nested" / "b.txt"
    second.parent.mkdir()
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")

    assert sha256_file(first) == (
        "b6a98d9ce9a2d9149288fa3df42d377c3e42737afdcdaf714e33c0a100b51060"
    )
    observed = hash_tree(tmp_path)
    assert observed == hash_tree(tmp_path)

    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "ignored").write_text("noise")
    assert hash_tree(tmp_path) == observed


def test_immutable_json_is_idempotent_and_conflict_closed(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    write_immutable_json(path, {"schema_version": 1, "value": 2})
    digest = sha256_file(path)

    write_immutable_json(path, {"schema_version": 1, "value": 2})
    assert sha256_file(path) == digest
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 2

    with pytest.raises(ValueError, match="conflicting immutable JSON"):
        write_immutable_json(path, {"schema_version": 1, "value": 3})


def test_immutable_jsonl_returns_hash_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"id": "b", "value": 2}, {"id": "a", "value": 1}]

    observed = write_immutable_jsonl(path, rows)

    assert observed == sha256_file(path)
    assert path.read_text(encoding="utf-8") == (
        '{"id":"b","value":2}\n{"id":"a","value":1}\n'
    )
    with pytest.raises(ValueError, match="conflicting immutable JSONL"):
        write_immutable_jsonl(path, list(reversed(rows)))


def test_non_finite_json_is_rejected_without_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"

    with pytest.raises(ValueError):
        write_immutable_json(path, {"value": float("nan")})

    assert not path.exists()
