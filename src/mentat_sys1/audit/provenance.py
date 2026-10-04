"""Deterministic hashes and immutable receipt writers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path

DEFAULT_IGNORED_PARTS = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(
    root: Path,
    ignored: frozenset[str] = DEFAULT_IGNORED_PARTS,
) -> str:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise ValueError(f"tree root is not a directory: {resolved}")
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file()
        and not any(part in ignored for part in path.relative_to(resolved).parts)
    )
    for path in files:
        relative = path.relative_to(resolved).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_immutable(path: Path, serialized: str, *, kind: str) -> None:
    destination = Path(path)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"conflicting immutable {kind}: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def write_immutable_json(path: Path, payload: object) -> str:
    serialized = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    _write_immutable(path, serialized, kind="JSON")
    return sha256_file(path)


def write_immutable_jsonl(
    path: Path,
    rows: Iterable[object],
) -> str:
    serialized = "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    _write_immutable(path, serialized, kind="JSONL")
    return sha256_file(path)
