"""Exact contamination checks for model-visible training material."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mentat_sys1.audit.provenance import sha256_file, write_immutable_json

VISIBLE_KEYS = frozenset(
    {
        "state",
        "question",
        "instructions",
        "criteria",
        "options",
        "description",
        "rationale",
    }
)
DERIVATIVE_ID_KEYS = frozenset(
    {"parent_id", "variant_of", "derived_from", "source_id", "original_id"}
)
SOURCE_GROUP_KEYS = frozenset({"source_group", "source_group_id", "group"})
IMAGE_HASH_KEYS = frozenset(
    {"image_sha256", "image_sha256s", "image_hash", "image_hashes"}
)
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REGISTERED_TRAIN = (
    87386,
    "50795aace3c26c65ed6229f0a18bc9bb2cff76af1e68efac21e203d78114519d",
)
REGISTERED_BENCHMARK_REVISION = "2fa63fa3226cb369795525ed011800f57dcbd894"
REGISTERED_BENCHMARKS = {
    "easy": (
        48,
        "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    ),
    "hard": (
        111,
        "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
    ),
    "original": (
        72,
        "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    ),
}


def word_ngrams(text: str, *, size: int = 8) -> set[tuple[str, ...]]:
    if size < 1:
        raise ValueError("n-gram size must be positive")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = WORD_PATTERN.findall(normalized)
    return {
        tuple(words[index : index + size])
        for index in range(len(words) - size + 1)
    }


def _string_leaves(value: object) -> Iterable[str]:
    if isinstance(value, str):
        if value:
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_leaves(item)


def _model_visible_strings(value: object, *, visible: bool = False) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if visible or key in VISIBLE_KEYS:
                yield from _string_leaves(item)
            else:
                yield from _model_visible_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _model_visible_strings(item, visible=visible)
    elif visible and isinstance(value, str) and value:
        yield value


def _values_for_keys(value: object, keys: frozenset[str]) -> set[str]:
    observed: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys:
                observed.update(_string_leaves(item))
            observed.update(_values_for_keys(item, keys))
    elif isinstance(value, list):
        for item in value:
            observed.update(_values_for_keys(item, keys))
    return observed


def _row_id(row: dict[str, object], *, side: str) -> str:
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"{side} row ID must be non-empty")
    return identifier


def _identities(
    rows: Sequence[dict[str, object]],
    *,
    side: str,
) -> tuple[dict[str, dict[str, object]], set[str], set[str], set[str]]:
    indexed: dict[str, dict[str, object]] = {}
    derivative_ids: set[str] = set()
    source_groups: set[str] = set()
    image_hashes: set[str] = set()
    for row in rows:
        identifier = _row_id(row, side=side)
        if identifier in indexed:
            raise ValueError(f"{side} row IDs must be unique")
        indexed[identifier] = row
        derivative_ids.update(_values_for_keys(row, DERIVATIVE_ID_KEYS))
        source_groups.update(_values_for_keys(row, SOURCE_GROUP_KEYS))
        image_hashes.update(
            value.casefold()
            for value in _values_for_keys(row, IMAGE_HASH_KEYS)
            if SHA256_PATTERN.fullmatch(value.casefold())
        )
    return indexed, derivative_ids, source_groups, image_hashes


def _ngram_matches(
    train_rows: Sequence[dict[str, object]],
    benchmark_rows: Sequence[dict[str, object]],
) -> list[dict[str, str]]:
    benchmark_index: dict[tuple[str, ...], set[str]] = {}
    for row in benchmark_rows:
        identifier = _row_id(row, side="benchmark")
        ngrams = {
            ngram
            for text in _model_visible_strings(row)
            for ngram in word_ngrams(text)
        }
        for ngram in ngrams:
            benchmark_index.setdefault(ngram, set()).add(identifier)

    matches: set[tuple[str, str, str]] = set()
    for row in train_rows:
        train_id = _row_id(row, side="train")
        ngrams = {
            ngram
            for text in _model_visible_strings(row)
            for ngram in word_ngrams(text)
        }
        for ngram in ngrams.intersection(benchmark_index):
            digest = hashlib.sha256("\0".join(ngram).encode("utf-8")).hexdigest()
            for benchmark_id in benchmark_index[ngram]:
                matches.add((train_id, benchmark_id, digest))
    return [
        {
            "train_id": train_id,
            "benchmark_id": benchmark_id,
            "ngram_sha256": digest,
        }
        for train_id, benchmark_id, digest in sorted(matches)
    ]


def contamination_report(
    *,
    train_rows: Sequence[dict[str, object]],
    benchmark_rows: Sequence[dict[str, object]],
    reviewed_ngram_allowlist: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    train, train_derivatives, train_groups, train_images = _identities(
        train_rows,
        side="train",
    )
    benchmark, benchmark_derivatives, benchmark_groups, benchmark_images = (
        _identities(benchmark_rows, side="benchmark")
    )
    exact_ids = sorted(set(train).intersection(benchmark))
    derivative_ids = sorted(
        train_derivatives.intersection(benchmark)
        | benchmark_derivatives.intersection(train)
    )
    source_groups = sorted(train_groups.intersection(benchmark_groups))
    image_hashes = sorted(train_images.intersection(benchmark_images))
    ngram_matches = _ngram_matches(train_rows, benchmark_rows)
    allowlist = dict(reviewed_ngram_allowlist or {})
    if any(
        not SHA256_PATTERN.fullmatch(digest)
        or not isinstance(review, str)
        or not review.strip()
        for digest, review in allowlist.items()
    ):
        raise ValueError("reviewed n-gram allowlist is invalid")
    observed_ngram_hashes = {
        match["ngram_sha256"] for match in ngram_matches
    }
    unused = set(allowlist) - observed_ngram_hashes
    if unused:
        raise ValueError("reviewed n-gram allowlist contains unused hashes")
    unresolved_ngrams = [
        match
        for match in ngram_matches
        if match["ngram_sha256"] not in allowlist
    ]
    resolved_ngrams = [
        {
            "ngram_sha256": digest,
            "occurrences": sum(
                match["ngram_sha256"] == digest for match in ngram_matches
            ),
            "review": allowlist[digest],
        }
        for digest in sorted(allowlist)
    ]
    matches = {
        "exact_ids": exact_ids,
        "derivative_ids": derivative_ids,
        "source_groups": source_groups,
        "image_hashes": image_hashes,
        "word_8grams": unresolved_ngrams,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "clean" if not any(matches.values()) else "blocked",
        "ngram_size": 8,
        "train_rows": len(train_rows),
        "benchmark_rows": len(benchmark_rows),
        "matches": matches,
        "resolved_word_8grams": resolved_ngrams,
    }
    return report


def audit_contamination(
    *,
    train_rows: Sequence[dict[str, object]],
    benchmark_rows: Sequence[dict[str, object]],
    reviewed_ngram_allowlist: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    report = contamination_report(
        train_rows=train_rows,
        benchmark_rows=benchmark_rows,
        reviewed_ngram_allowlist=reviewed_ngram_allowlist,
    )
    matches = report["matches"]
    exact_ids = matches["exact_ids"]
    derivative_ids = matches["derivative_ids"]
    source_groups = matches["source_groups"]
    image_hashes = matches["image_hashes"]
    ngram_matches = matches["word_8grams"]
    if exact_ids:
        raise ValueError(f"{len(exact_ids)} unresolved exact ID matches")
    if derivative_ids:
        raise ValueError(f"{len(derivative_ids)} unresolved derivative ID matches")
    if source_groups:
        raise ValueError(f"{len(source_groups)} unresolved source-group matches")
    if image_hashes:
        raise ValueError(f"{len(image_hashes)} unresolved image hash matches")
    if ngram_matches:
        raise ValueError(f"{len(ngram_matches)} unresolved 8-gram matches")
    return report


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path.name}:{line_number} must be an object")
            rows.append(payload)
    return rows


def load_registered_ngram_review(path: Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported n-gram review schema")
    if payload.get("ngram_size") != 8:
        raise ValueError("n-gram review size differs")
    train = payload.get("train")
    if train != {"rows": REGISTERED_TRAIN[0], "sha256": REGISTERED_TRAIN[1]}:
        raise ValueError("n-gram review train identity differs")
    benchmark = payload.get("benchmark")
    expected_benchmark = {
        "revision": REGISTERED_BENCHMARK_REVISION,
        "rows": sum(identity[0] for identity in REGISTERED_BENCHMARKS.values()),
        "splits": {
            name: identity[1]
            for name, identity in sorted(REGISTERED_BENCHMARKS.items())
        },
    }
    if benchmark != expected_benchmark:
        raise ValueError("n-gram review benchmark identity differs")
    adjudication = payload.get("adjudication")
    if not isinstance(adjudication, dict) or (
        adjudication.get("detected_occurrences") != 182
        or adjudication.get("unique_hashes") != 24
        or adjudication.get("maximum_contiguous_overlap_words") != 9
        or adjudication.get("overlap_10gram_count") != 0
    ):
        raise ValueError("n-gram review adjudication differs")
    reviewed = payload.get("reviewed")
    if not isinstance(reviewed, list) or len(reviewed) != 24:
        raise ValueError("n-gram review entries differ")
    allowlist: dict[str, str] = {}
    for entry in reviewed:
        if not isinstance(entry, dict):
            raise ValueError("n-gram review entry must be an object")
        digest = entry.get("ngram_sha256")
        classification = entry.get("classification")
        if (
            not isinstance(digest, str)
            or not SHA256_PATTERN.fullmatch(digest)
            or not isinstance(classification, str)
            or not classification
            or digest in allowlist
        ):
            raise ValueError("n-gram review entry is invalid")
        allowlist[digest] = classification
    return allowlist


def _verified_rows(
    path: Path,
    expected: tuple[int, str],
    *,
    label: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = _load_jsonl(path)
    observed = (len(rows), sha256_file(path))
    if observed != expected:
        raise ValueError(f"{label} identity differs")
    return rows, {
        "file": path.name,
        "rows": observed[0],
        "sha256": observed[1],
    }


def audit_jsonl_files(
    *,
    train_path: Path,
    benchmark_paths: Mapping[str, Path],
    output_path: Path,
    benchmark_revision: str,
    expected_train: tuple[int, str],
    expected_benchmarks: Mapping[str, tuple[int, str]],
    reviewed_ngram_allowlist: Mapping[str, str] | None = None,
    review_path: Path | None = None,
) -> dict[str, Any]:
    if not benchmark_revision:
        raise ValueError("benchmark revision must be non-empty")
    if set(benchmark_paths) != set(expected_benchmarks):
        raise ValueError("benchmark split set differs")

    train_rows, train_identity = _verified_rows(
        Path(train_path),
        expected_train,
        label="train data",
    )
    benchmark_rows: list[dict[str, object]] = []
    benchmark_identities: dict[str, dict[str, object]] = {}
    for name in sorted(benchmark_paths):
        rows, identity = _verified_rows(
            Path(benchmark_paths[name]),
            expected_benchmarks[name],
            label=f"benchmark {name}",
        )
        benchmark_rows.extend(rows)
        benchmark_identities[name] = identity

    report = audit_contamination(
        train_rows=train_rows,
        benchmark_rows=benchmark_rows,
        reviewed_ngram_allowlist=reviewed_ngram_allowlist,
    )
    report["benchmark_revision"] = benchmark_revision
    report["inputs"] = {
        "train": train_identity,
        "benchmarks": benchmark_identities,
    }
    if review_path is not None:
        review = Path(review_path)
        if not review.is_file():
            raise ValueError("n-gram review file is missing")
        report["inputs"]["ngram_review"] = {
            "file": review.name,
            "sha256": sha256_file(review),
        }
    write_immutable_json(Path(output_path), report)
    return report
