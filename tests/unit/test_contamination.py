from __future__ import annotations

import hashlib

import pytest

from mentat_sys1.audit.contamination import (
    audit_contamination,
    contamination_report,
    word_ngrams,
)


def test_word_ngrams_normalize_unicode_case_and_whitespace() -> None:
    text = "ＡLPHA  beta\nGamma delta epsilon zeta eta theta"

    assert word_ngrams(text, size=8) == {
        ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta")
    }


def test_any_exact_benchmark_8gram_blocks_release() -> None:
    train = [
        {
            "id": "train-1",
            "request": {
                "state": "one two three four five six seven eight nine",
            },
        }
    ]
    benchmark = [
        {
            "id": "bench-1",
            "question": "zero one two three four five six seven eight",
        }
    ]

    with pytest.raises(ValueError, match="unresolved 8-gram"):
        audit_contamination(train_rows=train, benchmark_rows=benchmark)

    report = contamination_report(
        train_rows=train,
        benchmark_rows=benchmark,
    )
    assert report["status"] == "blocked"
    assert len(report["matches"]["word_8grams"]) == 1


@pytest.mark.parametrize(
    ("train", "benchmark", "message"),
    [
        (
            [{"id": "same"}],
            [{"id": "same"}],
            "exact ID",
        ),
        (
            [{"id": "train", "source_group": "shared"}],
            [{"id": "bench", "group": "shared"}],
            "source-group",
        ),
        (
            [{"id": "train", "image_sha256": "a" * 64}],
            [{"id": "bench", "image_sha256": "a" * 64}],
            "image hash",
        ),
        (
            [{"id": "train", "parent_id": "bench"}],
            [{"id": "bench"}],
            "derivative ID",
        ),
    ],
)
def test_exact_identity_overlap_blocks_release(
    train: list[dict[str, object]],
    benchmark: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        audit_contamination(train_rows=train, benchmark_rows=benchmark)


def test_clean_audit_returns_aggregate_safe_report() -> None:
    report = audit_contamination(
        train_rows=[
            {
                "id": "train-1",
                "source_group": "train-group",
                "request": {"state": "alpha beta gamma"},
            }
        ],
        benchmark_rows=[
            {
                "id": "bench-1",
                "group": "bench-group",
                "state": "delta epsilon zeta",
            }
        ],
    )

    assert report["status"] == "clean"
    assert report["train_rows"] == 1
    assert report["benchmark_rows"] == 1
    assert report["matches"] == {
        "exact_ids": [],
        "derivative_ids": [],
        "source_groups": [],
        "image_hashes": [],
        "word_8grams": [],
    }


def test_reviewed_exact_ngram_hash_resolves_only_that_overlap() -> None:
    phrase = ("one", "two", "three", "four", "five", "six", "seven", "eight")
    digest = hashlib.sha256("\0".join(phrase).encode("utf-8")).hexdigest()
    train = [{"id": "train-1", "state": " ".join(phrase)}]
    benchmark = [{"id": "bench-1", "state": "zero " + " ".join(phrase)}]

    report = audit_contamination(
        train_rows=train,
        benchmark_rows=benchmark,
        reviewed_ngram_allowlist={digest: "generic fixed phrase"},
    )

    assert report["status"] == "clean"
    assert report["matches"]["word_8grams"] == []
    assert report["resolved_word_8grams"] == [
        {
            "ngram_sha256": digest,
            "occurrences": 1,
            "review": "generic fixed phrase",
        }
    ]
