from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from mentat_sys1.contracts import load_config
from mentat_sys1.release.verify import (
    assess_legacy_public,
    prediction_digest,
)

ROOT = Path(__file__).parents[2]


def test_legacy_public_gate_requires_registered_result_and_reload() -> None:
    assessment = assess_legacy_public(
        config=load_config(ROOT / "configs/mentat-sys1-v0.1.json"),
        tiers={
            "easy": {"correct": 48, "total": 48},
            "original": {"correct": 66, "total": 72},
            "hard": {"correct": 86, "total": 111},
        },
        invalid=0,
        severe_failures=0,
        second_process_match=True,
    )

    assert assessment["correct"] == 200
    assert assessment["total"] == 231
    assert assessment["passed"] is True


def test_legacy_public_gate_fails_closed_on_score_drift() -> None:
    with pytest.raises(ValueError, match="legacy-public result differs"):
        assess_legacy_public(
            config=load_config(ROOT / "configs/mentat-sys1-v0.1.json"),
            tiers={
                "easy": {"correct": 48, "total": 48},
                "original": {"correct": 65, "total": 72},
                "hard": {"correct": 86, "total": 111},
            },
            invalid=0,
            severe_failures=0,
            second_process_match=True,
        )


def test_v02_public_gate_requires_its_registered_result() -> None:
    assessment = assess_legacy_public(
        config=load_config(ROOT / "configs/mentat-sys1-v0.2.json"),
        tiers={
            "easy": {"correct": 48, "total": 48},
            "original": {"correct": 71, "total": 72},
            "hard": {"correct": 87, "total": 111},
        },
        invalid=0,
        severe_failures=0,
        second_process_match=True,
    )

    assert assessment["correct"] == 206
    assert assessment["total"] == 231
    assert assessment["passed"] is True


def test_prediction_digest_ignores_timing_but_detects_probability_drift() -> None:
    records = [
        {
            "task_id": "one",
            "status": "ok",
            "valid": True,
            "correct": True,
            "predicted": "a",
            "probs": {"a": 0.75, "b": 0.25},
            "latency_s": 1.0,
            "ts": 1.0,
        }
    ]
    second = deepcopy(records)
    second[0]["latency_s"] = 2.0
    second[0]["ts"] = 3.0

    assert prediction_digest(records) == prediction_digest(second)

    second[0]["probs"] = {"a": 0.74, "b": 0.26}
    assert prediction_digest(records) != prediction_digest(second)
