from __future__ import annotations

import numpy as np
import pytest

from mentat_sys1.inference.calibration import fit_temperature, metrics


def test_scalar_temperature_is_positive_and_does_not_change_predictions() -> None:
    logits = np.array([[5.0, 0.0], [0.2, 0.1], [0.0, 3.0]])
    targets = np.array([0, 1, 1])

    before = metrics(logits, targets, temperature=1.0)
    result = fit_temperature(logits, targets)
    after = metrics(logits, targets, temperature=result.temperature)

    assert np.isfinite(result.temperature) and result.temperature > 0
    assert np.array_equal(
        logits.argmax(1),
        (logits / result.temperature).argmax(1),
    )
    assert after.nll <= before.nll + 1e-12
    assert result.before == before
    assert result.after == after


def test_metrics_support_variable_candidate_counts() -> None:
    logits = [[3.0, 0.0], [0.0, 1.0, 4.0], [2.0, -1.0, 0.0, 1.0]]
    targets = [0, 2, 3]

    result = metrics(logits, targets, temperature=2.0)

    assert result.nll >= 0.0
    assert 0.0 <= result.ece <= 1.0
    assert 0.0 <= result.brier <= 2.0


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf")])
def test_metrics_reject_non_positive_or_non_finite_temperature(
    temperature: float,
) -> None:
    with pytest.raises(ValueError, match="temperature"):
        metrics([[1.0, 0.0]], [0], temperature=temperature)


def test_metrics_reject_invalid_targets_and_logits() -> None:
    with pytest.raises(ValueError, match="target"):
        metrics([[1.0, 0.0]], [2], temperature=1.0)
    with pytest.raises(ValueError, match="finite"):
        metrics([[1.0, float("nan")]], [0], temperature=1.0)
