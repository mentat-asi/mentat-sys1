from __future__ import annotations

import pytest

from mentat_sys1.inference.calibration import load_temperature_calibration


def test_scalar_calibration_expands_to_all_runtime_types() -> None:
    loaded = load_temperature_calibration(
        {
            "schema_version": 1,
            "method": "scalar_temperature",
            "temperature": 1.0551417227266353,
        },
        digest="a" * 64,
    )

    assert loaded.pooled_temperature == pytest.approx(1.0551417227266353)
    assert loaded.temperature_by_type == {
        "choice": pytest.approx(1.0551417227266353),
        "noul": pytest.approx(1.0551417227266353),
        "score": pytest.approx(1.0551417227266353),
    }
    assert loaded.version == "scalar_temperature-v1:aaaaaaaaaaaa"


def test_v02_calibration_loads_exact_type_temperatures() -> None:
    loaded = load_temperature_calibration(
        {
            "schema_version": 2,
            "method": "temperature_by_type",
            "pooled_temperature": 1.5758800892767102,
            "temperature_by_type": {
                "choice": 1.609674650339169,
                "noul": 1.3462360767794004,
                "score": 1.6476480715733943,
            },
        },
        digest="b" * 64,
    )

    assert loaded.pooled_temperature == pytest.approx(1.5758800892767102)
    assert loaded.temperature_by_type == {
        "choice": pytest.approx(1.609674650339169),
        "noul": pytest.approx(1.3462360767794004),
        "score": pytest.approx(1.6476480715733943),
    }
    assert loaded.version == "temperature_by_type-v2:bbbbbbbbbbbb"


@pytest.mark.parametrize(
    "temperatures",
    [
        {"choice": 1.0, "noul": 1.0},
        {"choice": 1.0, "noul": 1.0, "score": 0.0},
        {"choice": 1.0, "noul": 1.0, "score": float("inf")},
        {"choice": 1.0, "noul": 1.0, "score": 1.0, "extra": 1.0},
    ],
)
def test_v02_calibration_rejects_invalid_type_map(
    temperatures: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        load_temperature_calibration(
            {
                "schema_version": 2,
                "method": "temperature_by_type",
                "pooled_temperature": 1.0,
                "temperature_by_type": temperatures,
            },
            digest="c" * 64,
        )


@pytest.mark.parametrize("value", [True, 0.0, -1.0, float("nan")])
def test_calibration_rejects_invalid_pooled_temperature(value: object) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        load_temperature_calibration(
            {
                "schema_version": 2,
                "method": "temperature_by_type",
                "pooled_temperature": value,
                "temperature_by_type": {
                    "choice": 1.0,
                    "noul": 1.0,
                    "score": 1.0,
                },
            },
            digest="d" * 64,
        )


def test_calibration_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unsupported calibration schema"):
        load_temperature_calibration(
            {
                "schema_version": 3,
                "method": "temperature_by_type",
            },
            digest="e" * 64,
        )
