import pytest
from pydantic import ValidationError

from mentat_sys1.inference.scoring import (
    to_decision_request,
    to_systemone_response,
)

REQUEST = {
    "model": "mentat-sys1-v0.1",
    "state": {"listing": {"color": "blue"}},
    "questions": {
        "matches": {
            "type": "noul",
            "instructions": "Does the garment match?",
            "criteria": {"true": "It matches.", "false": "It differs."},
        },
        "category": {
            "type": "choice",
            "instructions": "Which category?",
            "criteria": {"shirt": None, "shoe": None},
        },
        "quality": {
            "type": "score",
            "instructions": "Rate quality.",
            "criteria": ["poor", "good", "excellent"],
        },
    },
}


def test_systemone_request_converts_to_typed_internal_fields() -> None:
    request = to_decision_request(REQUEST, expected_model="mentat-sys1-v0.1")

    assert [field.type for field in request.fields] == [
        "boolean",
        "choice",
        "ordinal",
    ]
    assert request.fields[1].id == "category"
    assert [option.value for option in request.fields[1].options] == [
        "shirt",
        "shoe",
    ]


def test_systemone_request_rejects_model_mismatch_and_extra_fields() -> None:
    with pytest.raises(ValueError, match="model"):
        to_decision_request(
            {**REQUEST, "model": "another-model"},
            expected_model="mentat-sys1-v0.1",
        )
    with pytest.raises(ValueError, match="unsupported request field"):
        to_decision_request(
            {**REQUEST, "private": "value"},
            expected_model="mentat-sys1-v0.1",
        )


def test_systemone_request_enforces_question_and_option_limits() -> None:
    too_many_questions = {
        "model": "mentat-sys1-v0.1",
        "questions": {
            f"q{index}": {"type": "noul", "instructions": "Decide."}
            for index in range(9)
        },
    }
    with pytest.raises(ValueError, match="at most 8"):
        to_decision_request(
            too_many_questions,
            expected_model="mentat-sys1-v0.1",
        )
    too_many_options = {
        "model": "mentat-sys1-v0.1",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Pick.",
                "criteria": {str(index): None for index in range(256)},
            }
        },
    }
    with pytest.raises((ValueError, ValidationError), match="255"):
        to_decision_request(too_many_options, expected_model="mentat-sys1-v0.1")


def test_response_matches_typesafe_shapes() -> None:
    request = to_decision_request(REQUEST, expected_model="mentat-sys1-v0.1")
    from mentat_sys1.inference.scoring import result_from_logits

    results = [
        result_from_logits(
            [(True, None), (False, None), ("__unknown__", None)],
            [3.0, 0.0, -1.0],
        ),
        result_from_logits(
            [("shirt", None), ("shoe", None), ("__unknown__", None)],
            [3.0, 0.0, -1.0],
        ),
        result_from_logits(
            [(0, "poor"), (1, "good"), (2, "excellent"), ("__unknown__", None)],
            [0.0, 1.0, 3.0, -1.0],
        ),
    ]

    response = to_systemone_response(
        request,
        results,
        model="mentat-sys1-v0.1",
    )

    assert response["model"] == "mentat-sys1-v0.1"
    assert response["answers"]["matches"]["type"] == "noul"
    assert response["answers"]["category"]["choice"] == "shirt"
    assert set(response["answers"]["category"]["probabilities"]) == {
        "shirt",
        "shoe",
    }
    assert response["answers"]["quality"]["legend"] == {
        "0": "poor",
        "1": "good",
        "2": "excellent",
    }
