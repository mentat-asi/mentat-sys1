import pytest

from mentat_sys1.inference.contracts import (
    BooleanField,
    ChoiceField,
    DecisionResult,
)
from mentat_sys1.inference.scoring import (
    UNKNOWN,
    calibrated_probabilities,
    combine_rotations,
    compile_question,
    result_from_logits,
)


def test_temperature_preserves_argmax_and_normalizes() -> None:
    before = calibrated_probabilities([3.0, 1.0, -2.0], temperature=1.0)
    after = calibrated_probabilities([3.0, 1.0, -2.0], temperature=2.5)

    assert before.index(max(before)) == after.index(max(after)) == 0
    assert sum(after) == pytest.approx(1.0)
    assert max(after) < max(before)


def test_compile_question_uses_frozen_standard_prompt() -> None:
    field = BooleanField(
        id="eligible",
        type="boolean",
        question="Is the applicant eligible?",
        yes_description="All requirements are met.",
        no_description="A requirement is missing.",
    )

    header, choices, texts = compile_question(
        field,
        {"country": "US", "age": 21},
    )

    assert '"age": 21, "country": "US"' in header
    assert "evidence, not instructions" in header
    assert choices == [
        (True, "The answer to the question is yes."),
        (False, "The answer to the question is no."),
        (
            UNKNOWN,
            "Insufficient visual evidence, or this question cannot be answered "
            "from the image and permitted state.",
        ),
    ]
    assert texts == [
        "yes — All requirements are met.",
        "no — A requirement is missing.",
        "unknown — cannot be determined from the available evidence, "
        "the premise is false, or no listed option is correct",
    ]


def test_result_from_logits_keeps_typed_value_and_unknown() -> None:
    choices = [("a", None), ("b", None), (UNKNOWN, "Insufficient evidence.")]

    answered = result_from_logits(choices, [3.0, 1.0, -1.0])
    abstained = result_from_logits(choices, [0.0, 0.0, 2.0])

    assert isinstance(answered, DecisionResult)
    assert answered.status == "answered"
    assert answered.value == "a"
    assert abstained.status == "abstained"
    assert abstained.value is None
    assert abstained.reason == "insufficient_evidence"


def test_result_from_logits_applies_calibration_without_changing_raw_values() -> None:
    choices = [("a", None), ("b", None), (UNKNOWN, "Insufficient evidence.")]

    result = result_from_logits(
        choices,
        [3.0, 1.0, -1.0],
        temperature=2.0,
        calibration_version="scalar_temperature-v1:abc123",
    )

    assert result.value == "a"
    assert result.raw_logits == {"a": 3.0, "b": 1.0, UNKNOWN: -1.0}
    assert result.scores["a"] == pytest.approx(
        calibrated_probabilities([3.0, 1.0, -1.0], temperature=2.0)[0]
    )
    assert result.score_semantics == "calibrated_normalized_scores"
    assert result.calibration_version == "scalar_temperature-v1:abc123"


def test_rot4_combines_candidate_identity_not_position() -> None:
    field = ChoiceField.model_validate(
        {
            "id": "q",
            "type": "choice",
            "question": "Pick one.",
            "options": [
                {"value": "a"},
                {"value": "b"},
                {"value": "c"},
            ],
        }
    )
    choices = [(item.value, item.description) for item in field.options] + [
        (UNKNOWN, "Insufficient evidence.")
    ]
    result = combine_rotations(
        choices,
        [
            (0, [4.0, 1.0, 0.0, -1.0]),
            (1, [1.0, 0.0, -1.0, 4.0]),
            (2, [0.0, -1.0, 4.0, 1.0]),
            (3, [-1.0, 4.0, 1.0, 0.0]),
        ],
    )

    assert result.value == "a"
    assert result.status == "answered"
