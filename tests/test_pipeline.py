from pathlib import Path

import pytest

from car_evaluation_model import train_and_evaluate
from car_pipeline import FEATURE_NAMES, load_artifact, validate_input
from interactive_predictor import CarEvaluationPredictor


VALID_CAR = {
    "buying": "low",
    "maint": "med",
    "doors": "4",
    "persons": "more",
    "lug_boot": "big",
    "safety": "high",
}


def test_validate_input_requires_exact_schema():
    assert list(validate_input(VALID_CAR)) == FEATURE_NAMES

    with pytest.raises(ValueError, match="Eksik alanlar"):
        validate_input({"buying": "low"})

    with pytest.raises(ValueError, match="Bilinmeyen alanlar"):
        validate_input({**VALID_CAR, "target": "vgood"})

    with pytest.raises(ValueError, match="geçersiz değer"):
        validate_input({**VALID_CAR, "safety": "extreme"})


def test_training_artifact_round_trip(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "car_evaluation_pipeline.joblib"

    result = train_and_evaluate(
        data_path=repository_root / "car.data",
        model_path=destination,
        cv_folds=3,
        selected_candidates=["Decision Tree"],
    )

    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert result["selected_model"] == "Decision Tree"
    assert destination.is_file()
    assert destination.with_suffix(".metadata.json").is_file()

    loaded = load_artifact(destination)
    assert loaded["feature_names"] == FEATURE_NAMES

    predictor = CarEvaluationPredictor(destination)
    prediction, probabilities = predictor.predict_single(VALID_CAR)
    assert prediction in {"unacc", "acc", "good", "vgood"}
    assert probabilities is not None
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9
