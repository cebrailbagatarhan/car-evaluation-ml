"""Ortak preprocessing, doğrulama ve model artifact yardımcıları."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ARTIFACT_FORMAT_VERSION = 1
DEFAULT_MODEL_PATH = Path("models/car_evaluation_pipeline.joblib")
FEATURE_NAMES = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
]
ALLOWED_VALUES = {
    "buying": ("vhigh", "high", "med", "low"),
    "maint": ("vhigh", "high", "med", "low"),
    "doors": ("2", "3", "4", "5more"),
    "persons": ("2", "4", "more"),
    "lug_boot": ("small", "med", "big"),
    "safety": ("low", "med", "high"),
}


def validate_input(data: Mapping) -> dict[str, str]:
    """Girdiyi tam şemaya ve bilinen kategori değerlerine göre doğrula."""
    if not isinstance(data, Mapping):
        raise ValueError("Girdi bir JSON object/dictionary olmalıdır")

    missing = [feature for feature in FEATURE_NAMES if feature not in data]
    if missing:
        raise ValueError("Eksik alanlar: " + ", ".join(missing))

    unexpected = sorted(set(data).difference(FEATURE_NAMES))
    if unexpected:
        raise ValueError("Bilinmeyen alanlar: " + ", ".join(unexpected))

    normalized = {}
    for feature in FEATURE_NAMES:
        value = str(data[feature]).strip()
        if value not in ALLOWED_VALUES[feature]:
            allowed = ", ".join(ALLOWED_VALUES[feature])
            raise ValueError(
                f"{feature} için geçersiz değer: {value!r}; izin verilen: {allowed}"
            )
        normalized[feature] = value

    return normalized


def input_frame(data: Mapping) -> pd.DataFrame:
    """Tek tahmin girdisini sabit feature sırasıyla DataFrame'e dönüştür."""
    normalized = validate_input(data)
    return pd.DataFrame([normalized], columns=FEATURE_NAMES)


def build_pipeline(estimator) -> Pipeline:
    """Kategori sırası dayatmayan OneHotEncoder + estimator pipeline'ı."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="error",
                    sparse_output=False,
                ),
                FEATURE_NAMES,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", estimator),
        ]
    )


def save_artifact(artifact: dict, model_path: Path | str) -> Path:
    """Model artifact'ini atomik dosya değişimiyle kaydet."""
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(destination)
    return destination


def load_artifact(model_path: Path | str = DEFAULT_MODEL_PATH) -> dict:
    """Yalnızca güvenilen yerel joblib artifact'ini yükle ve şemasını doğrula."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact bulunamadı: {path}. "
            "Önce car_evaluation_model.py çalıştırın."
        )

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Geçersiz model artifact formatı")
    if artifact.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            "Desteklenmeyen artifact format_version: "
            f"{artifact.get('format_version')}"
        )
    if artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Artifact feature şeması mevcut kodla uyuşmuyor")
    if "pipeline" not in artifact:
        raise ValueError("Artifact pipeline içermiyor")
    return artifact
