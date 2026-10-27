"""Komut satırı ve Python API'si için Car Evaluation predictor."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from car_pipeline import (
    ALLOWED_VALUES,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    input_frame,
    load_artifact,
    validate_input,
)


CLASS_DESCRIPTIONS = {
    "unacc": "Kabul Edilemez",
    "acc": "Kabul Edilebilir",
    "good": "İyi",
    "vgood": "Çok İyi",
}


class CarEvaluationPredictor:
    """Tek preprocessing+model pipeline artifact'iyle tahmin yap."""

    def __init__(self, model_path=None):
        configured_path = model_path or os.environ.get(
            "CAR_MODEL_PATH",
            str(DEFAULT_MODEL_PATH),
        )
        self.model_path = Path(configured_path)
        self.artifact = None
        self.pipeline = None
        self.last_error = None

    def load_model(self):
        """Güvenilen yerel artifact'i yükle; çağıranlar için bool döndür."""
        try:
            self.artifact = load_artifact(self.model_path)
            self.pipeline = self.artifact["pipeline"]
            self.last_error = None
            return True
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.last_error = str(exc)
            self.artifact = None
            self.pipeline = None
            return False

    def _ensure_loaded(self):
        if self.pipeline is None and not self.load_model():
            raise RuntimeError(self.last_error or "Model yüklenemedi")

    def predict_single(self, input_data):
        """Doğrulanmış tek girdinin sınıfını ve olasılıklarını döndür."""
        self._ensure_loaded()
        frame = input_frame(input_data)
        prediction = str(self.pipeline.predict(frame)[0])

        probabilities = None
        if hasattr(self.pipeline, "predict_proba"):
            values = self.pipeline.predict_proba(frame)[0]
            probabilities = {
                str(class_name): float(probability)
                for class_name, probability in zip(
                    self.pipeline.classes_,
                    values,
                )
            }

        return prediction, probabilities

    def predict_batch(self, rows):
        """Bir iterable içindeki girdileri sırayla doğrula ve tahmin et."""
        results = []
        for row in rows:
            normalized = validate_input(row)
            prediction, probabilities = self.predict_single(normalized)
            results.append(
                {
                    "input": normalized,
                    "prediction": prediction,
                    "probabilities": probabilities,
                }
            )
        return results

    def predict_file(self, filename):
        """CSV veya JSON dosyasındaki girdileri tahmin et."""
        path = Path(filename)
        if path.suffix.lower() == ".csv":
            rows = pd.read_csv(path).to_dict(orient="records")
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as input_file:
                rows = json.load(input_file)
            if isinstance(rows, dict) and "data" in rows:
                rows = rows["data"]
        else:
            raise ValueError("Yalnızca .csv ve .json desteklenir")

        if not isinstance(rows, list):
            raise ValueError("Toplu girdi bir liste olmalıdır")
        return self.predict_batch(rows)

    def run_interactive_system(self):
        """Basit ve şema doğrulamalı interaktif tahmin akışı."""
        if not self.load_model():
            print(f"❌ {self.last_error}")
            print("Önce: python car_evaluation_model.py")
            return

        print("🚗 CAR EVALUATION PREDICTOR")
        print("Çıkış için q yazın.")

        while True:
            values = {}
            for feature in FEATURE_NAMES:
                allowed = "/".join(ALLOWED_VALUES[feature])
                raw_value = input(f"{feature} ({allowed}): ").strip()
                if raw_value.lower() in {"q", "quit", "exit"}:
                    print("👋 Görüşürüz!")
                    return
                values[feature] = raw_value

            try:
                prediction, probabilities = self.predict_single(values)
                print(
                    f"🎯 {prediction.upper()} — "
                    f"{CLASS_DESCRIPTIONS.get(prediction, prediction)}"
                )
                if probabilities:
                    for class_name, probability in probabilities.items():
                        print(f"  {class_name:>6}: %{probability * 100:5.1f}")
            except ValueError as exc:
                print(f"❌ {exc}")


def predict_car(buying, maint, doors, persons, lug_boot, safety):
    """Mevcut CLI/web çağrıları için geriye uyumlu yardımcı."""
    predictor = CarEvaluationPredictor()
    if not predictor.load_model():
        return None, None

    return predictor.predict_single(
        {
            "buying": buying,
            "maint": maint,
            "doors": doors,
            "persons": persons,
            "lug_boot": lug_boot,
            "safety": safety,
        }
    )


if __name__ == "__main__":
    CarEvaluationPredictor().run_interactive_system()
