"""
Car Evaluation — sızıntısız model seçimi ve tek artifact üretimi.

Aday modeller yalnızca training split üzerindeki stratified cross-validation
ile seçilir. Hold-out test seti, seçilen pipeline yeniden fit edildikten sonra
bir kez raporlanır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from car_pipeline import (
    ARTIFACT_FORMAT_VERSION,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    build_pipeline,
    save_artifact,
)


CLASS_NAME = "class"


def candidate_estimators(random_state: int = 42):
    """Karşılaştırılacak estimator'ları döndür."""
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVM": SVC(probability=True, random_state=random_state),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        ),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        ),
    }


def load_data(data_path: Path | str):
    """UCI car.data dosyasını kategorik feature'larla yükle."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset bulunamadı: {path}")

    dataframe = pd.read_csv(path, names=FEATURE_NAMES + [CLASS_NAME])
    if dataframe.isnull().any().any():
        raise ValueError("Dataset eksik değer içeriyor")
    return dataframe


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_and_evaluate(
    data_path: Path | str = "car.data",
    model_path: Path | str = DEFAULT_MODEL_PATH,
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_state: int = 42,
    selected_candidates: list[str] | None = None,
):
    """CV ile model seç, test setini bir kez değerlendir ve artifact kaydet."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size 0 ile 1 arasında olmalıdır")
    if cv_folds < 2:
        raise ValueError("cv_folds en az 2 olmalıdır")

    dataframe = load_data(data_path)
    X = dataframe[FEATURE_NAMES]
    y = dataframe[CLASS_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    candidates = candidate_estimators(random_state)
    if selected_candidates is not None:
        unknown = sorted(set(selected_candidates).difference(candidates))
        if unknown:
            raise ValueError("Bilinmeyen modeller: " + ", ".join(unknown))
        candidates = {
            name: candidates[name]
            for name in selected_candidates
        }

    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    model_scores = {}

    print("Aday modeller training split üzerinde CV ile değerlendiriliyor:")
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cross_validation,
            scoring="accuracy",
            n_jobs=-1,
        )
        model_scores[name] = {
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "cv_scores": [float(score) for score in scores],
        }
        print(
            f"  {name:<24} "
            f"CV={scores.mean():.4f} ± {scores.std():.4f}"
        )

    best_model_name = max(
        model_scores,
        key=lambda name: (
            model_scores[name]["cv_mean"],
            -model_scores[name]["cv_std"],
            name,
        ),
    )

    final_pipeline = build_pipeline(candidates[best_model_name])
    final_pipeline.fit(X_train, y_train)
    test_predictions = final_pipeline.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, test_predictions))
    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0,
    )
    report = json.loads(json.dumps(report, default=float))

    print(f"\nSeçilen model: {best_model_name}")
    print(f"Hold-out test accuracy (tek kullanım): {test_accuracy:.4f}")

    artifact = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "pipeline": final_pipeline,
        "feature_names": FEATURE_NAMES,
        "classes": [str(label) for label in final_pipeline.classes_],
        "selected_model": best_model_name,
        "selection_scores": model_scores,
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "dataset_sha256": file_sha256(data_path),
        "random_state": random_state,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "sklearn_version": sklearn.__version__,
    }
    destination = save_artifact(artifact, model_path)

    metadata = {
        key: value
        for key, value in artifact.items()
        if key != "pipeline"
    }
    metadata_path = destination.with_suffix(".metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)

    print(f"Model artifact: {destination}")
    print(f"Metadata:       {metadata_path}")
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description="Car Evaluation model eğitimi")
    parser.add_argument("--data", default="car.data")
    parser.add_argument(
        "--model-output",
        default=str(DEFAULT_MODEL_PATH),
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    train_and_evaluate(
        data_path=args.data,
        model_path=args.model_output,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
