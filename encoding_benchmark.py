"""Nested-CV comparison of ordinal and one-hot encoding on UCI Car Evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


FEATURES = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
TARGET = "class"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    predictions = labels[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        included = (confidence > lower) & (confidence <= upper)
        if index == 0:
            included |= confidence == 0.0
        if included.any():
            score += included.mean() * abs(correct[included].mean() - confidence[included].mean())
    return float(score)


def multiclass_brier_score(
    y_true: np.ndarray, probabilities: np.ndarray, labels: np.ndarray
) -> float:
    one_hot = (y_true[:, None] == labels[None, :]).astype(float)
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def model_definitions(seed: int) -> dict[str, tuple[Any, dict[str, list[Any]], bool]]:
    return {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=seed),
            {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 2, 5]},
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=seed),
            {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 2]},
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=seed),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
            False,
        ),
        "SVM": (
            SVC(probability=True, random_state=seed),
            {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", "auto"]},
            True,
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 11], "model__weights": ["uniform", "distance"]},
            True,
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, random_state=seed),
            {"model__C": [0.1, 1.0, 10.0]},
            True,
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"model__var_smoothing": [1e-9, 1e-8, 1e-7]},
            False,
        ),
    }


def build_pipeline(encoding: str, estimator: Any, scale: bool) -> Pipeline:
    if encoding == "one-hot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoding == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    preprocessing = ColumnTransformer(
        [("categorical", encoder, FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps: list[tuple[str, Any]] = [("preprocess", preprocessing)]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", clone(estimator)))
    return Pipeline(steps)


def quick_grid(grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {name: [values[0]] for name, values in grid.items()}


def top_confusions(matrix: np.ndarray, labels: np.ndarray, limit: int = 6) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for true_index, true_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            if true_index == predicted_index or matrix[true_index, predicted_index] == 0:
                continue
            items.append(
                {
                    "true": str(true_label),
                    "predicted": str(predicted_label),
                    "count": int(matrix[true_index, predicted_index]),
                }
            )
    return sorted(items, key=lambda item: item["count"], reverse=True)[:limit]


def evaluate_configuration(
    X: pd.DataFrame,
    y: pd.Series,
    labels: np.ndarray,
    encoding: str,
    model_name: str,
    estimator: Any,
    parameter_grid: dict[str, list[Any]],
    scale: bool,
    outer_cv: StratifiedKFold,
    inner_folds: int,
    seed: int,
    n_jobs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oof_predictions = np.empty(len(y), dtype=object)
    oof_probabilities = np.zeros((len(y), len(labels)), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    best_parameters: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y), start=1):
        inner_cv = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=seed + fold
        )
        search = GridSearchCV(
            build_pipeline(encoding, estimator, scale),
            parameter_grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        search.fit(X_train, y_train)
        predictions = search.predict(X_test)
        probabilities = search.predict_proba(X_test)
        class_indices = {label: index for index, label in enumerate(labels)}
        aligned = np.zeros((len(test_index), len(labels)), dtype=float)
        for source_index, label in enumerate(search.best_estimator_.classes_):
            aligned[:, class_indices[label]] = probabilities[:, source_index]
        oof_predictions[test_index] = predictions
        oof_probabilities[test_index] = aligned

        fold_rows.append(
            {
                "encoding": encoding,
                "model": model_name,
                "fold": fold,
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "macro_f1": f1_score(y_test, predictions, average="macro"),
                "weighted_f1": f1_score(y_test, predictions, average="weighted"),
                "log_loss": log_loss(y_test, aligned, labels=labels),
                "brier": multiclass_brier_score(y_test.to_numpy(), aligned, labels),
                "ece": expected_calibration_error(y_test.to_numpy(), aligned, labels),
                "inner_best_macro_f1": search.best_score_,
            }
        )
        best_parameters.append({"fold": fold, "parameters": search.best_params_})

    matrix = confusion_matrix(y, oof_predictions, labels=labels)
    report = classification_report(
        y, oof_predictions, labels=labels, output_dict=True, zero_division=0
    )
    aggregate = {
        "encoding": encoding,
        "model": model_name,
        "oof_accuracy": accuracy_score(y, oof_predictions),
        "oof_balanced_accuracy": balanced_accuracy_score(y, oof_predictions),
        "oof_macro_f1": f1_score(y, oof_predictions, average="macro"),
        "oof_weighted_f1": f1_score(y, oof_predictions, average="weighted"),
        "oof_log_loss": log_loss(y, oof_probabilities, labels=labels),
        "oof_brier": multiclass_brier_score(y.to_numpy(), oof_probabilities, labels),
        "oof_ece": expected_calibration_error(y.to_numpy(), oof_probabilities, labels),
        "oof_roc_auc_weighted_ovr": roc_auc_score(
            y, oof_probabilities, labels=labels, multi_class="ovr", average="weighted"
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "top_confusions": top_confusions(matrix, labels),
        "best_parameters_by_fold": best_parameters,
    }
    return fold_rows, aggregate


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss", "brier", "ece"]
    grouped = folds.groupby(["encoding", "model"])[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index().sort_values(["model", "encoding"])


def markdown_report(summary: pd.DataFrame, aggregates: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    aggregate_lookup = {(item["encoding"], item["model"]): item for item in aggregates}
    lines = [
        "# Encoding benchmark results",
        "",
        f"Dataset SHA-256: `{metadata['dataset_sha256']}`",
        "",
        "Primary selection metric: nested-CV outer-fold macro F1. Lower Brier/ECE is better.",
        "",
        "| Model | Encoding | Accuracy | Balanced accuracy | Macro F1 | Brier | ECE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['encoding']} | {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
            f"{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | "
            f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
            f"{row['brier_mean']:.4f} | {row['ece_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## One-hot minus ordinal",
        "",
        "Positive Macro F1 delta favors one-hot; negative Brier delta favors one-hot.",
        "",
        "| Model | Δ accuracy | Δ macro F1 | Δ Brier | Δ ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for model in sorted(summary["model"].unique()):
        subset = summary[summary["model"] == model].set_index("encoding")
        if {"one-hot", "ordinal"}.issubset(subset.index):
            one_hot, ordinal = subset.loc["one-hot"], subset.loc["ordinal"]
            lines.append(
                f"| {model} | {one_hot['accuracy_mean'] - ordinal['accuracy_mean']:+.4f} | "
                f"{one_hot['macro_f1_mean'] - ordinal['macro_f1_mean']:+.4f} | "
                f"{one_hot['brier_mean'] - ordinal['brier_mean']:+.4f} | "
                f"{one_hot['ece_mean'] - ordinal['ece_mean']:+.4f} |"
            )

    lines.extend(["", "## Most frequent out-of-fold confusions", ""])
    for key, aggregate in sorted(aggregate_lookup.items()):
        encoding, model = key
        confusions = ", ".join(
            f"{item['true']}→{item['predicted']} ({item['count']})"
            for item in aggregate["top_confusions"][:3]
        ) or "none"
        lines.append(f"- **{model} / {encoding}:** {confusions}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("car.data"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--quick", action="store_true", help="Use one parameter candidate per model")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("outer and inner folds must be at least 2")
    return args


def main() -> int:
    args = parse_args()
    data_path = args.data.resolve()
    data = pd.read_csv(data_path, names=FEATURES + [TARGET])
    X, y = data[FEATURES], data[TARGET]
    labels = np.array(sorted(y.unique()))
    definitions = model_definitions(args.seed)
    selected = args.models or list(definitions)
    unknown = sorted(set(selected).difference(definitions))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {sorted(definitions)}")

    started = time.perf_counter()
    all_folds: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for model_name in selected:
        estimator, grid, scale = definitions[model_name]
        if args.quick:
            grid = quick_grid(grid)
        for encoding in ("ordinal", "one-hot"):
            outer_cv = StratifiedKFold(
                n_splits=args.outer_folds, shuffle=True, random_state=args.seed
            )
            folds, aggregate = evaluate_configuration(
                X,
                y,
                labels,
                encoding,
                model_name,
                estimator,
                grid,
                scale,
                outer_cv,
                args.inner_folds,
                args.seed,
                args.n_jobs,
            )
            all_folds.extend(folds)
            aggregates.append(aggregate)

    folds_frame = pd.DataFrame(all_folds)
    summary = summarize_folds(folds_frame)
    metadata = {
        "schema_version": 1,
        "dataset_path": str(args.data),
        "dataset_sha256": sha256_file(data_path),
        "rows": len(data),
        "features": FEATURES,
        "classes": labels.tolist(),
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "selection_metric": "macro_f1",
        "quick": args.quick,
        "n_jobs": args.n_jobs,
        "command": ["python", *sys.argv],
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
        },
    }
    report = {"metadata": metadata, "summary": summary.to_dict(orient="records"), "aggregates": aggregates}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds_frame.to_csv(args.output_dir / "nested_cv_folds.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        markdown_report(summary, aggregates, metadata), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
