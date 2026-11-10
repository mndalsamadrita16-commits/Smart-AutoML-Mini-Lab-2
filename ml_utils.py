"""
Machine Learning Utilities for AutoML Mini Lab.

Provides data loading, preprocessing pipelines, model evaluation,
cross-validation, hyperparameter tuning via GridSearchCV, and neighbor lookup.
Every computation uses real scikit-learn models and genuine datasets without hardcoding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetBundle:
    """Container for a loaded machine learning classification dataset."""

    name: str
    description: str
    X: pd.DataFrame
    y: pd.Series
    feature_names: List[str]
    class_names: List[str]
    raw_images: Optional[np.ndarray] = None


def load_dataset(name: str) -> DatasetBundle:
    """
    Load a classification dataset strictly from sklearn.datasets.

    Supported datasets:
    - 'Iris'
    - 'Wine'
    - 'Breast Cancer Wisconsin'
    - 'Digits'

    Args:
        name: Name of the dataset to load.

    Returns:
        DatasetBundle containing X, y, metadata, and class/feature names.

    Raises:
        ValueError: If an unknown dataset name is supplied.
    """
    clean_name = name.strip().lower()

    if "iris" in clean_name:
        raw = load_iris(as_frame=True)
        desc = (
            "Fisher's classic Iris dataset containing 150 instances of iris flowers "
            "across 3 species (Setosa, Versicolor, Virginica) with 4 sepal and petal measurements."
        )
        return DatasetBundle(
            name="Iris",
            description=desc,
            X=raw.data,
            y=raw.target,
            feature_names=list(raw.feature_names),
            class_names=[str(c) for c in raw.target_names],
        )

    elif "wine" in clean_name:
        raw = load_wine(as_frame=True)
        desc = (
            "Chemical analysis of wines grown in the same region in Italy derived from "
            "3 different cultivars. Contains 13 continuous chemical constituents."
        )
        return DatasetBundle(
            name="Wine",
            description=desc,
            X=raw.data,
            y=raw.target,
            feature_names=list(raw.feature_names),
            class_names=[str(c) for c in raw.target_names],
        )

    elif "cancer" in clean_name:
        raw = load_breast_cancer(as_frame=True)
        desc = (
            "Wisconsin Diagnostic Breast Cancer dataset. 30 numeric morphological features "
            "computed from digitized images of fine needle aspirates (FNA) of breast masses (Malignant vs Benign)."
        )
        return DatasetBundle(
            name="Breast Cancer Wisconsin",
            description=desc,
            X=raw.data,
            y=raw.target,
            feature_names=list(raw.feature_names),
            class_names=[str(c) for c in raw.target_names],
        )

    elif "digit" in clean_name:
        raw = load_digits(as_frame=True)
        desc = (
            "Normalized 8x8 pixel grayscale images of handwritten digits (0 through 9). "
            "Total 64 features per sample representing pixel intensities."
        )
        return DatasetBundle(
            name="Digits",
            description=desc,
            X=raw.data,
            y=raw.target,
            feature_names=list(raw.feature_names),
            class_names=[str(c) for c in raw.target_names],
            raw_images=raw.images,
        )

    else:
        raise ValueError(f"Unknown dataset requested: '{name}'. Choose from Iris, Wine, Breast Cancer Wisconsin, or Digits.")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of dataset allocated to the held-out test set.
        random_state: Random state seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if len(X) == 0:
        raise ValueError("Cannot split an empty dataset.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_pipeline(
    n_neighbors: int = 5,
    metric: str = "euclidean",
    weights: str = "uniform",
    standardize: bool = True,
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with an optional StandardScaler and KNeighborsClassifier.

    Args:
        n_neighbors: Number of nearest neighbors K.
        metric: Distance metric ('euclidean', 'manhattan', 'minkowski', 'chebyshev').
        weights: Weight function ('uniform' or 'distance').
        standardize: Whether to standardize features using StandardScaler.

    Returns:
        Configured scikit-learn Pipeline.
    """
    scaler_step = StandardScaler() if standardize else "passthrough"
    knn_step = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
    )
    return Pipeline([("scaler", scaler_step), ("knn", knn_step)])


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    class_names: List[str],
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        class_names: List of human-readable class names.

    Returns:
        Tuple of (scalar_metrics_dict, report_dataframe, confusion_matrix_ndarray)
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    summary = {
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
    }

    # Generate classification report as dataframe
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(y_true, y_pred)
    return summary, report_df, cm


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: List[str],
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train pipeline and evaluate both via Stratified 5-Fold CV on training data
    and on the untouched held-out test set.

    Args:
        pipeline: scikit-learn Pipeline to evaluate.
        X_train: Training features.
        y_train: Training targets.
        X_test: Held-out test features.
        y_test: Held-out test targets.
        class_names: Target class names.
        cv_folds: Number of stratified cross-validation folds.
        random_state: Random state for fold shuffling.

    Returns:
        Dictionary with cv_mean, cv_std, test_metrics, report_df, cm, fitted_pipeline, y_pred.
    """
    # 5-fold Stratified CV on training data only
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

    # Fit once on full training set
    pipeline.fit(X_train, y_train)

    # Predict on held-out test set
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception:
            y_proba = None

    scalar_metrics, report_df, cm = calculate_metrics(y_test, y_pred, class_names)

    return {
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "cv_scores": cv_scores.tolist(),
        "test_metrics": scalar_metrics,
        "report_df": report_df,
        "confusion_matrix": cm,
        "fitted_pipeline": pipeline,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_k: int = 15,
    random_state: int = 42,
) -> Tuple[GridSearchCV, Dict[str, Any]]:
    """
    Run GridSearchCV with 5-fold StratifiedKFold on the training data.
    The test data is never touched during this optimization.

    Searches:
    - scaler: [StandardScaler(), 'passthrough']
    - knn__n_neighbors: 1 to min(max_k, n_samples)
    - knn__metric: ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
    - knn__weights: ['uniform', 'distance']

    Args:
        X_train: Training features.
        y_train: Training targets.
        max_k: Maximum neighbor count to test (default 15).
        random_state: Seed for CV fold shuffling.

    Returns:
        Tuple of (fitted GridSearchCV object, cleaned readable best_params dict)
    """
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    upper_k = min(max_k, len(X_train))
    k_range = list(range(1, upper_k + 1))

    param_grid = {
        "scaler": [StandardScaler(), "passthrough"],
        "knn__n_neighbors": k_range,
        "knn__metric": ["euclidean", "manhattan", "minkowski", "chebyshev"],
        "knn__weights": ["uniform", "distance"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_p = grid.best_params_.copy()
    readable_scaler = "StandardScaler" if isinstance(best_p.get("scaler"), StandardScaler) else "passthrough"
    cleaned_params = {
        "scaler": readable_scaler,
        "n_neighbors": int(best_p.get("knn__n_neighbors", 5)),
        "metric": str(best_p.get("knn__metric", "euclidean")),
        "weights": str(best_p.get("knn__weights", "uniform")),
    }

    return grid, cleaned_params


def get_accuracy_vs_k(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    metric: str = "euclidean",
    weights: str = "uniform",
    standardize: bool = True,
    max_k: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate training accuracy and 5-fold CV accuracy for K from 1 to max_k.

    Args:
        X_train: Training features.
        y_train: Training targets.
        metric: Distance metric to use.
        weights: Weighting scheme.
        standardize: Whether to standardize features.
        max_k: Upper limit for K.
        random_state: Random state for CV.

    Returns:
        DataFrame with columns ['k', 'cv_mean', 'cv_std', 'train_acc'].
    """
    upper_k = min(max_k, len(X_train))
    k_values = list(range(1, upper_k + 1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    records = []
    for k in k_values:
        pipe = build_pipeline(
            n_neighbors=k,
            metric=metric,
            weights=weights,
            standardize=standardize,
        )
        # CV scores on training data
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        # Training accuracy
        pipe.fit(X_train, y_train)
        y_train_pred = pipe.predict(X_train)
        train_acc = float(accuracy_score(y_train, y_train_pred))

        records.append({
            "k": k,
            "cv_mean": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
            "train_acc": train_acc,
        })

    return pd.DataFrame(records)


def run_default_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: List[str],
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fit and evaluate the standard default baseline model:
    K=5, euclidean metric, uniform weights, no scaling (passthrough).

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        class_names: Class labels.
        random_state: Random state.

    Returns:
        Evaluation dictionary from evaluate_model.
    """
    default_pipe = build_pipeline(
        n_neighbors=5,
        metric="euclidean",
        weights="uniform",
        standardize=False,
    )
    return evaluate_model(
        default_pipe,
        X_train,
        y_train,
        X_test,
        y_test,
        class_names,
        cv_folds=5,
        random_state=random_state,
    )


def run_dataset_comparison(
    test_size: float = 0.2,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Execute the ML pipeline independently on all four supported datasets:
    - Iris
    - Wine
    - Breast Cancer Wisconsin
    - Digits

    For each dataset:
    1. Stratified split with test_size and random_state
    2. 5-fold StratifiedKFold GridSearchCV on training set
    3. Final evaluation on held-out test set
    4. Records best K, metric, weights, scaling, CV accuracy, test accuracy, macro F1.

    Args:
        test_size: Test proportion.
        random_state: Random state seed.

    Returns:
        List of dictionaries with summary metrics for each dataset.
    """
    dataset_names = ["Iris", "Wine", "Breast Cancer Wisconsin", "Digits"]
    results = []

    for name in dataset_names:
        bundle = load_dataset(name)
        X_tr, X_te, y_tr, y_te = split_data(
            bundle.X,
            bundle.y,
            test_size=test_size,
            random_state=random_state,
        )

        grid, best_params = run_grid_search(X_tr, y_tr, max_k=15, random_state=random_state)
        best_estimator = grid.best_estimator_

        # Held-out evaluation
        y_pred = best_estimator.predict(X_te)
        test_acc = float(accuracy_score(y_te, y_pred))
        test_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))

        results.append({
            "Dataset": bundle.name,
            "Best K": best_params["n_neighbors"],
            "Best Metric": best_params["metric"],
            "Best Weights": best_params["weights"],
            "Standardization": best_params["scaler"],
            "CV Accuracy": round(float(grid.best_score_), 4),
            "Test Accuracy": round(test_acc, 4),
            "Macro F1": round(test_f1, 4),
            "Samples": len(bundle.X),
            "Features": bundle.X.shape[1],
            "Classes": len(bundle.class_names),
        })

    return results


def get_neighbors(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    input_sample: np.ndarray | pd.Series,
    n_neighbors: int,
    class_names: List[str],
) -> pd.DataFrame:
    """
    Find the K nearest neighbors from the actual training set for a given input sample.

    Args:
        pipeline: Fitted scikit-learn Pipeline containing scaler and knn.
        X_train: Training DataFrame.
        y_train: Training target labels.
        input_sample: 1D array or Series with matching feature dimensions.
        n_neighbors: K count.
        class_names: Class labels.

    Returns:
        DataFrame with columns ['Rank', 'Train Index', 'Distance', 'Predicted Class', 'True Class'].
    """
    scaler = pipeline.named_steps.get("scaler")
    knn = pipeline.named_steps.get("knn")

    # Format input sample to 2D
    sample_2d = np.array(input_sample).reshape(1, -1)

    if scaler is not None and scaler != "passthrough":
        scaled_input = scaler.transform(sample_2d)
    else:
        scaled_input = sample_2d

    distances, indices = knn.kneighbors(scaled_input, n_neighbors=n_neighbors, return_distance=True)

    dist_list = distances[0]
    idx_list = indices[0]

    rows = []
    for rank, (d, idx) in enumerate(zip(dist_list, idx_list), start=1):
        actual_label = y_train.iloc[idx]
        class_str = class_names[actual_label] if 0 <= actual_label < len(class_names) else str(actual_label)
        rows.append({
            "Rank": rank,
            "Train Index": int(X_train.index[idx]),
            "Distance": round(float(d), 4),
            "True Class": class_str,
        })

    return pd.DataFrame(rows)


def run_standardization_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: List[str],
    k: int = 5,
    metric: str = "euclidean",
    weights: str = "uniform",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compare identical KNN models with and without StandardScaler on the same dataset.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        class_names: Class labels.
        k: Neighbor count.
        metric: Distance metric.
        weights: Weighting scheme.
        random_state: Seed.

    Returns:
        Dictionary with unscaled and scaled evaluation results and deltas.
    """
    unscaled_pipe = build_pipeline(k, metric, weights, standardize=False)
    scaled_pipe = build_pipeline(k, metric, weights, standardize=True)

    res_unscaled = evaluate_model(unscaled_pipe, X_train, y_train, X_test, y_test, class_names, random_state=random_state)
    res_scaled = evaluate_model(scaled_pipe, X_train, y_train, X_test, y_test, class_names, random_state=random_state)

    delta_acc = res_scaled["test_metrics"]["accuracy"] - res_unscaled["test_metrics"]["accuracy"]
    delta_f1 = res_scaled["test_metrics"]["macro_f1"] - res_unscaled["test_metrics"]["macro_f1"]
    delta_cv = res_scaled["cv_mean"] - res_unscaled["cv_mean"]

    return {
        "unscaled": res_unscaled,
        "scaled": res_scaled,
        "delta_test_accuracy": delta_acc,
        "delta_macro_f1": delta_f1,
        "delta_cv_accuracy": delta_cv,
    }


def serialize_params(params: Dict[str, Any]) -> str:
    """
    Serialize model parameters to formatted JSON string.

    Args:
        params: Dictionary of parameters.

    Returns:
        JSON string.
    """
    cleaned = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, list, dict)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return json.dumps(cleaned, indent=2)
