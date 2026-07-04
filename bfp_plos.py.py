#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.preprocessing import OneHotEncoder
except Exception:  # pragma: no cover
    OneHotEncoder = None

try:
    from sklearn.svm import SVR
    HAS_SVR = True
except Exception:
    HAS_SVR = False

try:
    from xgboost import XGBRegressor
    import xgboost
    HAS_XGB = True
except Exception:
    XGBRegressor = None
    xgboost = None
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    shap = None
    HAS_SHAP = False


RANDOM_STATE = 42
DEFAULT_TARGET = "BFP"


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(text: str, max_len: int = 90) -> str:
    text = str(text)
    text = re.sub(r"[^\w\s.-]+", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text.strip())
    text = text.replace("/", "_").replace("\\", "_")
    return text[:max_len].strip("_") or "feature"


def json_safe(obj: Any) -> Any:
    """Convert numpy/pandas/sklearn objects into JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "get_params"):
        return str(obj)
    return obj


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_rmse_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        scores.append(rmse(y_true[idx], y_pred[idx]))

    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def plot_save(fig: plt.Figure, out_png: str | Path, also_pdf: bool = True, dpi: int = 600) -> None:
    out_png = Path(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def clean_column_name(name: str) -> str:
    """Fix obvious typographical issues in feature labels for manuscript figures."""
    name = str(name).strip()
    replacements = {
        "0rmal": "normal",
        "e0ugh": "enough",
        "vegatables": "vegetables",
        "sweeten food": "sweetened food",
        "carbonated beverage": "carbonated beverage",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    name = re.sub(r"\s+", " ", name)
    return name


def make_unique(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}__{seen[name]}")
    return out


def read_csv_robust(data_path: str | Path) -> pd.DataFrame:
    """Read CSV safely on Windows and give clear diagnostics if the file is missing.

    This avoids the long pandas traceback that normally appears at pd.read_csv().
    """
    path = Path(str(data_path).strip().strip('"').strip("'"))
    path = path.expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        csv_files = sorted([p.name for p in Path.cwd().glob("*.csv")])
        msg = [
            f"CSV file not found: {path}",
            f"Current working directory: {Path.cwd()}",
        ]
        if csv_files:
            msg.append("CSV files found in current directory: " + ", ".join(csv_files[:20]))
        else:
            msg.append("No CSV files were found in the current directory.")
        msg.append('Use an absolute path, for example: --data "C:\\2026_10.1_Hanh_SinhHoc\\Data(4).csv"')
        raise FileNotFoundError("\n".join(msg))

    last_error = None
    for enc in ["utf-8-sig", "utf-8", "cp1258", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise last_error if last_error is not None else RuntimeError(f"Could not read CSV: {path}")


def load_and_prepare_data(
    data_path: str | Path,
    target: str = DEFAULT_TARGET,
    clean_feature_names: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame, pd.DataFrame]:
    df = read_csv_robust(data_path)

    if clean_feature_names:
        old_cols = list(df.columns)
        new_cols = make_unique([clean_column_name(c) for c in old_cols])
        mapping = pd.DataFrame({"original_column": old_cols, "analysis_column": new_cols})
        df.columns = new_cols
        target = clean_column_name(target)
    else:
        mapping = pd.DataFrame({"original_column": df.columns, "analysis_column": df.columns})

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns include: {list(df.columns[:10])}..."
        )

    # Remove rows with missing target, if any.
    before_n = len(df)
    df = df.loc[~df[target].isna()].copy()
    removed_target_missing = before_n - len(df)

    y = df[target].astype(float)
    X = df.drop(columns=[target])

    # ID-like leakage detection: only remove from predictors, never from target.
    removed_cols = []
    for col in list(X.columns):
        if X[col].nunique(dropna=False) == len(X):
            removed_cols.append(col)

    if removed_cols:
        X = X.drop(columns=removed_cols)

    data_report = pd.DataFrame(
        {
            "n_rows_original": [before_n],
            "n_rows_used": [len(df)],
            "n_rows_removed_missing_target": [removed_target_missing],
            "n_predictors_after_id_screening": [X.shape[1]],
            "removed_id_like_columns": [", ".join(removed_cols) if removed_cols else ""],
        }
    )

    return X, y, removed_cols, mapping, data_report


def make_onehot_encoder() -> Any:
    if OneHotEncoder is None:
        raise ImportError("OneHotEncoder is unavailable in this sklearn installation.")

    # sklearn >= 1.2 uses sparse_output; older versions use sparse.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("No predictor columns were available after preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_processed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    names: List[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue

        columns = list(columns)

        if name == "num":
            names.extend([str(c) for c in columns])

        elif name == "cat":
            try:
                onehot = transformer.named_steps["onehot"]
                names.extend([str(x) for x in onehot.get_feature_names_out(columns)])
            except Exception:
                names.extend([str(c) for c in columns])

    return names


# ---------------------------------------------------------------------------
# Model definitions and tuning
# ---------------------------------------------------------------------------

def model_specs(seed: int = RANDOM_STATE, quick: bool = False, n_jobs: int = -1) -> Dict[str, Dict[str, Any]]:
    rf_estimators = [200, 300] if quick else [300, 500, 800]
    xgb_estimators = [200, 400] if quick else [400, 600, 800]

    specs: Dict[str, Dict[str, Any]] = {
        "ElasticNet": {
            "estimator": ElasticNet(max_iter=30000, random_state=seed),
            "params": {
                "model__alpha": np.logspace(-4, 1, 30),
                "model__l1_ratio": np.linspace(0.1, 0.9, 9),
            },
        },
        "Ridge": {
            "estimator": Ridge(),
            "params": {
                "model__alpha": np.logspace(-4, 4, 40),
            },
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=seed, n_jobs=n_jobs),
            "params": {
                "model__n_estimators": rf_estimators,
                "model__max_depth": [None, 5, 10, 15, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
            },
        },
    }

    if HAS_SVR:
        specs["SVR"] = {
            "estimator": SVR(),
            "params": {
                "model__C": np.logspace(-1, 2, 20),
                "model__epsilon": [0.05, 0.1, 0.2, 0.5, 1.0],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["rbf"],
            },
        }

    if HAS_XGB:
        specs["XGBoost"] = {
            "estimator": XGBRegressor(
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=n_jobs,
                tree_method="hist",
                verbosity=0,
            ),
            "params": {
                "model__n_estimators": xgb_estimators,
                "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
                "model__max_depth": [2, 3, 4, 5],
                "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__reg_lambda": [0.1, 1.0, 5.0, 10.0],
            },
        }

    return specs


def fit_tuned_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    specs: Dict[str, Dict[str, Any]],
    inner_cv: KFold,
    seed: int = RANDOM_STATE,
    n_iter: int = 15,
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict[str, Any], Optional[float]]:
    if model_name not in specs:
        raise KeyError(f"Unknown model name: {model_name}")

    preprocessor = build_preprocessor(X_train)
    estimator = clone(specs[model_name]["estimator"])
    pipe = Pipeline(steps=[("pre", preprocessor), ("model", estimator)])
    params = specs[model_name].get("params", {})

    if params:
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=params,
            n_iter=n_iter,
            cv=inner_cv,
            scoring="neg_mean_squared_error",
            random_state=seed,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, dict(search.best_params_), float(search.best_score_)

    pipe.fit(X_train, y_train)
    return pipe, {}, None


def nested_model_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    specs: Dict[str, Dict[str, Any]],
    out_dir: Path,
    seed: int = RANDOM_STATE,
    n_iter: int = 15,
    n_jobs: int = -1,
) -> pd.DataFrame:
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=seed)

    fold_rows: List[Dict[str, Any]] = []

    for model_name in specs.keys():
        print(f"[Nested CV] {model_name}")
        for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[test_idx]

            best_pipe, best_params, best_score = fit_tuned_pipeline(
                X_train=X_train,
                y_train=y_train,
                model_name=model_name,
                specs=specs,
                inner_cv=inner_cv,
                seed=seed,
                n_iter=n_iter,
                n_jobs=n_jobs,
            )

            pred = best_pipe.predict(X_valid)
            metrics = evaluate_predictions(y_valid.values, pred)

            fold_rows.append(
                {
                    "Model": model_name,
                    "Fold": fold_id,
                    **metrics,
                    "Best_inner_score_neg_MSE": best_score,
                    "Best_params": json.dumps(json_safe(best_params), ensure_ascii=False),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "nested_cv_fold_results.csv", index=False)

    summary = (
        fold_df.groupby("Model")
        .agg(
            Nested_RMSE_mean=("RMSE", "mean"),
            Nested_RMSE_std=("RMSE", "std"),
            Nested_MAE_mean=("MAE", "mean"),
            Nested_MAE_std=("MAE", "std"),
            Nested_R2_mean=("R2", "mean"),
            Nested_R2_std=("R2", "std"),
        )
        .reset_index()
        .sort_values("Nested_RMSE_mean", ascending=True)
    )

    summary.to_csv(out_dir / "nested_cv_results.csv", index=False)
    return summary


# ---------------------------------------------------------------------------
# Calibration and figures
# ---------------------------------------------------------------------------

def calibration_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    filename_prefix: str = "fig2_calibration_plot",
) -> Dict[str, float]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.scatter(y_pred, y_true, alpha=0.65, s=28)

    x_vals = np.linspace(float(np.min(y_pred)), float(np.max(y_pred)), 200)
    ax.plot(x_vals, intercept + slope * x_vals, linewidth=2, label="Calibration line")
    ax.plot(x_vals, x_vals, linestyle="--", linewidth=2, label="Ideal line")

    ax.set_xlabel("Predicted BFP (%)", fontsize=12)
    ax.set_ylabel("Observed BFP (%)", fontsize=12)
    ax.set_title("Calibration plot", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    plot_save(fig, out_dir / f"{filename_prefix}.png", also_pdf=True, dpi=600)

    return {
        "Calibration_slope": float(slope),
        "Calibration_intercept": float(intercept),
        "Calibration_R2": float(r_value**2),
        "Calibration_p_value": float(p_value),
        "Calibration_std_error": float(std_err),
    }


def create_workflow_figure(out_dir: Path) -> None:
    """Generate a cleaner high-resolution workflow figure for the manuscript."""
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.axis("off")

    boxes = [
        ("Dataset\nn = 1,208 adolescents", 0.04, 0.68),
        ("Data preprocessing\nmissing-value imputation\nfeature scaling\nID-like screening", 0.24, 0.68),
        ("Nested model comparison\n5 outer folds\n3 inner folds\ntuned hyperparameters", 0.46, 0.68),
        ("Model selection\nlowest nested RMSE", 0.70, 0.68),
        ("Final test evaluation\n80/20 train-test split\nRMSE, MAE, R²\nbootstrap CI", 0.70, 0.28),
        ("Calibration assessment\nslope, intercept\nlinear/isotonic sensitivity", 0.46, 0.28),
        ("Explainability\npermutation importance\nSHAP summary\ndependence plots", 0.24, 0.28),
        ("Subgroup analysis\nGender subgroup metrics\nsex-stratified models", 0.04, 0.28),
    ]

    width = 0.18
    height = 0.18

    for text, x, y in boxes:
        rect = matplotlib.patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="black",
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.04 + width, 0.77), (0.24, 0.77)),
        ((0.24 + width, 0.77), (0.46, 0.77)),
        ((0.46 + width, 0.77), (0.70, 0.77)),
        ((0.79, 0.68), (0.79, 0.46)),
        ((0.70, 0.37), (0.46 + width, 0.37)),
        ((0.46, 0.37), (0.24 + width, 0.37)),
        ((0.24, 0.37), (0.04 + width, 0.37)),
    ]

    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", linewidth=1.4),
        )

    plot_save(fig, out_dir / "fig1_workflow_revision.png", also_pdf=True, dpi=600)


def recalibration_sensitivity(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    specs: Dict[str, Dict[str, Any]],
    out_dir: Path,
    seed: int = RANDOM_STATE,
    n_iter: int = 15,
    n_jobs: int = -1,
    calibration_fraction: float = 0.20,
) -> pd.DataFrame:
    """
    Sensitivity analysis requested by reviewers.

    A separate base model is trained on a model-development subset. A calibration
    subset, not used for fitting that base model, is used to estimate linear and
    isotonic recalibration maps. All raw and recalibrated predictions are then
    evaluated on the same held-out test set.
    """
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_full,
        y_train_full,
        test_size=calibration_fraction,
        random_state=seed,
    )

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=seed)

    base_pipe, base_params, _ = fit_tuned_pipeline(
        X_train=X_fit,
        y_train=y_fit,
        model_name=model_name,
        specs=specs,
        inner_cv=inner_cv,
        seed=seed,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )

    pred_cal = base_pipe.predict(X_cal)
    pred_test_raw = base_pipe.predict(X_test)

    # Linear recalibration for continuous regression.
    linear_cal = LinearRegression()
    linear_cal.fit(pred_cal.reshape(-1, 1), y_cal.values)
    pred_test_linear = linear_cal.predict(pred_test_raw.reshape(-1, 1))

    # Non-parametric monotonic recalibration.
    iso_cal = IsotonicRegression(out_of_bounds="clip")
    iso_cal.fit(pred_cal, y_cal.values)
    pred_test_iso = iso_cal.predict(pred_test_raw)

    rows = []
    for method, pred in [
        ("Raw_base_model", pred_test_raw),
        ("Linear_recalibration", pred_test_linear),
        ("Isotonic_recalibration", pred_test_iso),
    ]:
        row = {"Method": method, **evaluate_predictions(y_test.values, pred)}
        row.update(calibration_analysis_values(y_test.values, pred))
        rows.append(row)

    recal_df = pd.DataFrame(rows)
    recal_df.to_csv(out_dir / "recalibration_sensitivity.csv", index=False)

    pred_df = pd.DataFrame(
        {
            "Observed_BFP": y_test.values,
            "Raw_base_prediction": pred_test_raw,
            "Linear_recalibrated_prediction": pred_test_linear,
            "Isotonic_recalibrated_prediction": pred_test_iso,
        },
        index=y_test.index,
    )
    pred_df.to_csv(out_dir / "recalibration_sensitivity_predictions.csv", index=True)

    save_json(
        {
            "model_name": model_name,
            "base_model_best_params": base_params,
            "linear_recalibration_intercept": float(linear_cal.intercept_),
            "linear_recalibration_slope": float(linear_cal.coef_[0]),
            "calibration_fraction_within_training_set": calibration_fraction,
        },
        out_dir / "recalibration_sensitivity_metadata.json",
    )

    return recal_df


def calibration_analysis_values(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_pred, y_true)
    return {
        "Calibration_slope": float(slope),
        "Calibration_intercept": float(intercept),
        "Calibration_R2": float(r_value**2),
    }


# ---------------------------------------------------------------------------
# Importance, SHAP, and subgroup analyses
# ---------------------------------------------------------------------------

def permutation_importance_analysis(
    final_pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
    n_repeats: int = 30,
    n_jobs: int = -1,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    perm = permutation_importance(
        final_pipe,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=n_jobs,
        scoring="r2",
    )

    imp = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean_decrease_in_R2": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean_decrease_in_R2", ascending=False)

    imp.to_csv(out_dir / "permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6.8))
    top = imp.head(15).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean_decrease_in_R2"], xerr=top["importance_std"])
    ax.set_xlabel("Permutation importance: mean decrease in R²", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Top predictors based on permutation importance", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    plot_save(fig, out_dir / "permutation_importance_top15.png", also_pdf=True, dpi=600)

    return imp


def subgroup_metrics_by_gender(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    gender_col: str,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []

    for group_value in sorted(pd.unique(X_test[gender_col])):
        mask = X_test[gender_col].values == group_value
        if int(np.sum(mask)) < 2:
            continue
        metrics = evaluate_predictions(y_test.values[mask], y_pred[mask])
        rows.append(
            {
                "Gender_group": group_value,
                "n_test": int(np.sum(mask)),
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "subgroup_metrics_by_gender.csv", index=False)
    return df


def sex_stratified_models(
    X: pd.DataFrame,
    y: pd.Series,
    gender_col: str,
    model_name: str,
    specs: Dict[str, Dict[str, Any]],
    out_dir: Path,
    seed: int = RANDOM_STATE,
    n_iter: int = 15,
    n_jobs: int = -1,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=seed)

    for group_value in sorted(pd.unique(X[gender_col])):
        mask = X[gender_col] == group_value
        X_g = X.loc[mask].drop(columns=[gender_col]).copy()
        y_g = y.loc[mask].copy()

        if len(X_g) < 80:
            rows.append(
                {
                    "Gender_group": group_value,
                    "n_total": len(X_g),
                    "status": "Skipped: subgroup too small for stable split",
                }
            )
            continue

        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
            X_g,
            y_g,
            test_size=0.2,
            random_state=seed,
        )

        pipe_g, params_g, _ = fit_tuned_pipeline(
            X_train=X_train_g,
            y_train=y_train_g,
            model_name=model_name,
            specs=specs,
            inner_cv=inner_cv,
            seed=seed,
            n_iter=n_iter,
            n_jobs=n_jobs,
        )

        pred_g = pipe_g.predict(X_test_g)
        metrics_g = evaluate_predictions(y_test_g.values, pred_g)
        ci_low, ci_high = bootstrap_rmse_ci(y_test_g.values, pred_g, n_boot=1000, seed=seed)

        rows.append(
            {
                "Gender_group": group_value,
                "model": model_name,
                "n_total": len(X_g),
                "n_train": len(X_train_g),
                "n_test": len(X_test_g),
                **metrics_g,
                "RMSE_95CI_low": ci_low,
                "RMSE_95CI_high": ci_high,
                "Best_params": json.dumps(json_safe(params_g), ensure_ascii=False),
                "status": "Completed",
            }
        )

        pred_rows.append(
            pd.DataFrame(
                {
                    "Gender_group": group_value,
                    "Observed_BFP": y_test_g.values,
                    "Predicted_BFP": pred_g,
                },
                index=y_test_g.index,
            )
        )

    strat_df = pd.DataFrame(rows)
    strat_df.to_csv(out_dir / "sex_stratified_model_metrics.csv", index=False)

    if pred_rows:
        pd.concat(pred_rows).to_csv(out_dir / "sex_stratified_predictions.csv", index=True)

    return strat_df


def select_features_for_dependence(
    imp_df: pd.DataFrame,
    available_features: List[str],
    gender_col: Optional[str],
    top_k: int = 5,
) -> List[str]:
    selected: List[str] = []

    # Include Gender if available and important.
    if gender_col and gender_col in available_features:
        selected.append(gender_col)

    for feat in imp_df["feature"].tolist():
        if feat in available_features and feat not in selected:
            selected.append(feat)
        if len(selected) >= top_k:
            break

    return selected[:top_k]


def shap_analysis(
    final_pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    imp_df: pd.DataFrame,
    out_dir: Path,
    gender_col: Optional[str] = "Gender",
    top_k_dependence: int = 5,
    max_display: int = 20,
) -> Optional[pd.DataFrame]:
    if not HAS_SHAP:
        warnings.warn("SHAP is not installed. Skipping SHAP analysis.")
        (out_dir / "shap_skipped.txt").write_text("SHAP is not installed.\n", encoding="utf-8")
        return None

    model = final_pipe.named_steps["model"]
    model_type = model.__class__.__name__.lower()

    if not ("forest" in model_type or "xgb" in model_type or "gradient" in model_type):
        msg = f"Final model {model.__class__.__name__} is not tree-based. SHAP TreeExplainer skipped."
        warnings.warn(msg)
        (out_dir / "shap_skipped.txt").write_text(msg + "\n", encoding="utf-8")
        return None

    pre = final_pipe.named_steps["pre"]
    X_train_trans = pre.transform(X_train)
    X_test_trans = pre.transform(X_test)

    if hasattr(X_train_trans, "toarray"):
        X_train_trans = X_train_trans.toarray()
    if hasattr(X_test_trans, "toarray"):
        X_test_trans = X_test_trans.toarray()

    feature_names = get_processed_feature_names(pre)
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    # Use a moderate background sample for stability and speed.
    rng = np.random.default_rng(RANDOM_STATE)
    if X_train_trans.shape[0] > 500:
        bg_idx = rng.choice(X_train_trans.shape[0], size=500, replace=False)
        background = X_train_trans[bg_idx]
    else:
        background = X_train_trans

    try:
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
        )
        shap_values = explainer.shap_values(X_test_trans, check_additivity=False)
    except TypeError:
        explainer = shap.TreeExplainer(model, data=background)
        shap_values = explainer.shap_values(X_test_trans)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    # SHAP summary plot: revised high-resolution Figure 3 candidate.
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_trans_df,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
        plot_size=(10, 7.5),
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_shap_summary_revised.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_shap_summary_revised.pdf", bbox_inches="tight")
    plt.close(fig)

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_SHAP": mean_abs,
        }
    ).sort_values("mean_abs_SHAP", ascending=False)
    shap_imp.to_csv(out_dir / "shap_mean_abs_importance.csv", index=False)

    # Dependence plots and direction summary for manuscript text.
    available_original_features = [f for f in X_test.columns if f in feature_names]
    selected = select_features_for_dependence(
        imp_df=imp_df,
        available_features=available_original_features,
        gender_col=gender_col if gender_col in X_test.columns else None,
        top_k=top_k_dependence,
    )

    direction_rows: List[Dict[str, Any]] = []

    for rank, feat in enumerate(selected, start=1):
        if feat not in feature_names:
            continue

        idx = feature_names.index(feat)
        shap_feat = shap_values[:, idx]
        x_original = X_test[feat].values

        # Spearman direction between original feature coding and SHAP contribution.
        try:
            rho, p_value = stats.spearmanr(x_original, shap_feat, nan_policy="omit")
        except Exception:
            rho, p_value = np.nan, np.nan

        median_val = np.nanmedian(x_original)
        low_mask = x_original <= median_val
        high_mask = x_original > median_val

        mean_shap_low = float(np.nanmean(shap_feat[low_mask])) if np.any(low_mask) else np.nan
        mean_shap_high = float(np.nanmean(shap_feat[high_mask])) if np.any(high_mask) else np.nan

        if np.isfinite(rho) and abs(rho) >= 0.10:
            direction = "Higher coded values tend to increase predicted BFP" if rho > 0 else "Higher coded values tend to decrease predicted BFP"
        else:
            direction = "No clear monotonic SHAP direction"

        direction_rows.append(
            {
                "feature": feat,
                "spearman_rho_feature_vs_SHAP": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "median_feature_value": float(median_val) if np.isfinite(median_val) else np.nan,
                "mean_SHAP_at_or_below_median": mean_shap_low,
                "mean_SHAP_above_median": mean_shap_high,
                "direction_summary": direction,
            }
        )

        fig, ax = plt.subplots(figsize=(7.4, 5.8))
        ax.scatter(x_original, shap_feat, alpha=0.65, s=28)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel("SHAP contribution to predicted BFP", fontsize=11)
        ax.set_title(f"SHAP dependence plot: {feat}", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(alpha=0.25)
        fname = f"shap_dependence_{rank}_{sanitize_filename(feat)}.png"
        plot_save(fig, out_dir / fname, also_pdf=True, dpi=600)

    direction_df = pd.DataFrame(direction_rows)
    direction_df.to_csv(out_dir / "shap_direction_summary.csv", index=False)

    # Optional interaction: Gender × highest-ranked non-gender feature.
    if gender_col and gender_col in feature_names:
        top_non_gender = None
        for feat in imp_df["feature"].tolist():
            if feat != gender_col and feat in feature_names:
                top_non_gender = feat
                break

        if top_non_gender is not None:
            try:
                # For interaction values, tree-path-dependent explainer is usually more compatible.
                explainer_inter = shap.TreeExplainer(model)
                shap_inter = explainer_inter.shap_interaction_values(X_test_trans)
                if isinstance(shap_inter, list):
                    shap_inter = shap_inter[0]

                gender_idx = feature_names.index(gender_col)
                top_idx = feature_names.index(top_non_gender)
                interaction_vals = np.asarray(shap_inter)[:, gender_idx, top_idx]

                inter_df = pd.DataFrame(
                    {
                        "Gender": X_test[gender_col].values,
                        "Feature": top_non_gender,
                        "Feature_value": X_test[top_non_gender].values if top_non_gender in X_test.columns else X_test_trans[:, top_idx],
                        "Interaction_SHAP_value": interaction_vals,
                    },
                    index=X_test.index,
                )
                inter_df.to_csv(out_dir / "shap_interaction_gender_top_lifestyle.csv", index=True)

                fig, ax = plt.subplots(figsize=(7.4, 5.5))
                ax.hist(interaction_vals, bins=30)
                ax.axvline(0, linestyle="--", linewidth=1)
                ax.set_xlabel(f"SHAP interaction value: {gender_col} × {top_non_gender}", fontsize=10)
                ax.set_ylabel("Frequency", fontsize=11)
                ax.set_title(f"SHAP interaction: {gender_col} × {top_non_gender}", fontsize=12)
                ax.tick_params(axis="both", labelsize=10)
                plot_save(fig, out_dir / "shap_interaction_gender_top_lifestyle.png", also_pdf=True, dpi=600)

            except Exception as exc:
                (out_dir / "shap_interaction_warning.txt").write_text(
                    f"SHAP interaction analysis could not be completed: {repr(exc)}\n",
                    encoding="utf-8",
                )

    return direction_df


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)

    print("Loading data...")
    X, y, removed_cols, col_mapping, data_report = load_and_prepare_data(
        data_path=args.data,
        target=args.target,
        clean_feature_names=not args.no_clean_feature_names,
    )

    col_mapping.to_csv(out_dir / "feature_name_mapping.csv", index=False)
    data_report.to_csv(out_dir / "data_preprocessing_report.csv", index=False)

    gender_col = clean_column_name(args.gender_col) if not args.no_clean_feature_names else args.gender_col
    if gender_col not in X.columns:
        warnings.warn(f"Gender column '{gender_col}' not found. Gender subgroup analyses will be skipped.")
        gender_col_for_analysis = None
    else:
        gender_col_for_analysis = gender_col

    n_iter = args.n_iter
    if args.quick:
        n_iter = min(n_iter, 5)

    specs = model_specs(seed=args.seed, quick=args.quick, n_jobs=args.n_jobs)

    if args.models:
        requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
        unavailable = [m for m in requested_models if m not in specs]
        if unavailable:
            raise ValueError(
                f"Requested model(s) not available: {unavailable}. "
                f"Available models: {list(specs.keys())}"
            )
        specs = {m: specs[m] for m in requested_models}

    print("Running nested cross-validation with hyperparameter tuning...")
    nested_summary = nested_model_comparison(
        X=X,
        y=y,
        specs=specs,
        out_dir=out_dir,
        seed=args.seed,
        n_iter=n_iter,
        n_jobs=args.n_jobs,
    )

    best_model_name = str(nested_summary.iloc[0]["Model"])
    print(f"Best model by nested RMSE: {best_model_name}")

    # Final independent 80/20 train-test evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=args.seed)

    print("Fitting final tuned model on the training split...")
    final_pipe, final_best_params, final_inner_score = fit_tuned_pipeline(
        X_train=X_train,
        y_train=y_train,
        model_name=best_model_name,
        specs=specs,
        inner_cv=inner_cv,
        seed=args.seed,
        n_iter=n_iter,
        n_jobs=args.n_jobs,
    )

    y_pred = final_pipe.predict(X_test)
    final_metrics = evaluate_predictions(y_test.values, y_pred)
    ci_low, ci_high = bootstrap_rmse_ci(
        y_test.values,
        y_pred,
        n_boot=args.bootstrap_iter,
        seed=args.seed,
    )

    calibration_metrics = calibration_analysis(
        y_true=y_test.values,
        y_pred=y_pred,
        out_dir=out_dir,
        filename_prefix="fig2_calibration_plot_revised",
    )

    final_metrics_full = {
        "Best_Model": best_model_name,
        **final_metrics,
        "RMSE_95CI_low": ci_low,
        "RMSE_95CI_high": ci_high,
        **calibration_metrics,
        "Final_best_params": final_best_params,
        "Final_inner_best_score_neg_MSE": final_inner_score,
        "Removed_id_like_columns": removed_cols,
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_predictors": int(X.shape[1]),
    }

    save_json(final_metrics_full, out_dir / "final_metrics.json")

    pd.DataFrame(
        {
            "Observed_BFP": y_test.values,
            "Predicted_BFP": y_pred,
            "Residual": y_test.values - y_pred,
        },
        index=y_test.index,
    ).to_csv(out_dir / "test_predictions.csv", index=True)

    print("Generating revised workflow figure...")
    create_workflow_figure(out_dir)

    print("Running permutation importance...")
    imp_df = permutation_importance_analysis(
        final_pipe=final_pipe,
        X_test=X_test,
        y_test=y_test,
        out_dir=out_dir,
        n_repeats=args.perm_repeats,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    if gender_col_for_analysis is not None:
        print("Running gender subgroup metrics...")
        subgroup_metrics_by_gender(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            gender_col=gender_col_for_analysis,
            out_dir=out_dir,
        )

        print("Running sex-stratified model analysis...")
        sex_stratified_models(
            X=X,
            y=y,
            gender_col=gender_col_for_analysis,
            model_name=best_model_name,
            specs=specs,
            out_dir=out_dir,
            seed=args.seed,
            n_iter=n_iter,
            n_jobs=args.n_jobs,
        )

    if not args.skip_recalibration:
        print("Running recalibration sensitivity analysis...")
        recalibration_sensitivity(
            X_train_full=X_train,
            y_train_full=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=best_model_name,
            specs=specs,
            out_dir=out_dir,
            seed=args.seed,
            n_iter=n_iter,
            n_jobs=args.n_jobs,
            calibration_fraction=args.calibration_fraction,
        )

    if not args.skip_shap:
        print("Running SHAP analysis...")
        shap_analysis(
            final_pipe=final_pipe,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            imp_df=imp_df,
            out_dir=out_dir,
            gender_col=gender_col_for_analysis,
            top_k_dependence=args.top_k_dependence,
            max_display=args.shap_max_display,
        )

    # Save reproducibility metadata.
    metadata = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
        "xgboost_available": HAS_XGB,
        "xgboost_version": xgboost.__version__ if HAS_XGB else None,
        "shap_available": HAS_SHAP,
        "shap_version": shap.__version__ if HAS_SHAP else None,
        "random_state": args.seed,
        "target": args.target,
        "test_size": args.test_size,
        "n_iter_randomized_search": n_iter,
        "n_jobs": args.n_jobs,
        "data_path_used": str(args.data),
        "output_directory": str(out_dir),
    }
    save_json(metadata, out_dir / "run_metadata.json")

    # Compact summary table for manuscript drafting.
    summary_rows = [
        {
            "section": "Final test performance",
            "item": "Best model",
            "value": best_model_name,
        },
        {
            "section": "Final test performance",
            "item": "RMSE",
            "value": f"{final_metrics['RMSE']:.3f}",
        },
        {
            "section": "Final test performance",
            "item": "MAE",
            "value": f"{final_metrics['MAE']:.3f}",
        },
        {
            "section": "Final test performance",
            "item": "R2",
            "value": f"{final_metrics['R2']:.3f}",
        },
        {
            "section": "Final test performance",
            "item": "RMSE 95% CI",
            "value": f"{ci_low:.3f}–{ci_high:.3f}",
        },
        {
            "section": "Calibration",
            "item": "Slope / intercept",
            "value": f"{calibration_metrics['Calibration_slope']:.3f} / {calibration_metrics['Calibration_intercept']:.3f}",
        },
        {
            "section": "Tuning",
            "item": "Final tuned hyperparameters",
            "value": json.dumps(json_safe(final_best_params), ensure_ascii=False),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "manuscript_results_summary.csv", index=False)

    print("\nPLOS revision pipeline complete.")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print("Key files:")
    print("  - nested_cv_results.csv")
    print("  - final_metrics.json")
    print("  - subgroup_metrics_by_gender.csv")
    print("  - sex_stratified_model_metrics.csv")
    if not args.skip_recalibration:
        print("  - recalibration_sensitivity.csv")
    print("  - permutation_importance.csv")
    if not args.skip_shap:
        print("  - shap_direction_summary.csv")
    print("  - fig1_workflow_revision.png/pdf")
    print("  - fig2_calibration_plot_revised.png/pdf")
    if not args.skip_shap:
        print("  - fig3_shap_summary_revised.png/pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Revised PLOS ONE analysis pipeline for adolescent BFP explainable ML."
    )

    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file.")
    parser.add_argument("--out", type=str, default="artifacts_plos_revision", help="Output directory.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target column name.")
    parser.add_argument("--gender-col", type=str, default="Gender", help="Gender column name for subgroup analyses.")

    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Final test-set fraction.")
    parser.add_argument("--n-iter", type=int, default=15, help="RandomizedSearchCV iterations per model/fold.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for sklearn/XGBoost. Default is 1 for Windows stability.")
    parser.add_argument("--models", type=str, default=None, help="Optional comma-separated model subset, e.g. RandomForest or RandomForest,XGBoost.")
    parser.add_argument("--bootstrap-iter", type=int, default=1000, help="Bootstrap iterations for RMSE CI.")
    parser.add_argument("--perm-repeats", type=int, default=30, help="Permutation importance repeats.")
    parser.add_argument("--calibration-fraction", type=float, default=0.20, help="Calibration fraction inside training set.")

    parser.add_argument("--top-k-dependence", type=int, default=5, help="Number of SHAP dependence plots.")
    parser.add_argument("--shap-max-display", type=int, default=20, help="Max features in SHAP summary plot.")

    parser.add_argument("--quick", action="store_true", help="Use fewer search iterations/trees for testing.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis.")
    parser.add_argument("--skip-recalibration", action="store_true", help="Skip recalibration sensitivity analysis.")
    parser.add_argument("--no-clean-feature-names", action="store_true", help="Keep original column names exactly.")

    return parser.parse_args()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        run_pipeline(parse_args())
