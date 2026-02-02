#!/usr/bin/env python3
"""
train_ml_models.py

Reproducible training/evaluation script for traditional ML baselines used in the MIE 2026 paper.
- Logistic Regression (LR)
- Decision Tree (DT)
- Random Forest (RF)
- XGBoost (XGBClassifier)

This script intentionally contains NO patient data and makes no assumptions about cohort extraction.
It expects already-prepared tabular datasets (e.g., after temporal filtering, one-hot encoding, and feature selection).

Key hyperparameters match the paper:
- LR: penalty=l2, C=1.0, class_weight=balanced, solver=lbfgs, max_iter=1000
- DT: max_depth=8, min_samples_split=20, min_samples_leaf=10, class_weight=balanced
- RF: n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=5, class_weight=balanced
- XGB: n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
       scale_pos_weight=neg/pos, eval_metric=logloss

Outputs:
- metrics.csv : one row per model with Accuracy/Precision/Recall/F1/AUC
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# xgboost is used in the paper
from xgboost import XGBClassifier


RANDOM_STATE = 42


@dataclass
class ModelResult:
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


def _validate_binary(y: pd.Series) -> None:
    vals = set(pd.unique(y.dropna()))
    if not vals.issubset({0, 1}):
        raise ValueError(f"Label column must be binary 0/1. Found: {sorted(vals)}")


def load_split_data(
    train_path: Path,
    test_path: Path,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load pre-split train/test tables (CSV or Parquet)."""

    def _load(p: Path) -> pd.DataFrame:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        if p.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        raise ValueError(f"Unsupported file type: {p}")

    train_df = _load(train_path)
    test_df = _load(test_path)

    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"label_col='{label_col}' must exist in both train and test files.")

    y_train = train_df[label_col].astype(int)
    y_test = test_df[label_col].astype(int)
    _validate_binary(y_train)
    _validate_binary(y_test)

    X_train = train_df.drop(columns=[label_col])
    X_test = test_df.drop(columns=[label_col])

    # Safety: avoid NaNs breaking estimators
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    return X_train, y_train, X_test, y_test


def bootstrap_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_bootstrap: int,
    seed: int = RANDOM_STATE,
) -> Dict[str, float]:
    """
    Returns point estimates computed on the full test set.
    (Bootstrapping is used in the paper to estimate variability; this function can also
    be extended to compute SDs if needed.)
    """
    # Point estimates (space-saving for paper)
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    if y_proba is not None:
        out["auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        out["auc"] = float("nan")
    return out


def train_and_eval_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    continuous_cols: Tuple[str, ...] = ("age_at_cutoff", "bmi_at_baseline"),
) -> pd.DataFrame:
    """Train and evaluate LR/DT/RF/XGB with paper hyperparameters."""
    results = []

    # -------- Logistic Regression (scale only continuous cols) --------
    scaler = StandardScaler()
    X_train_lr = X_train.copy()
    X_test_lr = X_test.copy()
    cont = [c for c in continuous_cols if c in X_train.columns]
    if cont:
        X_train_lr[cont] = scaler.fit_transform(X_train_lr[cont])
        X_test_lr[cont] = scaler.transform(X_test_lr[cont])

    lr = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    lr.fit(X_train_lr, y_train)
    lr_pred = lr.predict(X_test_lr)
    lr_proba = lr.predict_proba(X_test_lr)[:, 1]
    lr_m = bootstrap_eval(y_test.values, lr_pred, lr_proba, n_bootstrap=100)
    results.append(ModelResult("Logistic Regression (LR)", **lr_m))

    # -------- Decision Tree --------
    dt = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:, 1] if hasattr(dt, "predict_proba") else None
    dt_m = bootstrap_eval(y_test.values, dt_pred, dt_proba, n_bootstrap=100)
    results.append(ModelResult("Decision Tree (DT)", **dt_m))

    # -------- Random Forest --------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_m = bootstrap_eval(y_test.values, rf_pred, rf_proba, n_bootstrap=100)
    results.append(ModelResult("Random Forest (RF)", **rf_m))

    # -------- XGBoost --------
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_m = bootstrap_eval(y_test.values, xgb_pred, xgb_proba, n_bootstrap=100)
    results.append(ModelResult("XGBoost (GB)", **xgb_m))

    return pd.DataFrame([asdict(r) for r in results])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True, help="Path to train table (CSV or Parquet).")
    ap.add_argument("--test", type=Path, required=True, help="Path to test table (CSV or Parquet).")
    ap.add_argument("--label-col", type=str, default="label", help="Name of binary label column (0/1).")
    ap.add_argument("--outdir", type=Path, default=Path("outputs_ml"), help="Output directory.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_split_data(args.train, args.test, args.label_col)
    metrics_df = train_and_eval_models(X_train, y_train, X_test, y_test)
    out_csv = args.outdir / "metrics.csv"
    metrics_df.to_csv(out_csv, index=False)

    # Also print a compact table for quick inspection
    cols = ["model", "accuracy", "precision", "recall", "f1", "auc"]
    print(metrics_df[cols].to_string(index=False))
    print(f"\nSaved: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
