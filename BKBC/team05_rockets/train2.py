#!/usr/bin/env python3
"""
train.py — Train a model on BKBC data and save weights
========================================================
Loads the combat-corrected BKBC training CSV, trains the requested model
on ALL samples (no train/test split), and saves the model weights.

Save formats
------------
    XGBoost              →  weights/xgboost_model.json       (XGBoost native)
    Elastic Net / Lasso  →  weights/elastic_net_model.pkl    (joblib / sklearn)

USAGE
-----
    python train.py --model-name "Elastic Net" --data /path/to/train.csv
    python train.py --model-name "Lasso LR"    --data /path/to/train.csv
    python train.py --model-name "XGBoost"     --data /path/to/train.csv

OUTPUTS  (written to weights/<slug>_*)
-------
    weights/<slug>_model.json          — XGBoost only
    weights/<slug>_model.pkl           — sklearn Pipeline models
    weights/<slug>_feature_cols.json   — ordered feature column list
    weights/<slug>_metadata.json       — coefficients / params (human audit)
    weights/<slug>_feature_importance.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

from model import build_model
from preprocess import load_data, build_features_and_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR      = Path(__file__).resolve().parent
_DEFAULT_DATA    = _SCRIPT_DIR.parent.parent.parent / "BKBC" / "BKBC_train" / "train.csv"
_DEFAULT_WEIGHTS = _SCRIPT_DIR / "weights"


# ---------------------------------------------------------------------------
# Slug helper
# ---------------------------------------------------------------------------

def model_name_to_slug(model_name: str) -> str:
    """
    'Elastic Net' → 'elastic_net'
    'Lasso LR'    → 'lasso_lr'
    'XGBoost'     → 'xgboost'
    'TopK XGBoost' → 'topk_xgboost'
    """
    return model_name.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Save model — format dispatched on model type
# ---------------------------------------------------------------------------

def save_model(model, model_name: str, out_dir: Path) -> Path:
    """
    Persist a trained model in the format appropriate for its type.

    XGBClassifier
        → <slug>_model.json   via model.save_model()
          (XGBoost native JSON, can be loaded back with XGBClassifier.load_model())

    sklearn Pipeline / LogisticRegression / anything else
        → <slug>_model.pkl    via joblib.dump()
          (serialises the ENTIRE pipeline: fitted scaler state + coefficients)

    Returns the Path that was written.
    """
    slug = model_name_to_slug(model_name)

    if isinstance(model, XGBClassifier):
        path = out_dir / f"{slug}_model.json"
        model.save_model(str(path))
        logging.info(f"Saved model (XGBoost JSON) : {path}")
    else:
        path = out_dir / f"{slug}_model.pkl"
        joblib.dump(model, path)
        logging.info(f"Saved model (joblib pkl)   : {path}")

    return path


# ---------------------------------------------------------------------------
# Feature importance — works for both tree and linear models
# ---------------------------------------------------------------------------

def extract_feature_importance(
    model,
    feature_cols: list[str],
) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame sorted by descending importance.

    Tree models  → .feature_importances_  (gain)
    Linear models inside a Pipeline → named_steps['clf'].coef_
    Bare linear models              → .coef_

    Returns None if no signal is available.
    """
    # Tree-based (XGBoost, RandomForest, …)
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature":    feature_cols,
            "importance": model.feature_importances_,
            "type":       "gain",
        }).sort_values("importance", ascending=False)

    # Unwrap Pipeline to get the final estimator
    estimator = (
        model.named_steps.get("clf", model)
        if hasattr(model, "named_steps")
        else model
    )

    # Linear models (LogisticRegression, ElasticNet, …)
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_.ravel()
        return pd.DataFrame({
            "feature":     feature_cols,
            "importance":  np.abs(coef),
            "coefficient": coef,
            "type":        "abs_coefficient",
        }).sort_values("importance", ascending=False)

    return None


# ---------------------------------------------------------------------------
# Metadata sidecar — human audit only, NOT used by predict.py
# ---------------------------------------------------------------------------

def save_model_metadata(
    model,
    model_name: str,
    feature_cols: list[str],
    out_dir: Path,
) -> Path:
    """
    Write a JSON sidecar with coefficients and hyper-parameters.

    This file is for inspection / auditing only.
    predict.py loads the .json or .pkl, NOT this file.
    """
    slug = model_name_to_slug(model_name)
    meta: dict = {"model_name": model_name}

    estimator = (
        model.named_steps.get("clf", model)
        if hasattr(model, "named_steps")
        else model
    )

    if hasattr(estimator, "coef_"):
        coef = estimator.coef_.ravel().tolist()
        meta["coefficients"] = dict(zip(feature_cols, coef))
        meta["intercept"]    = (
            float(estimator.intercept_[0])
            if hasattr(estimator, "intercept_")
            else None
        )
        meta["params"] = estimator.get_params()

    path = out_dir / f"{slug}_metadata.json"
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    logging.info(f"Saved metadata (audit only): {path}")

    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a model on BKBC data and save weights."
    )
    p.add_argument(
        "--data",
        default=str(_DEFAULT_DATA),
        help="Path to training CSV (default: ../../BKBC_train/train.csv)",
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_WEIGHTS),
        help="Directory to save model weights (default: ./weights/)",
    )
    p.add_argument(
        "--model-name",
        default="XGBoost",
        choices=["XGBoost", "Lasso LR", "Elastic Net", "TopK XGBoost"],
        help="Which model to train (default: XGBoost)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = model_name_to_slug(args.model_name)

    # ── Load & preprocess ────────────────────────────────────────────────────
    df = load_data(args.data)
    X, y, feature_cols = build_features_and_labels(df)

    # ── Train ────────────────────────────────────────────────────────────────
    model = build_model(args.model_name)
    logging.info(f"Training '{args.model_name}' on {len(y):,} samples …")
    model.fit(X, y)

    # ── Sanity-check metrics (training set only) ─────────────────────────────
    y_prob = model.predict_proba(X)[:, 1]
    auc    = roc_auc_score(y, y_prob)
    ll     = log_loss(y, y_prob)

    print(f"\n{'=' * 50}")
    print(f"  Model     : {args.model_name}")
    print(f"  Samples   : {len(y):,}  (ATI={y.sum()}, No-ATI={(y==0).sum()})")
    print(f"  Features  : {len(feature_cols)}")
    print(f"  Train AUC : {auc:.3f}")
    print(f"  Train Loss: {ll:.3f}")
    print(f"{'=' * 50}")

    # ── Persist model (.json for XGBoost, .pkl for everything else) ──────────
    save_model(model, args.model_name, out_dir)

    # ── Feature columns ───────────────────────────────────────────────────────
    features_path = out_dir / f"{slug}_feature_cols.json"
    with open(features_path, "w") as fh:
        json.dump(feature_cols, fh)
    logging.info(f"Saved features : {features_path}")

    # ── Metadata / coefficients (audit only) ─────────────────────────────────
    save_model_metadata(model, args.model_name, feature_cols, out_dir)

    # ── Feature importance ────────────────────────────────────────────────────
    importance = extract_feature_importance(model, feature_cols)
    if importance is not None:
        csv_path = out_dir / f"{slug}_feature_importance.csv"
        importance.to_csv(csv_path, index=False)
        logging.info(f"Saved importance: {csv_path}")

        signal_label = (
            "abs(coefficient)" if "coefficient" in importance.columns else "gain"
        )
        display_cols = [
            c for c in ("feature", "importance", "coefficient")
            if c in importance.columns
        ]
        print(f"\nTop 10 features by {signal_label}:")
        print(importance.head(10)[display_cols].to_string(index=False))

    print(f"\nWeights saved to: {out_dir}/")
    print("Next step: bash predict.sh /path/to/new_data.csv")


if __name__ == "__main__":
    main()