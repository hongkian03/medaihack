#!/usr/bin/env python3
"""
predict.py — Run a trained model on new / external data
=========================================================
Loads a model saved by train.py and generates predictions for any new
dataset that shares the same feature schema.

Supported model types
---------------------
    XGBoost      → weights/xgboost_model.json       (XGBoost native)
    Elastic Net  → weights/elastic_net_model.pkl    (joblib / sklearn Pipeline)
    Lasso LR     → weights/lasso_lr_model.pkl       (joblib / sklearn Pipeline)

Column order in the input file does not matter — alignment to the training
feature list is handled automatically.  Missing columns are filled with NaN.

USAGE
-----
    python predict.py --data /path/to/new_data.csv
    python predict.py --data /path/to/new_data.csv --model-name "Elastic Net"
    python predict.py --data /path/to/new_data.csv --model-name "Lasso LR" --out preds.csv

OUTPUT COLUMNS
--------------
    sample_id   — sample_id if present in input, otherwise row index
    prob_ati    — predicted probability of ATI (0–1)
    pred_label  — hard prediction: 0 = No ATI, 1 = ATI  (threshold 0.5)
    true_label  — ground-truth (0/1) if an 'ati' column is present; else omitted
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from utils import model_name_to_slug

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR      = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = _SCRIPT_DIR / "weights"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a trained ATI model on new proteomics data."
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to input CSV (same feature schema as training data)",
    )
    p.add_argument(
        "--model-name",
        default="XGBoost",
        choices=["XGBoost", "Lasso LR", "Elastic Net"],
        help="Which trained model to use for inference (default: XGBoost)",
    )
    p.add_argument(
        "--weights",
        default=str(_DEFAULT_WEIGHTS),
        help="Directory containing model weights (default: ./weights/)",
    )
    p.add_argument(
        "--out",
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading — dispatches on file extension
# ---------------------------------------------------------------------------

def load_model(model_name: str, weights_dir: Path):
    """
    Restore a trained model and its feature column list from disk.

    Dispatch logic
    --------------
    If <slug>_model.json exists  →  XGBoost native format
                                    loaded via XGBClassifier.load_model()

    If <slug>_model.pkl  exists  →  joblib format (sklearn Pipeline)
                                    loaded via joblib.load()
                                    restores the ENTIRE pipeline:
                                    fitted StandardScaler + LogisticRegression

    Raises FileNotFoundError when neither artefact is found, with a
    message pointing the user to train.py.
    """
    slug     = model_name_to_slug(model_name)
    json_path = weights_dir / f"{slug}_model.json"
    pkl_path  = weights_dir / f"{slug}_model.pkl"

    if json_path.exists():
        logging.info(f"Loading model (XGBoost JSON) : {json_path}")
        model = XGBClassifier()
        model.load_model(str(json_path))

    elif pkl_path.exists():
        logging.info(f"Loading model (joblib pkl)   : {pkl_path}")
        model = joblib.load(pkl_path)

    else:
        raise FileNotFoundError(
            f"No saved model found for '{model_name}' in {weights_dir}.\n"
            f"Looked for:\n  {json_path}\n  {pkl_path}\n"
            f"Run:  python train.py --model-name '{model_name}'"
        )

    # Load the matching feature column list
    features_path = weights_dir / f"{slug}_feature_cols.json"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature column list not found: {features_path}\n"
            f"Run:  python train.py --model-name '{model_name}'"
        )

    logging.info(f"Loading feature list : {features_path}")
    with open(features_path) as fh:
        feature_cols = json.load(fh)

    logging.info(f"Model expects {len(feature_cols)} features")
    return model, feature_cols


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, pd.Series]:
    """
    Align a new DataFrame to the feature columns expected by the model.

    Columns present in feature_cols but absent from df are filled with NaN.
    XGBoost handles NaN natively; the sklearn Pipelines propagate NaN through
    StandardScaler (mean imputation is NOT applied — ensure upstream data
    quality is consistent with training).

    Returns
    -------
    X   : np.ndarray  shape (n_samples, n_features)
    ids : pd.Series   sample identifiers (sample_id column or RangeIndex)
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logging.warning(
            f"{len(missing)} feature column(s) absent from input — filled with NaN. "
            f"First few: {missing[:5]}"
        )

    X   = df.reindex(columns=feature_cols).values.astype(float)
    ids = (
        df["sample_id"]
        if "sample_id" in df.columns
        else pd.RangeIndex(len(df))
    )
    return X, ids


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_predict(
    model,
    X: np.ndarray,
    ids,
    y_true=None,
) -> pd.DataFrame:
    """
    Generate ATI predictions for a feature matrix.

    Both XGBClassifier and sklearn Pipelines expose .predict_proba(),
    so this function requires no model-type branching.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    out = pd.DataFrame({
        "sample_id" : ids,
        "prob_ati"  : y_prob,
        "pred_label": y_pred,
    })
    if y_true is not None:
        out["true_label"] = np.asarray(y_true)
    return out


# ---------------------------------------------------------------------------
# Optional evaluation (only when ground-truth labels are present)
# ---------------------------------------------------------------------------

def evaluate(results: pd.DataFrame, model_name: str) -> None:
    """Print classification metrics when ground-truth labels are available."""
    from sklearn.metrics import classification_report, roc_auc_score, log_loss

    y_true = results["true_label"].values
    y_pred = results["pred_label"].values
    y_prob = results["prob_ati"].values

    print(f"\n{'=' * 50}")
    print(f"  Model   : {model_name}")
    print(f"  Samples : {len(results):,}  |  "
          f"No ATI: {(y_true == 0).sum()}  |  ATI: {(y_true == 1).sum()}")
    print(f"{'=' * 50}")

    if len(np.unique(y_true)) > 1:
        print(classification_report(y_true, y_pred, target_names=["No ATI", "ATI"]))
        print(f"AUC (ROC) : {roc_auc_score(y_true, y_prob):.3f}")
        print(f"Log loss  : {log_loss(y_true, y_prob):.3f}")
    else:
        print("  (Only one class present in labels — AUC not defined)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args        = parse_args()
    weights_dir = Path(args.weights)

    # Load model + feature list (format resolved automatically)
    model, feature_cols = load_model(args.model_name, weights_dir)

    # Load input data
    logging.info(f"Loading data: {args.data}")
    df = pd.read_csv(args.data, low_memory=False, na_values=[".", ""])
    logging.info(f"  {len(df):,} samples loaded")

    # Align features
    X, ids = prepare_features(df, feature_cols)

    # Extract ground-truth labels if present
    y_true = None
    if "ati" in df.columns:
        y_true = df["ati"].astype(int).values
        logging.info("Found 'ati' column — evaluation metrics will be computed")

    # Predict (same call regardless of model type)
    results = run_predict(model, X, ids, y_true=y_true)

    # Evaluate if labels are available
    if y_true is not None:
        evaluate(results, args.model_name)

    # Save
    results.to_csv(args.out, index=False)
    logging.info(f"Predictions saved to: {args.out}")


if __name__ == "__main__":
    main()