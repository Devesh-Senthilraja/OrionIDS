import logging
from pathlib import Path
import json

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AblationStudy")

# --- Directories (adjust paths if needed) ---
BASE_PRED_DIR = Path("results/meta_re_evaluation/base_predictions")
OUTPUT_DIR    = Path("results/ablation_study")
ROC_PR_DIR    = OUTPUT_DIR / "roc_pr_curves"
CM_DIR        = OUTPUT_DIR / "confusion_matrices"
for d in (OUTPUT_DIR, ROC_PR_DIR, CM_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Base classifiers used in stacking ---
BASE_CLASSIFIERS = ["DecisionTree","RandomForest","XGBoost","LightGBM"]

# --- Fixed MLP meta-model (best params) ---
META_MODEL = MLPClassifier(max_iter=500, random_state=42, alpha=0.0001, hidden_layer_sizes=(50,))

def save_results(name, y_true, y_pred, y_score):
    """Compute metrics, save ROC/PR & CM, and return summary dict."""
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    pr, re, _  = precision_recall_curve(y_true, y_score)
    pr_auc = auc(re, pr)
    cm     = confusion_matrix(y_true, y_pred)

    # save curves
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(ROC_PR_DIR/f"{name}_roc.csv", index=False)
    pd.DataFrame({"precision":pr,"recall":re}).to_csv(ROC_PR_DIR/f"{name}_pr.csv", index=False)
    pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]) \
      .to_csv(CM_DIR/f"{name}_cm.csv")

    return {
        "setup": name,
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc
    }

def run_ablation(day: str, protocol: str, drop: str):
    """
    Ablation for one (day, protocol, drop_base) combination.
    drop=None runs the full 4-model stack for comparison.
    """
    # Load cached base preds
    # Files saved by Step 2: 
    #   <day>_<protocol>_train_base_preds.csv
    #   <day>_<protocol>_test_base_preds.csv
    train_df = pd.read_csv(BASE_PRED_DIR/f"{day}_{protocol}_train_base_preds.csv")
    test_df  = pd.read_csv(BASE_PRED_DIR/f"{day}_{protocol}_test_base_preds.csv")

    # Build feature matrix: all pred_<model> except dropped one
    cols = [f"pred_{m}" for m in BASE_CLASSIFIERS if m != drop]
    X_tr = train_df[cols].values
    y_tr = train_df["y_true"].values
    X_te = test_df[cols].values
    y_te = test_df["y_true"].values

    # Train MLP
    meta = META_MODEL
    meta.fit(X_tr, y_tr)

    # Predict & score
    if hasattr(meta, "predict_proba"):
        y_score = meta.predict_proba(X_te)[:,1]
    else:
        y_score = meta.decision_function(X_te)
    y_pred = meta.predict(X_te)

    # Name the run
    name = f"{day}_{protocol}_ablate_{drop or 'None'}"
    logger.info(f"Running {name}")

    # Save metrics + artifacts
    result = save_results(name, y_te, y_pred, y_score)
    return result

if __name__=="__main__":
    # Gather all days & protocols for which we have base preds
    files = list(BASE_PRED_DIR.glob("*_train_base_preds.csv"))
    tasks = []
    for fp in files:
        stem = fp.stem  # e.g. "02-14-2018_same_day_train_base_preds"
        parts = stem.split('_')
        day        = parts[0]                      # 02-22-2018
        protocol   = '_'.join(parts[1:3])          # same_day  or  cross_day
        # Full-stack baseline
        tasks.append((day, protocol, None))
        # Ablate each base
        for m in BASE_CLASSIFIERS:
            tasks.append((day, protocol, m))

    # Execute in parallel
    logger.info(f"Dispatching {len(tasks)} ablation tasks")
    results = Parallel(n_jobs=1)(
        delayed(run_ablation)(day, protocol, drop) for day, protocol, drop in tasks
    )

    # Save a summary CSV
    pd.DataFrame(results).to_csv(OUTPUT_DIR/"ablation_overall_metrics.csv", index=False)
    logger.info("Ablation study complete.")