import json
import logging
import re
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, auc, confusion_matrix
)

# --- Config ---
CLEAN_DIR      = Path("cleaned/CIC-IDS2018")                                       # cleaned per-day CSVs
BASE_PRED_DIR  = Path("results/meta_re_evaluation/base_predictions")               # cached base preds
OUT_DIR        = Path("results/meta_feature_selection")                            # output folder
ROC_DIR        = OUT_DIR / "roc_pr_curves"
CM_DIR         = OUT_DIR / "confusion_matrices"

for d in (OUT_DIR, ROC_DIR, CM_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Candidate raw features to test
CANDIDATE_META_FEATURES = [
    # Flow metrics
    "Flow Duration", "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",

    # Packet size stats
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",

    # Timing / IAT
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",

    # Flags & protocol
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Protocol", "Dst Port", "Init Fwd Win Byts", "Init Bwd Win Byts",

    # Rates & ratios
    "Down/Up Ratio", "Fwd Pkts/s", "Bwd Pkts/s", "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",

    # Connection activity
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts", "Fwd Act Data Pkts", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
CONSOLIDATED_FEATURES = []

# Base classifiers whose predictions we stack
BASE_CLASSIFIERS = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]

# Fixed MLP meta-model (best params)
META_MODEL = MLPClassifier(
    hidden_layer_sizes=(50,),
    alpha=1e-4,
    max_iter=500,
    random_state=42
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("MetaFeatureStudy")

# --- Helpers ---
def make_safe_tag(raw_feats):
    """Sanitize feature list into a filesystem-safe string."""
    if not raw_feats:
        return "none"
    # replace non-word chars with underscore, then join
    safe = [re.sub(r"[^\w]+", "_", feat).strip("_") for feat in raw_feats]
    return "_".join(safe)

def save_metrics(name, y_true, y_pred, y_score, raw_feats):
    """Compute & save metrics, curves, and confusion matrix for one setup."""
    log.debug(f"Saving metrics and artifacts for {name}")
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc  = auc(fpr, tpr)
    pr, re, _= precision_recall_curve(y_true, y_score)
    pr_auc   = auc(re, pr)
    cm       = confusion_matrix(y_true, y_pred)

    pd.DataFrame({"fpr": fpr, "tpr": tpr}) \
      .to_csv(ROC_DIR / f"{name}_roc.csv", index=False)
    pd.DataFrame({"precision": pr, "recall": re}) \
      .to_csv(ROC_DIR / f"{name}_pr.csv", index=False)
    pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]) \
      .to_csv(CM_DIR / f"{name}_cm.csv")

    log.info(f"Saved {name}: F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    return {
        "setup": name,
        "raw_features": json.dumps(raw_feats),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

def run_test(day, protocol, raw_feats):
    """Train & evaluate MLP stack including the given raw_feats."""
    tag = make_safe_tag(raw_feats)
    setup_name = f"{day}_{protocol}_mf_{tag}"
    log.info(f"Starting meta-feature test: {setup_name}")
    try:
        train_df = pd.read_csv(BASE_PRED_DIR / f"{day}_{protocol}_train_base_preds.csv")
        test_df  = pd.read_csv(BASE_PRED_DIR / f"{day}_{protocol}_test_base_preds.csv")

        pred_cols = [f"pred_{m}" for m in BASE_CLASSIFIERS]
        X_tr = train_df[pred_cols].copy()
        X_te = test_df[pred_cols].copy()
        y_tr = train_df["y_true"].values
        y_te = test_df["y_true"].values

        if raw_feats:
            if protocol == "same_day":
                # single-day case: raw_df has the same rows as train_df/test_df before splitting
                raw_df = pd.read_csv(CLEAN_DIR / f"{day}.csv", low_memory=False)
                for feat in raw_feats:
                    X_tr[feat] = raw_df[feat].iloc[ train_df.index ].values
                    X_te[feat] = raw_df[feat].iloc[ test_df.index  ].values

            else:  # cross_day
                # 1) build raw_tr by concatenating other days *and* downsampling to 20%
                other_days = [p.stem for p in CLEAN_DIR.glob("*.csv") if p.stem != day]
                raw_full = pd.concat(
                    (pd.read_csv(CLEAN_DIR / f"{d}.csv", low_memory=False) for d in other_days),
                    ignore_index=True
                )
                # exactly mirror the 20% downsampling used during base-pred generation
                raw_tr = raw_full.sample(frac=0.2, random_state=42).reset_index(drop=True)
                # now raw_tr has same length/order as your train_df
                for feat in raw_feats:
                    X_tr[feat] = raw_tr[feat].values

                # 3) build raw_te only from the current day
                raw_te = pd.read_csv(CLEAN_DIR / f"{day}.csv", low_memory=False)
                for feat in raw_feats:
                    X_te[feat] = raw_te[feat].values

        clf = META_MODEL.fit(X_tr, y_tr)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_te)[:,1]
        else:
            y_score = clf.decision_function(X_te)
        y_pred = clf.predict(X_te)

        return save_metrics(setup_name, y_te, y_pred, y_score, raw_feats)
    except Exception as e:
        log.error(f"Error during {setup_name}: {e}", exc_info=True)
        return {
            "setup": setup_name,
            "raw_features": json.dumps(raw_feats),
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan
        }

# --- Main Dispatch ---
if __name__ == "__main__":
    days = [p.stem for p in CLEAN_DIR.glob("*.csv")]
    protocols = ["same_day", "cross_day"]

    tasks = []
    for day in days:
        for protocol in protocols:
            tasks.append((day, protocol, []))  # Baseline: no raw features
            for feat in CANDIDATE_META_FEATURES:
                tasks.append((day, protocol, [feat]))  # Single-feature tests
            tasks.append((day, protocol, CONSOLIDATED_FEATURES))  # Consolidated set

    log.info(f"Dispatching {len(tasks)} meta-feature experiments")
    results = Parallel(n_jobs=20)(
        delayed(run_test)(day, protocol, raw_feats) for day, protocol, raw_feats in tasks
    )

    # Save summary table
    summary_file = OUT_DIR / "meta_feature_selection_metrics.csv"
    pd.DataFrame(results).to_csv(summary_file, index=False)
    log.info(f"Meta-feature study complete. Results saved to {summary_file}")