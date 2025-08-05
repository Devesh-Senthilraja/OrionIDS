import os
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)

# Base learners
from sklearn.tree import   DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import    LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import         XGBClassifier
from lightgbm import        LGBMClassifier
from catboost import        CatBoostClassifier

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MetaModelStudy")

# --- Directories ---
CLEAN_DIR      = Path("data/cleaned/CIC-IDS2018")              # cleaned per-day CSVs
META_DIR       = Path("results/meta_re_evaluation")       # meta eval outputs
BASE_PRED_DIR  = META_DIR / "base_predictions"            # where we cache base preds
META_PRED_DIR = META_DIR / "meta_predictions"
ROC_PR_DIR     = META_DIR / "roc_pr_curves"
CM_DIR         = META_DIR / "confusion_matrices"

for d in (META_DIR, BASE_PRED_DIR, ROC_PR_DIR, CM_DIR, META_PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Feature set for base models (flags & protocol features) ---
FLAGS_PROTOCOL_FEATURES = ["FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Protocol", "Dst Port", "Init Fwd Win Byts", "Init Bwd Win Byts"]

# --- The 4 selected base classifiers (use your best hyperparameters) ---
BASE_MODELS = {
    "DecisionTree":   DecisionTreeClassifier(random_state=42, max_depth=10),
    "RandomForest":   RandomForestClassifier(n_estimators=200, random_state=42, max_depth=None),
    "XGBoost":        XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, max_depth=3, n_estimators=100, learning_rate=0.1),
    "LightGBM":       LGBMClassifier(random_state=42, learning_rate=0.01, n_estimators=100, num_leaves=31, verbose=-1),
}

# --- All 10 candidate meta-models (default params) ---
META_MODELS = {
    "DecisionTree":       DecisionTreeClassifier(random_state=42, max_depth=10),
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=42, max_depth=None),
    "ExtraTrees":         ExtraTreesClassifier(n_estimators=100, random_state=42, max_depth=None),
    "XGBoost":            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, max_depth=3, n_estimators=100, learning_rate=0.1),
    "LightGBM":           LGBMClassifier(random_state=42, learning_rate=0.01, n_estimators=100, num_leaves=31, verbose=-1),
    "CatBoost":           CatBoostClassifier(verbose=0, random_state=42, depth=4, iterations=100, learning_rate=0.1),
    "LogisticRegression": LogisticRegression(solver="saga", max_iter=2000, random_state=42, C=0.01),
    "MLP":                MLPClassifier(max_iter=500, random_state=42, alpha=0.0001, hidden_layer_sizes=(50,)),
    "KNN":                KNeighborsClassifier(n_neighbors=3),
    "LinearSVC":          LinearSVC(max_iter=10000, random_state=42, C=0.1),
}

# --- Utility functions ---
def downsample(X, y, frac):
    """Randomly downsample X,y together by fraction frac."""
    if frac >= 1.0:
        return X, y
    df = pd.concat([X, y.rename("y")], axis=1)
    df2 = df.sample(frac=frac, random_state=42).reset_index(drop=True)
    return df2[X.columns], df2["y"]

def get_class_weight(y):
    """Compute a class_weight dict for {0,1} classes."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

def save_metrics_and_artifacts(name, y_true, y_pred, y_scores):
    """Compute metrics, ROC/PR curves, CM, and save them under ROC_PR_DIR/CM_DIR."""
    # overall metrics
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    pr, re, _ = precision_recall_curve(y_true, y_scores)
    pr_auc   = auc(re, pr)
    cm       = confusion_matrix(y_true, y_pred)

    # save curves
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(ROC_PR_DIR/f"{name}_roc.csv", index=False)
    pd.DataFrame({"precision":pr,"recall":re}).to_csv(ROC_PR_DIR/f"{name}_pr.csv", index=False)
    pd.DataFrame(cm).to_csv(CM_DIR/f"{name}_cm.csv", index=False)

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_scores
    })
    pred_df.to_csv(META_PRED_DIR / f"{name}_raw.csv", index=False)

    # return dict for summary
    return {
        "setup": name,
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc
    }

# --- Step A: Generate & cache base-model predictions ---
def generate_base_predictions(day, protocol):
    """
    Train the 4 base models on 'flags_and_protocol' features and
    save both train/test predictions for use by all meta-models.
    """
    logger.info(f"Generating base preds for {day} | {protocol}")
    df = pd.read_csv(CLEAN_DIR/f"{day}.csv", low_memory=False)
    y = df["Label"]
    X = df[FLAGS_PROTOCOL_FEATURES]

    # split
    if protocol == "same_day":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        X_tr, y_tr = downsample(X_tr, y_tr, frac=0.5)
    else:  # cross-day
        # train on all other days
        other_days = [p.stem for p in CLEAN_DIR.glob("*.csv") if p.stem != day]
        dfs = [pd.read_csv(CLEAN_DIR/f"{d}.csv", low_memory=False) for d in other_days]
        df_tr = pd.concat(dfs, ignore_index=True)
        X_tr = df_tr[FLAGS_PROTOCOL_FEATURES]
        y_tr = df_tr["Label"]
        X_tr, y_tr = downsample(X_tr, y_tr, frac=0.2)
        X_te, y_te = X, y

    # container for all preds
    df_tr_preds = pd.DataFrame({"y_true": y_tr})
    df_te_preds = pd.DataFrame({"y_true": y_te})

    cw = get_class_weight(y_tr)

    for name, model in BASE_MODELS.items():
        m = model
        if hasattr(m, "class_weight"):
            m.set_params(class_weight=cw)
        m.fit(X_tr, y_tr)

        # get scores & labels
        if hasattr(m, "predict_proba"):
            tr_sc = m.predict_proba(X_tr)[:,1]
            te_sc = m.predict_proba(X_te)[:,1]
        elif hasattr(m, "decision_function"):
            tr_sc = m.decision_function(X_tr)
            te_sc = m.decision_function(X_te)
        else:
            tr_sc = m.predict(X_tr)
            te_sc = m.predict(X_te)

        tr_lb = m.predict(X_tr)
        te_lb = m.predict(X_te)

        df_tr_preds[f"pred_{name}"]  = tr_sc
        df_tr_preds[f"label_{name}"] = tr_lb
        df_te_preds[f"pred_{name}"]  = te_sc
        df_te_preds[f"label_{name}"] = te_lb

    # save to disk
    tr_fn = BASE_PRED_DIR/f"{day}_{protocol}_train_base_preds.csv"
    te_fn = BASE_PRED_DIR/f"{day}_{protocol}_test_base_preds.csv"
    df_tr_preds.to_csv(tr_fn, index=False)
    df_te_preds.to_csv(te_fn, index=False)
    logger.info(f"  Saved base preds → {tr_fn.name}, {te_fn.name}")

# --- Step B: Voting evaluation ---
def evaluate_voting(day, protocol):
    df_tr = pd.read_csv(BASE_PRED_DIR/f"{day}_{protocol}_train_base_preds.csv")
    df_te = pd.read_csv(BASE_PRED_DIR/f"{day}_{protocol}_test_base_preds.csv")
    # features are 'pred_<Model>'
    cols = [c for c in df_tr.columns if c.startswith("pred_")]
    y_te = df_te["y_true"]

    # Hard voting: majority label
    votes = df_te[[f"label_{m}" for m in BASE_MODELS]].values
    hard_pred = (votes.sum(axis=1) >= (len(BASE_MODELS)/2)).astype(int)
    hard_score = hard_pred  # no soft scores
    name = f"{day}_{protocol}_voting_hard"
    res_h = save_metrics_and_artifacts(name, y_te, hard_pred, hard_score)

    # Soft voting: average probability
    probs = df_te[cols].values
    soft_score = probs.mean(axis=1)
    soft_pred  = (soft_score >= 0.5).astype(int)
    name = f"{day}_{protocol}_voting_soft"
    res_s = save_metrics_and_artifacts(name, y_te, soft_pred, soft_score)

    return [res_h, res_s]

# --- Step C: Meta-model evaluation ---
def evaluate_meta_models(day, protocol):
    df_tr = pd.read_csv(BASE_PRED_DIR/f"{day}_{protocol}_train_base_preds.csv")
    df_te = pd.read_csv(BASE_PRED_DIR/f"{day}_{protocol}_test_base_preds.csv")
    cols = [c for c in df_tr.columns if c.startswith("pred_")]
    y_tr, y_te = df_tr["y_true"], df_te["y_true"]
    X_tr, X_te = df_tr[cols], df_te[cols]

    results = []
    for name, model in META_MODELS.items():
        logger.info(f"Meta-model {name} on {day}|{protocol}")
        m = model
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)

        if hasattr(m, "predict_proba"):
            y_score = m.predict_proba(X_te)[:,1]
        elif hasattr(m, "decision_function"):
            y_score = m.decision_function(X_te)
        else:
            y_score = y_pred

        setup = f"{day}_{protocol}_meta_{name}"
        res = save_metrics_and_artifacts(setup, y_te, y_pred, y_score)
        results.append(res)
    return results

# --- Orchestrate all steps in parallel ---
if __name__ == "__main__":
    days = [p.stem for p in CLEAN_DIR.glob("*.csv")]
    protocols = ["same_day", "cross_day"]

    tasks = []
    for day in days:
        for protocol in protocols:
            tasks.append((day, protocol))

    # 1) generate base preds
    Parallel(n_jobs=1)(
        delayed(generate_base_predictions)(day, protocol) for day, protocol in tasks
    )

    # 2) evaluate voting
    voting_results = Parallel(n_jobs=1)(
        delayed(evaluate_voting)(day, protocol) for day, protocol in tasks
    )

    # 3) evaluate all meta-models
    meta_results = Parallel(n_jobs=1)(
        delayed(evaluate_meta_models)(day, protocol) for day, protocol in tasks
    )

    # flatten & save summaries
    all_v = [r for sub in voting_results for r in sub]
    all_m = [r for sub in meta_results   for r in sub]

    pd.DataFrame(all_v).to_csv(META_DIR/"voting_overall_metrics.csv", index=False)
    pd.DataFrame(all_m).to_csv(META_DIR/"meta_overall_metrics.csv",   index=False)

    logger.info("Meta model re-evaluation complete.")
