import os
import json
import time
import logging
from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, precision_recall_curve,
    auc, confusion_matrix
)

# --- CONFIGURE LOGGING -----
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('BaseModelStudy')

# --- DIRECTORIES -----
CLEAN_DIR = Path("data/cleaned/CIC-IDS2018")
RESULTS_DIR = Path("results/base_re_evaluation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OVERALL_CSV = RESULTS_DIR / "overall_metrics.csv"
PER_ATTACK_CSV = RESULTS_DIR / "per_attack_metrics.csv"
ROC_PR_DIR = RESULTS_DIR / "roc_pr_curves"
CM_DIR = RESULTS_DIR / "confusion_matrices"
for d in (ROC_PR_DIR, CM_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- MODEL & FEATURE CONFIG -----
MODELS = {
    "LogisticRegression": (LogisticRegression(solver='saga', max_iter=1000, random_state=42), {"C": [0.01, 0.1, 1]}),
    "DecisionTree": (DecisionTreeClassifier(random_state=42), {"max_depth": [5, 10, 20]}),
    "RandomForest": (RandomForestClassifier(random_state=42, n_jobs=1), {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}),
    "ExtraTrees": (ExtraTreesClassifier(random_state=42, n_jobs=1), {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=1), {"learning_rate": [0.1, 0.01], "n_estimators": [100, 200], "max_depth": [3, 5]}),
    "LightGBM": (LGBMClassifier(random_state=42, n_jobs=1, verbose=-1), {"learning_rate": [0.1, 0.01], "n_estimators": [100, 200], "num_leaves": [31, 50]}),
    "CatBoost": (CatBoostClassifier(verbose=0, random_state=42), {"depth": [4, 6], "iterations": [100, 200], "learning_rate": [0.1, 0.01]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
    "LinearSVC": (LinearSVC(max_iter=10000, random_state=42), {"C": [0.1, 1, 10]}),
    "MLP": (MLPClassifier(max_iter=500, random_state=42), {"hidden_layer_sizes": [(50,), (100,)], "alpha": [1e-4, 1e-3]})
}

FEATURE_CATEGORIES = {
    "flow_metrics": ["Flow Duration", "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"],
    "packet_size_stats": ["Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg"],
    "timing_iat": ["Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min"],
    "flags_and_protocol": ["FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Protocol", "Dst Port", "Init Fwd Win Byts", "Init Bwd Win Byts"],
    "rates_and_ratios": ["Down/Up Ratio", "Fwd Pkts/s", "Bwd Pkts/s", "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg"],
    "connection_activity": ["Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts", "Fwd Act Data Pkts", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
}

logger.info(f"Loaded {len(MODELS)} models and {len(FEATURE_CATEGORIES)} feature groups")

# --- DATA LOADING & CACHING -----
@lru_cache(maxsize=None)
def load_cleaned_csv(fp: str) -> pd.DataFrame:
    logger.debug(f"Loading CSV: {fp}")
    df = pd.read_csv(fp, low_memory=False)
    logger.debug(f"Loaded {fp}: {df.shape[0]} rows, {df.shape[1]} cols")
    return df

# --- SAMPLING FUNCTIONS -----
def apply_downsample(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    logger.debug(f"Downsampling to fraction={frac}, original rows={len(df)}")
    df_sampled = df.sample(frac=frac, random_state=42)
    logger.debug(f"Downsampled rows={len(df_sampled)}")
    return df_sampled

# --- EVALUATION FUNCTION -----
def evaluate_model(
    file_fp: Path,
    feature_group: str,
    feats: list,
    model_name: str,
    model_grid: tuple
):
    model, param_grid = model_grid
    logger.info(f"Evaluating {model_name} on {file_fp.name} with features {feature_group}")
    start_time = time.time()

    results = []
    per_attack_results = []

    try:
        df = load_cleaned_csv(str(file_fp))
        X = df[feats]
        y = df['Label']
    except Exception as e:
        logger.error(f"Failed to load data for {file_fp}: {e}")
        return results, per_attack_results

    for protocol in ['same_day', 'cross_day']:
        try:
            logger.info(f"Starting protocol = {protocol} for {model_name}")
            # Split & sampling
            if protocol == 'same_day':
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.3, stratify=y, random_state=42
                )
                tmp = apply_downsample(pd.concat([X_tr, y_tr], axis=1), frac=0.5)
                X_train = tmp.drop('Label', axis=1)
                y_train = tmp['Label']
                X_test, y_test = X_te, y_te
            else:
                # Cross-day: train on all other days
                all_fps = sorted(CLEAN_DIR.glob('*.csv'))
                train_dfs = []
                for fp in all_fps:
                    if fp == file_fp: continue
                    try:
                        train_dfs.append(load_cleaned_csv(str(fp)))
                    except Exception as e:
                        logger.warning(f"Could not load {fp}: {e}")
                df_tr = pd.concat(train_dfs, ignore_index=True)
                tmp = apply_downsample(df_tr, frac=0.2)
                X_train = tmp[feats]
                y_train = tmp['Label']
                df_te = load_cleaned_csv(str(file_fp))
                X_test = df_te[feats]
                y_test = df_te['Label']

            # Class weighting
            neg, pos = (y_train==0).sum(), (y_train==1).sum()
            total = neg + pos
            class_weight = {0: total/(2*neg), 1: total/(2*pos)}
            logger.debug(f"Class weights: {class_weight}")

            # Clone base estimator and set class weights
            est = clone(model)
            if hasattr(est, 'class_weight'):
                est.set_params(class_weight=class_weight)

            # Grid search
            gs = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=1)
            logger.debug(f"Starting GridSearchCV for {model_name}")
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            logger.info(f"Best params: {gs.best_params_}")

            # Predictions
            if hasattr(best, 'predict_proba'):
                logger.info(f"Using predict_proba for {model_name}")
                y_scores = best.predict_proba(X_test)[:, 1]
            elif hasattr(best, 'decision_function'):
                logger.info(f"Using decision_function for {model_name}")
                y_scores = best.decision_function(X_test)
            else:
                logger.info(f"Using predict for {model_name}")
                y_scores = best.predict(X_test) 
            y_pred = best.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            pr, re, _ = precision_recall_curve(y_test, y_scores)
            pr_auc = auc(re, pr)
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Metrics for {model_name} on {protocol}: F1={f1:.3f}, ROC-AUC={roc_auc:.3f}")

            # Save intermediate outputs
            pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
                ROC_PR_DIR / f"{file_fp.stem}_{feature_group}_{model_name}_{protocol}_roc.csv",
                index=False
            )
            pd.DataFrame({'precision': pr, 'recall': re}).to_csv(
                ROC_PR_DIR / f"{file_fp.stem}_{feature_group}_{model_name}_{protocol}_pr.csv",
                index=False
            )
            pd.DataFrame(cm).to_csv(
                CM_DIR / f"{file_fp.stem}_{feature_group}_{model_name}_{protocol}_cm.csv",
                index=False
            )

            # Record results
            results.append({
                'file': file_fp.name,
                'feature_group': feature_group,
                'model': model_name,
                'protocol': protocol,
                'best_params': json.dumps(gs.best_params_),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
            })

            # Per-attack breakdown
            df_test = load_cleaned_csv(str(file_fp)).loc[y_test.index]
            df_test['_pred'] = y_pred
            for atk in df_test['Attack Type'].unique():
                mask = df_test['Attack Type'] == atk
                if mask.sum() < 2:
                    continue
                y_true_atk = df_test.loc[mask, 'Label']
                y_pred_atk = df_test.loc[mask, '_pred']
                per_attack_results.append({
                    'file': file_fp.name,
                    'feature_group': feature_group,
                    'model': model_name,
                    'protocol': protocol,
                    'attack_type': atk,
                    'accuracy': accuracy_score(y_true_atk, y_pred_atk),
                    'precision': precision_score(y_true_atk, y_pred_atk, zero_division=0),
                    'recall': recall_score(y_true_atk, y_pred_atk, zero_division=0),
                    'f1': f1_score(y_true_atk, y_pred_atk, zero_division=0)
                })

        except Exception as e:
            logger.error(f"Error in {model_name}/{feature_group}/{protocol} on {file_fp.name}: {e}", exc_info=True)
            continue

    elapsed = time.time() - start_time
    logger.info(f"Completed {model_name}/{feature_group}/{file_fp.name} in {elapsed:.1f}s")
    return results, per_attack_results

# --- DISPATCH PARALLEL JOBS -----
if __name__ == '__main__':
    tasks = []
    for fp in sorted(CLEAN_DIR.glob("*.csv")):
        for group, feats in FEATURE_CATEGORIES.items():
            try:
                sample = pd.read_csv(fp, nrows=1, low_memory=False)
                valid_feats = [f for f in feats if f in sample.columns]
                if not valid_feats:
                    logger.debug(f"Skipping {group} for {fp.name} (no features present)")
                    continue
            except Exception as e:
                logger.warning(f"Could not read sample from {fp}: {e}")
                continue
            for model_name, (model, grid) in MODELS.items():
                tasks.append((fp, group, valid_feats, model_name, (model, grid)))
    logger.info(f"Dispatching {len(tasks)} tasks across 4 workers")

    all_results = Parallel(n_jobs=2, backend="threading")(delayed(evaluate_model)(*t) for t in tasks)

    # Collect & save
    overall, per_attack = zip(*all_results)
    pd.DataFrame([r for sub in overall for r in sub]).to_csv(OVERALL_CSV, index=False)
    pd.DataFrame([r for sub in per_attack for r in sub]).to_csv(PER_ATTACK_CSV, index=False)

    logger.info("Base model re-evaluation finished.")
