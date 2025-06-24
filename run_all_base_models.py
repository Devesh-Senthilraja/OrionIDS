# run_all_base_models.py

# Robust driver to train and evaluate multiple classifiers on all cleaned CIC-IDS-2018 datasets
# across defined feature groups. Computes overall and per-attack metrics and saves results.

import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import time

# --- Configuration ---
CLEAN_DIR = Path("cleaned")
RESULTS_DIR = Path("results") / "base_models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OVERALL_CSV = RESULTS_DIR / "overall_metrics.csv"
PER_ATTACK_CSV = RESULTS_DIR / "per_attack_metrics.csv"

# Classifiers and parameter grids
MODELS = {
    "LogisticRegression": (LogisticRegression(solver='saga', max_iter=1000, random_state=42), {"C": [0.01, 0.1, 1]}),
    "DecisionTree": (DecisionTreeClassifier(random_state=42), {"max_depth": [5, 10, 20]}),
    "RandomForest": (RandomForestClassifier(random_state=42, n_jobs=1), {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}),
    "ExtraTrees": (ExtraTreesClassifier(random_state=42, n_jobs=1), {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=1), {"learning_rate": [0.1, 0.01], "n_estimators": [100, 200], "max_depth": [3, 5]}),
    "LightGBM": (LGBMClassifier(random_state=42, n_jobs=1), {"learning_rate": [0.1, 0.01], "n_estimators": [100, 200], "num_leaves": [31, 50]}),
    "CatBoost": (CatBoostClassifier(verbose=0, random_state=42), {"depth": [4, 6], "iterations": [100, 200], "learning_rate": [0.1, 0.01]}),
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
    "LinearSVC": (LinearSVC(max_iter=10000, random_state=42), {"C": [0.1, 1, 10]}),
    "MLP": (MLPClassifier(max_iter=500, random_state=42), {"hidden_layer_sizes": [(50,), (100,)], "alpha": [1e-4, 1e-3]})
}

# Feature categories (must match preprocessor)
FEATURE_CATEGORIES = {
    "flow_metrics": ["Flow Duration", "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"],
    "packet_size_stats": ["Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg"],
    "timing_iat": ["Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min"],
    "flags_and_protocol": ["FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Protocol", "Dst Port", "Init Fwd Win Byts", "Init Bwd Win Byts"],
    "rates_and_ratios": ["Down/Up Ratio", "Fwd Pkts/s", "Bwd Pkts/s", "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg"],
    "connection_activity": ["Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts", "Fwd Act Data Pkts", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
}

# Function to train and evaluate one combination
def run_experiment(params):
    file_path, group_name, feats, model_name, (estimator, grid) = params
    start_time = time.time()
    print(f"\n--- Starting {model_name} on {file_path.name}/{group_name} at {time.ctime()} ---")

    # Load data
    df = pd.read_csv(file_path, low_memory=False)
    X = df[feats]
    y = df['Label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    # Grid search
    print(f"Grid-searching {model_name} (params: {list(grid.keys())})...")
    gs = GridSearchCV(estimator, grid, cv=3, scoring='f1', n_jobs=1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    print(f"Best params for {model_name}: {gs.best_params_}")

    # Predict
    print(f"Evaluating {model_name}...")
    y_pred = best.predict(X_test)

    # Overall metrics
    overall = {
        'file': file_path.name,
        'attack_type': file_path.stem,
        'feature_group': group_name,
        'model': model_name,
        'best_params': json.dumps(gs.best_params_),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }

    # Per-attack metrics
    per_attack = []
    df_test = df.loc[y_test.index].copy()
    df_test['_pred'] = y_pred
    for atk in sorted(df_test['Attack Type'].unique()):
        mask = df_test['Attack Type'] == atk
        if mask.sum() < 2:
            continue
        pa_y = df_test.loc[mask, 'Label']
        pa_pred = df_test.loc[mask, '_pred']
        per_attack.append({
            'file': file_path.name,
            'attack_type': atk,
            'feature_group': group_name,
            'model': model_name,
            'accuracy': accuracy_score(pa_y, pa_pred),
            'f1_score': f1_score(pa_y, pa_pred),
            'precision': precision_score(pa_y, pa_pred),
            'recall': recall_score(pa_y, pa_pred)
        })

    elapsed = time.time() - start_time
    print(f"--- Completed {model_name} on {file_path.name}/{group_name} in {elapsed:.1f}s ---")
    return overall, per_attack

# Build task list
tasks = []
for file_path in sorted(CLEAN_DIR.glob("*.csv")):
    print(f"Queueing file: {file_path.name}")
    for group, feats in FEATURE_CATEGORIES.items():
        sample_cols = pd.read_csv(file_path, nrows=1, low_memory=False).columns
        avail = [f for f in feats if f in sample_cols]
        if not avail:
            print(f"  Skipping group {group}: no features present.")
            continue
        print(f"  Feature group: {group} ({len(avail)} features)")
        for model_name, model_def in MODELS.items():
            tasks.append((file_path, group, avail, model_name, model_def))

print(f"Total tasks to run: {len(tasks)}")

# Run in parallel
results = Parallel(n_jobs=3)(delayed(run_experiment)(t) for t in tasks)

# Separate results
overall_list, per_attack_list = zip(*results)

# Save DataFrames
print(f"Saving overall metrics to {OVERALL_CSV}")
pd.DataFrame(overall_list).to_csv(OVERALL_CSV, index=False)
print(f"Saving per-attack metrics to {PER_ATTACK_CSV}")
pd.DataFrame([row for sub in per_attack_list for row in sub]).to_csv(PER_ATTACK_CSV, index=False)

print("All done!")
