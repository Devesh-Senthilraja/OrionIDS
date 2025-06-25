import pandas as pd
import numpy as np
from pathlib import Path
from pipeline.data_loader import Experiment
from pipeline.sampling import apply_smote, downsample_df
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

MODELS = {
    "LogisticRegression": (
        lambda: LogisticRegression(solver="saga", max_iter=1000, random_state=42, n_jobs=1),
        {"C": [0.01, 0.1, 1]}
    ),
    "DecisionTree": (
        lambda: DecisionTreeClassifier(random_state=42),
        {"max_depth": [5, 10, 20]}
    ),
    "RandomForest": (
        lambda: RandomForestClassifier(random_state=42, n_jobs=1),
        {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
    ),
    "ExtraTrees": (
        lambda: ExtraTreesClassifier(random_state=42, n_jobs=1),
        {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
    ),
    "XGBoost": (
        lambda: XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=1),
        {"learning_rate": [0.1, 0.01], "n_estimators": [100, 200], "max_depth": [3, 5]}
    ),
    "LightGBM": (
        lambda: LGBMClassifier(random_state=42, n_jobs=1),
        {"learning_rate": [0.1, 0.01], "n_estimators": [100, 200], "num_leaves": [31, 50]}
    ),
    "CatBoost": (
        lambda: CatBoostClassifier(verbose=0, random_state=42),
        {"depth": [4, 6], "iterations": [100, 200], "learning_rate": [0.1, 0.01]}
    ),
    "KNN": (
        lambda: KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7]}
    ),
    "LinearSVC": (
        lambda: LinearSVC(max_iter=10000, random_state=42),
        {"C": [0.1, 1, 10]}
    ),
    "MLP": (
        lambda: MLPClassifier(max_iter=500, random_state=42),
        {"hidden_layer_sizes": [(50,), (100,)], "alpha": [1e-4, 1e-3]}
    )
}

FEATURE_CATEGORIES = {
    "flow_metrics":        ["Flow Duration","Flow Byts/s","Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min"],
    "packet_size_stats":   ["Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","Pkt Size Avg","Fwd Seg Size Avg","Bwd Seg Size Avg"],
    "timing_iat":          ["Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min"],
    "flags_and_protocol":  ["FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt","URG Flag Cnt","CWE Flag Count","ECE Flag Cnt","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Protocol","Dst Port","Init Fwd Win Byts","Init Bwd Win Byts"],
    "rates_and_ratios":    ["Down/Up Ratio","Fwd Pkts/s","Bwd Pkts/s","Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg","Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg"],
    "connection_activity": ["Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts","Fwd Act Data Pkts","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"],
    "meta_features":       ["Flow Duration","SYN Flag Cnt","Fwd IAT Mean"]
}

NEEDED_COLS = set().union(*FEATURE_CATEGORIES.values()) | {"Attack Type", "Label"}

def load_and_clean(fp: Path) -> pd.DataFrame:
    """
    Read only NEEDED_COLS, drop infinities & NaNs, downcast numerics.
    """
    df = pd.read_csv(fp, usecols=list(NEEDED_COLS), low_memory=False)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # downcast ints/floats to reduce mem
    for c in df.select_dtypes(include=["float64","int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df

def run_training(exp: Experiment, verbose: bool = False, n_jobs: int = 1):
    """
    For one Experiment:
      1) load & cap each train file
      2) apply SMOTE or class_weight
      3) train each base model ↔ feature‐set combo
      4) optionally stack & train meta‐model
    Stores:
      exp.trained_base: dict[name → (model, feature_list)]
      exp.trained_meta: (model_name, model) or None
    """
    # — 1) LOAD & CAP TRAINING DATA —
    dfs = []
    for fp in exp.train_files:
        if verbose: print(f"[TRAIN] Loading {fp.name}")
        df = load_and_clean(fp)
        if len(df) > exp.cap_train_per_file:
            df = df.sample(n=exp.cap_train_per_file, random_state=42)
            if verbose: print(f"  ↘ downsampled to {len(df)} rows")
        dfs.append(df)
    train_df = pd.concat(dfs, ignore_index=True)

    # — 2) HANDLE CLASS IMBALANCE —
    class_weight = None
    if exp.config["smote"]:
        if verbose: print("[TRAIN] Applying SMOTE")
        Xs, ys = apply_smote(
            train_df.drop(columns=["Attack Type","Label"]),
            train_df["Label"]
        )
        train_df = pd.DataFrame(Xs, columns=train_df.columns.drop(["Attack Type","Label"]))
        train_df["Label"] = ys
    elif exp.config["class_weighting"]:
        class_weight = "balanced"
        if verbose: print("[TRAIN] Using class_weight='balanced'")

    # — 3) TRAIN BASE MODELS —
    exp.trained_base = {}
    for model_name, feat_sets in exp.config["base_models"].items():
        # decide grouping vs separate
        if exp.config["feature_grouping"]:
            combos = [(model_name + "_all",
                       sum((FEATURE_CATEGORIES[f] for f in feat_sets), []))]
        else:
            combos = [(f"{model_name}_{fset}", FEATURE_CATEGORIES[fset])
                      for fset in feat_sets]

        for combo_name, features in combos:
            if verbose: print(f"[TRAIN] Base '{combo_name}' on {len(features)} features")
            model_fn, grid = MODELS[model_name]
            model = model_fn()
            if class_weight:
                try:
                    model.set_params(class_weight=class_weight)
                except:
                    pass
            if exp.config["grid_search_base"]:
                if verbose: print("  ↗ GridSearchCV (base)")
                model = GridSearchCV(
                    model, grid,
                    cv=3, n_jobs=n_jobs,
                    scoring="f1"
                )
            X = train_df[features]
            y = train_df["Label"]
            model.fit(X, y)
            exp.trained_base[combo_name] = (model, features)

    # — 4) TRAIN META MODEL (if enabled) —
    exp.trained_meta = {}
    if exp.config["use_meta"]:
        if verbose: print("[TRAIN] Building meta‐feats & training meta‐models")

        # 1) build DataFrame of all base‐model probability columns
        base_preds_df = pd.concat([
            pd.Series(mdl.predict_proba(train_df[feats])[:,1], name=name)
            for name,(mdl,feats) in exp.trained_base.items()
        ], axis=1)

        # 2) loop each meta_model → list of raw feature‐categories
        for meta_name, raw_cats in exp.config["meta_models"].items():

            # 2a) decide grouping vs separate combos
            if exp.config["feature_grouping"]:
                combos = [(f"{meta_name}_all", raw_cats)]
            else:
                combos = [
                    (f"{meta_name}_{cat}", [cat])
                    for cat in raw_cats
                ]

            # 2b) train each combo
            for combo_name, cats in combos:
                if verbose:
                    print(f"[TRAIN] Meta '{combo_name}' (from {cats})")

                df_meta = base_preds_df.copy()
                if exp.config["include_meta_features"]:
                    for cat in cats:
                        cols = FEATURE_CATEGORIES[cat]
                        df_meta[cols] = train_df[cols].values

                df_meta["Label"] = train_df["Label"].values
                Xm = df_meta.drop(columns=["Label"])
                ym = df_meta["Label"]

                fn, grid = MODELS[meta_name]
                mdl = fn()
                if exp.config["class_weighting"]:
                    try: mdl.set_params(class_weight="balanced")
                    except: pass
                if exp.config["grid_search_meta"]:
                    if verbose: print("  ↗ GridSearchCV (meta)")
                    mdl = GridSearchCV(mdl, grid, cv=3, n_jobs=n_jobs, scoring="f1")

                mdl.fit(Xm, ym)
                exp.trained_meta[combo_name] = mdl

        if verbose:
            print(f"[TRAIN] Done meta‐models: {list(exp.trained_meta.keys())}\n")

    if verbose:
        print(f"[TRAIN] Finished '{exp.name}'\n")