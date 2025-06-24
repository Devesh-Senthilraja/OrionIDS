# run_full_stack_pipeline.py

# End-to-end stacking pipeline:
# 1) Trains combinations of base models on CIC-IDS-2018 cleaned CSVs
# 2) Generates meta-features (base-model probabilities) + optional raw features
# 3) Trains & evaluates meta-models under within-day and leave-one-day-out protocols
# 4) Toggles SMOTE, class weighting, and inclusion of meta-features
# 5) Saves results to CSV for analysis

import pandas as pd
import numpy as np
import time
import gc
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- EXPERIMENT CONFIGURATION ---
SAMPLING_MODES = ["none", "smote", "class_weight"]
INCLUDE_META   = [False, True]

# --- BASE LEARNERS & FEATURE SETS ---
BASE_MODELS = {
    "ET_flow": (
        lambda: __import__("sklearn.ensemble").ensemble.ExtraTreesClassifier(
            n_estimators=200, n_jobs=1, random_state=42
        ),
        ["Flow Duration","Flow Byts/s","Flow Pkts/s",
         "Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min"]
    ),
    "RF_flags": (
        lambda: __import__("sklearn.ensemble").ensemble.RandomForestClassifier(
            n_estimators=200, n_jobs=1, random_state=42
        ),
        ["FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt",
         "ACK Flag Cnt","URG Flag Cnt","CWE Flag Count","ECE Flag Cnt",
         "Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags"]
    ),
    "ET_timing": (
        lambda: __import__("sklearn.ensemble").ensemble.ExtraTreesClassifier(
            n_estimators=200, n_jobs=1, random_state=42
        ),
        ["Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
         "Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min"]
    ),
    "LGBM_packet": (
        lambda: __import__("lightgbm").LGBMClassifier(
            n_estimators=200, learning_rate=0.01, num_leaves=50,
            n_jobs=1, random_state=42
        ),
        ["Tot Fwd Pkts","TotLen Fwd Pkts","Fwd Pkt Len Mean",
         "Pkt Len Max","Pkt Len Min"]
    )
}

# --- RAW META-FEATURES ---
META_FEATURES = [
    "Flow Duration",   # session-length anomalies
    "SYN Flag Cnt",    # brute-force bursts
    "Fwd IAT Mean"     # slow-DoS timing irregularities
]

# --- META-MODELS TO TEST ---
META_MODELS = {
    "LR": lambda: __import__("sklearn.linear_model").linear_model.LogisticRegression(
        max_iter=1000, n_jobs=1, random_state=42
    ),
    "RF_meta": lambda: __import__("sklearn.ensemble").ensemble.RandomForestClassifier(
        n_estimators=200, n_jobs=1, random_state=42
    )
}

# --- DATA & OUTPUT DIRS ---
CLEAN_DIR = Path("cleaned")
ALL_FILES = sorted(CLEAN_DIR.glob("*.csv"))
OUT_DIR = Path("results/meta_stack")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WITHIN_PATH = OUT_DIR / "within_day_results.csv"
CROSS_PATH  = OUT_DIR / "cross_day_results.csv"

# --- MEMORY & DOWNSAMPLING ---
# maximum rows to keep in training to avoid OOM
MAX_WITHIN_SAMPLES = 200_000
MAX_CROSS_SAMPLES  = 800_000

def downsample_df(df, max_samples):
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"↘ Downsampled to {len(df)} rows.")
    return df

# --- ONLY READ THE COLUMNS WE NEED ---
NEEDED_COLS = {
    col for mdl in BASE_MODELS.values() for col in mdl[1]
} | set(META_FEATURES) | {"Attack Type", "Label"}

def load_and_clean(fp):
    df = pd.read_csv(fp, usecols=list(NEEDED_COLS), low_memory=False)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # downcast numeric columns
    for c in df.select_dtypes(include=["float64","int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    print(f"Loaded {fp.name}: {len(df)} rows (cols: {df.shape[1]}).")
    return df

def apply_sampling(df, mode):
    y = df["Label"]
    if mode == "smote":
        X, yr = SMOTE(random_state=42).fit_resample(
            df.drop(columns=["Attack Type","Label"]), y
        )
        df2 = pd.DataFrame(X, columns=df.columns.drop(["Attack Type","Label"]))
        df2["Label"] = yr
        df2["Attack Type"] = yr.map({0:"Benign",1:"Attack"})
        print(f"SMOTE applied: {len(df2)} rows.")
        return df2, None

    if mode == "class_weight":
        pos, neg = (y==1).sum(), (y==0).sum()
        total = pos + neg
        cw = {0: total/(2*neg), 1: total/(2*pos)}
        print(f"Class weights: {cw}")
        return df, cw

    print("No sampling.")
    return df, None

def get_base_predictions(df, keys, class_weight):
    y = df["Label"]
    preds = pd.DataFrame(index=df.index)
    for key in keys:
        clf = BASE_MODELS[key][0]()
        feats = BASE_MODELS[key][1]
        if class_weight:
            clf.set_params(class_weight=class_weight)
        clf.fit(df[feats], y)
        preds[f"pred_{key}"] = clf.predict_proba(df[feats])[:,1]
    return preds, y

def evaluate_meta(Xtr, ytr, Xte, yte, include_meta, combo_keys, raw_tr=None, raw_te=None):
    # attach raw features if requested
    if include_meta and raw_tr is not None and raw_te is not None:
        for m in META_FEATURES:
            Xtr[m] = raw_tr.loc[ytr.index, m].values
            Xte[m] = raw_te.loc[yte.index, m].values
        print("✔ Added raw meta-features.")

    results = []
    for mname, build in META_MODELS.items():
        clf = build()
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        results.append({
            "meta_model": mname,
            "base_combo": "+".join(combo_keys),
            "include_meta": include_meta,
            "accuracy": accuracy_score(yte, ypred),
            "f1":        f1_score(yte, ypred),
            "precision": precision_score(yte, ypred),
            "recall":    recall_score(yte, ypred)
        })
    return results

def run_within_day(fp, keys, sampling, include_meta):
    start = time.time()
    print(f"\n-- WITHIN-DAY {fp.name} | sampling={sampling} | include_meta={include_meta} --")

    df = load_and_clean(fp)
    tr, te = train_test_split(
        df, test_size=0.3, stratify=df["Label"], random_state=42
    )

    tr = downsample_df(tr, MAX_WITHIN_SAMPLES)
    tr, cw = apply_sampling(tr, sampling)

    base_tr, ytr = get_base_predictions(tr, keys, cw)
    base_te, yte = get_base_predictions(te, keys, None)

    res = evaluate_meta(base_tr, ytr, base_te, yte, include_meta, keys,
                        raw_tr=tr, raw_te=te)
    for r in res:
        r.update({"file": fp.name, "sampling": sampling})

    print(f"→ Done in {time.time()-start:.1f}s.")

    # free up memory
    del df, tr, te, base_tr, base_te
    gc.collect()
    return res

def run_cross_day(fp_leave, keys, sampling, include_meta):
    start = time.time()
    print(f"\n-- CROSS-DAY leave {fp_leave.name} | sampling={sampling} | include_meta={include_meta} --")

    # load & concat all other days
    dfs = [load_and_clean(f) for f in ALL_FILES if f != fp_leave]
    tr = pd.concat(dfs, ignore_index=True)
    te = load_and_clean(fp_leave)

    tr = downsample_df(tr, MAX_CROSS_SAMPLES)
    tr, cw = apply_sampling(tr, sampling)

    base_tr, ytr = get_base_predictions(tr, keys, cw)
    base_te, yte = get_base_predictions(te, keys, None)

    res = evaluate_meta(base_tr, ytr, base_te, yte, include_meta, keys,
                        raw_tr=tr, raw_te=te)
    for r in res:
        r.update({"leave_out": fp_leave.name, "sampling": sampling})

    print(f"→ Done in {time.time()-start:.1f}s.")

    del dfs, tr, te, base_tr, base_te
    gc.collect()
    return res

if __name__ == "__main__":
    keys = list(BASE_MODELS.keys())
    print(f"Using base models: {keys}")

    tasks = [
        (f, keys, s, im)
        for f in ALL_FILES
        for s in SAMPLING_MODES
        for im in INCLUDE_META
    ]

    # within-day
    print(f"{len(tasks)} within-day jobs...")
    within = Parallel(n_jobs=2, backend="threading")(
        delayed(run_within_day)(fp, keys, s, im)
        for fp, keys, s, im in tasks
    )
    pd.DataFrame([r for sub in within for r in sub]).to_csv(WITHIN_PATH, index=False)
    print("Within-day results →", WITHIN_PATH)

    # cross-day
    print(f"{len(tasks)} cross-day jobs...")
    cross = Parallel(n_jobs=1)(
        delayed(run_cross_day)(fp, keys, s, im)
        for fp, keys, s, im in tasks
    )
    pd.DataFrame([r for sub in cross for r in sub]).to_csv(CROSS_PATH, index=False)
    print("Cross-day results →", CROSS_PATH)