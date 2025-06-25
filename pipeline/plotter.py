# pipeline/plotter.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

def plot_confusion_heatmaps(exp_name, verbose=False):
    """
    Loads metrics.json under results/exp_name and plots heatmaps
    of (averaged) confusion-matrix components for base vs meta.
    Requires evaluator also dump confusion counts if desired.
    """
    metrics = json.load(open(Path("results")/exp_name/"metrics.json"))
    df = pd.DataFrame(metrics)
    # pivot by combo and model_type
    for mtype in ["base","meta"]:
        sub = df[df["model_type"]==mtype]
        if sub.empty: continue
        pivot = sub.pivot_table("f1", index="data_file", columns="combo")
        plt.figure(figsize=(10,8))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"{exp_name} — {mtype.capitalize()} F1 Across Files")
        out = Path("results")/exp_name/f"{mtype}_f1_heatmap.png"
        plt.tight_layout(); plt.savefig(out); plt.close()
        if verbose: print(f"[PLOT] Saved {out}")

def plot_shap_summary(exp_name, exp, verbose=False):
    """
    Re-runs SHAP on the final meta‐model’s hold‐out data.
    (Or pulls saved SHAP values if you serialized them.)
    """
    # naïve re-import
    from pipeline.data_loader import load_and_clean
    meta_models = exp.trained_meta
    if not meta_models:
        if verbose: print("[PLOT] No meta models to SHAP")
        return

    # pick first test_file
    df = load_and_clean(exp.test_files[0])
    base_preds = {
        nm: mdl.predict_proba(df[feat])[:,1]
        for nm,(mdl,feat) in exp.trained_base.items()
    }
    X = pd.DataFrame(base_preds)
    if exp.config["include_meta_features"]:
        for cat in exp.config["meta_models"][list(exp.config["meta_models"].keys())[0]]:
            X[cat] = df[cat].values

    # shap for each meta combo
    for name,mdl in meta_models.items():
        explainer = shap.TreeExplainer(mdl)
        shap_vals = explainer.shap_values(X)
        plt.figure()
        shap.summary_plot(shap_vals[1], X, show=False)
        plt.title(f"{exp_name} SHAP — {name}")
        out = Path("results")/exp_name/f"shap_{name}.png"
        plt.tight_layout(); plt.savefig(out); plt.close()
        if verbose: print(f"[PLOT] Saved {out}")
