import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === CONFIGURE PATHS ===
BASE_PRED_DIR = "results/meta_re_evaluation/base_predictions"
RAW_PRED_DIR  = "results/meta_re_evaluation/meta_predictions"
# (adjust these to your actual folders)

# Models to include
BASE_MODELS = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]
META_MODEL  = "MLP"

# Protocols and days
protocols = ["same_day", "cross_day"]
# infer days from one of the directories
files = os.listdir(BASE_PRED_DIR)
days = sorted({fname.split("_")[0] for fname in files if fname.endswith("_test_base_preds.csv")})

# Helper: accumulate confusion matrices per model/protocol
def accumulate_cm(model_name, protocol, is_meta=False):
    cm_sum = np.zeros((2,2), dtype=int)
    for day in days:
        if not is_meta:
            fn = f"{day}_{protocol}_test_base_preds.csv"
            df = pd.read_csv(os.path.join(BASE_PRED_DIR, fn))
            y_true = df["y_true"]
            y_pred = df[f"label_{model_name}"]
        else:
            fn = f"{day}_{protocol}_meta_{model_name}_raw.csv"
            df = pd.read_csv(os.path.join(RAW_PRED_DIR, fn))
            y_true = df["y_true"]
            y_pred = df["y_pred"]
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        cm_sum += cm
    return cm_sum

# Prepare grid
all_models = BASE_MODELS + [META_MODEL]
n_cols = len(all_models)
n_rows = len(protocols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), 
                         constrained_layout=True)

for i, protocol in enumerate(protocols):
    for j, model_name in enumerate(all_models):
        ax = axes[i][j] if n_rows>1 else axes[j]
        is_meta = (model_name == META_MODEL)
        cm = accumulate_cm(model_name, protocol, is_meta=is_meta)
        # normalize to proportions
        cm_norm = cm.astype(float) / cm.sum()
        sns.heatmap(
            cm_norm, annot=cm, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"],
            ax=ax
        )
        title = f"{model_name}\n({protocol.replace('_',' ').title()})"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

# Global labels
fig.suptitle("Accumulated Confusion Matrices (counts) — Base Models & Meta-Model", 
             fontsize=16, y=1.02)
for ax in axes.flatten():
    ax.set_ylabel(ax.get_yticklabels()[0].get_text(), rotation=0, labelpad=30)
    ax.set_xlabel(ax.get_xticklabels()[-1].get_text(), rotation=90, labelpad=10)

# Save
outfn = "visuals/figure8_confusion_matrices.png"
fig.savefig(outfn, dpi=300, bbox_inches='tight')
print(f"Saved {outfn}")

