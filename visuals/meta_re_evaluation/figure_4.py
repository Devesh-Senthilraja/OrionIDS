import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# === UPDATE THESE PATHS ===
BASE_PRED_DIR = "/home/devesh-senthilraja/OrionIDS/results/meta_re_evaluation/base_predictions"
RAW_PRED_DIR  = "/home/devesh-senthilraja/OrionIDS/results/meta_re_evaluation/meta_predictions"

# Models: your 4 base learners + chosen meta
BASE_MODELS = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]
META_MODEL  = "MLP"
MODELS = BASE_MODELS + [META_MODEL]

# Protocols
PROTOCOLS = ["same_day", "cross_day"]

# Figure for each protocol
for protocol in PROTOCOLS:
    plt.figure(figsize=(10, 6))
    for model in MODELS:
        y_trues = []
        y_scores = []
        # Discover days from base preds folder
        days = sorted({fname.split("_")[0]
                       for fname in os.listdir(BASE_PRED_DIR)
                       if fname.endswith(f"_{protocol}_test_base_preds.csv")})
        for day in days:
            if model in BASE_MODELS:
                fn = f"{day}_{protocol}_test_base_preds.csv"
                df = pd.read_csv(os.path.join(BASE_PRED_DIR, fn))
                y_true = df["y_true"]
                y_score = df[f"pred_{model}"]
            else:
                fn = f"{day}_{protocol}_meta_{model}_raw.csv"
                df = pd.read_csv(os.path.join(RAW_PRED_DIR, fn))
                y_true = df["y_true"]
                y_score = df["y_score"]
            y_trues.append(y_true)
            y_scores.append(y_score)

        # Concatenate all days
        y_true_all = pd.concat(y_trues, ignore_index=True)
        y_score_all = pd.concat(y_scores, ignore_index=True)

        # Compute curve & AP
        precision, recall, _ = precision_recall_curve(y_true_all, y_score_all)
        ap = average_precision_score(y_true_all, y_score_all)

        # Plot
        plt.plot(recall, precision, label=f"{model} (AP={ap:.3f})")

    plt.title(f"Precision–Recall Curves ({protocol.replace('_',' ').title()})", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    outfn = f"/home/devesh-senthilraja/OrionIDS/results/meta_re_evaluation/figure4_pr_curves_{protocol}.png"
    plt.savefig(outfn, dpi=300)
    print(f"Saved {outfn}")
