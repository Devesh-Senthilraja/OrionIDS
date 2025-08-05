import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# === UPDATE THESE PATHS ===
PRED_DIR = "results/meta_re_evaluation/meta_predictions"

# Only plot MLP and both voting strategies
MODELS = ["MLP", "hard", "soft"]
PROTOCOLS = ["same_day", "cross_day"]

# Figure for each protocol
for protocol in PROTOCOLS:
    plt.figure(figsize=(10, 6))

    for model in MODELS:
        y_trues = []
        y_scores = []

        # Detect filename pattern
        pattern_type = "meta" if model == "MLP" else "voting"

        # Discover days
        days = sorted({
            fname.split("_")[0]
            for fname in os.listdir(PRED_DIR)
            if fname.endswith(f"{protocol}_{pattern_type}_{model}_raw.csv")
        })

        for day in days:
            fn = f"{day}_{protocol}_{pattern_type}_{model}_raw.csv"
            path = os.path.join(PRED_DIR, fn)
            if not os.path.isfile(path):
                print(f"Missing file: {fn}")
                continue

            df = pd.read_csv(path)
            y_true = df["y_true"]
            y_score = df["y_score"]
            y_trues.append(y_true)
            y_scores.append(y_score)

        if not y_trues or not y_scores:
            print(f"Skipping {model} ({protocol}) — no valid data found.")
            continue

        y_true_all = pd.concat(y_trues, ignore_index=True)
        y_score_all = pd.concat(y_scores, ignore_index=True)

        precision, recall, _ = precision_recall_curve(y_true_all, y_score_all)
        ap = average_precision_score(y_true_all, y_score_all)

        plt.plot(recall, precision, label=f"{model} (AP={ap:.3f})")

    plt.title(f"Precision–Recall Curves ({protocol.replace('_',' ').title()})", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    outfn = f"visuals/figure4_pr_curves_{protocol}.png"
    plt.savefig(outfn, dpi=300)
    print(f"Saved {outfn}")
