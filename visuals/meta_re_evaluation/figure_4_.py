import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from collections import defaultdict

# === CONFIG ===
curve_dir = "/home/devesh-senthilraja/OrionIDS/results/meta_re_evaluation/roc_pr_curves/"  # Change this to your PR CSV directory
common_recalls = np.linspace(0, 1, 200)  # Common recall points for interpolation
save_dir = "./"  # Or change to desired output folder

# === PARSING LOGIC ===
def parse_filename(fname):
    parts = fname.split("_")
    protocol = "same_day" if "same" in parts else "cross_day"
    if "voting" in fname:
        model = "Soft Voting" if "soft" in fname else "Hard Voting"
    else:
        model = fname.split("meta_")[-1].replace("_pr.csv", "")
    return model, protocol

# === DATA COLLECTION ===
curves = defaultdict(list)
file_paths = glob(os.path.join(curve_dir, "*_pr.csv"))

for path in file_paths:
    filename = os.path.basename(path)
    model, protocol = parse_filename(filename)
    df = pd.read_csv(path)
    if not df['recall'].is_monotonic_increasing:
        df = df.sort_values('recall')  # Ensure recall is sorted
    try:
        interp_precision = np.interp(common_recalls, df['recall'], df['precision'], left=1.0, right=0.0)
        curves[(model, protocol)].append(interp_precision)
    except Exception as e:
        print(f"Skipping {filename}: {e}")

# === AVERAGING ===
avg_curves = {}
for (model, protocol), precision_list in curves.items():
    if len(precision_list) > 0:
        avg_precision = np.mean(precision_list, axis=0)
        avg_curves[(model, protocol)] = avg_precision

# === PLOTTING ===
for protocol in ['same_day', 'cross_day']:
    plt.figure(figsize=(10, 6))
    for (model, proto), precision_vals in avg_curves.items():
        if proto == protocol:
            plt.plot(common_recalls, precision_vals, label=model)
    plt.title(f"Averaged PR Curves ({protocol.replace('_', ' ').title()})", fontsize=14)
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"/home/devesh-senthilraja/OrionIDS/results/meta_re_evaluation/figure4_avg_pr_{protocol}.png"), dpi=300)
