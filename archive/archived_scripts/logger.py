# logger.py

import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from config import metrics_log_path, plots_dir

Path(plots_dir).mkdir(parents=True, exist_ok=True)

def log_metrics(model_name, metrics):
    try:
        with open(metrics_log_path, "r") as f:
            log = json.load(f)
    except FileNotFoundError:
        log = {}

    log[model_name] = metrics

    with open(metrics_log_path, "w") as f:
        json.dump(log, f, indent=4)

def plot_confusion(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{plots_dir}/{model_name}_confusion.png")
    plt.close()
