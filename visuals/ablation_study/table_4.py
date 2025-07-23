import pandas as pd

# Load the ablation results CSV
df = pd.read_csv("ablation_overall_metrics.csv")

# === Extract fields from setup string ===
df['protocol'] = df['setup'].apply(lambda x: 'same_day' if 'same_day' in x else 'cross_day')
df['removed_model'] = df['setup'].apply(lambda x: x.split('_ablate_')[-1])

# === Group and average across days ===
table4 = df.groupby(['protocol', 'removed_model'])[
    ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
].mean().reset_index()

# Optional: round for readability
table4 = table4.round(4)

# Save the table to CSV
table4.to_csv("table4_ablation_summary.csv", index=False)
