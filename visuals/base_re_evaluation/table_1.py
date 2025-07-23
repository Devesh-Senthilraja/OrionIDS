import pandas as pd

# Load the CSV
df = pd.read_csv("/home/devesh-senthilraja/OrionIDS/results/base_re_evaluation/overall_metrics.csv")

# Group by model, feature group, and protocol — then average the metrics
table1 = df.groupby(['model', 'feature_group', 'protocol'])[
    ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
].mean().reset_index()

# Round values for presentation
table1 = table1.round(4)

# Sort for readability
table1 = table1.sort_values(by=['protocol', 'model', 'feature_group'])

# Save to file
table1.to_csv("/home/devesh-senthilraja/OrionIDS/results/base_re_evaluation/table1_base_model_overall_metrics.csv", index=False)
