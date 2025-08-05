import pandas as pd

# Load raw data
voting_df = pd.read_csv("results/meta_re_evaluation/voting_overall_metrics.csv")
meta_df = pd.read_csv("results/meta_re_evaluation/meta_overall_metrics.csv")

# Parse protocol and model info
voting_grouped = voting_df['setup'].str.extract(r'(\d{2}-\d{2}-\d{4})_([a-z_]+)_voting_(hard|soft)')
voting_df['date'] = voting_grouped[0]
voting_df['protocol'] = voting_grouped[1].str.replace('_', '')
voting_df['model_name'] = voting_grouped[2].map({'hard': 'Hard Voting', 'soft': 'Soft Voting'})
voting_df['model_type'] = 'Voting Ensemble'

meta_grouped = meta_df['setup'].str.extract(r'(\d{2}-\d{2}-\d{4})_([a-z_]+)_meta_(.*)')
meta_df['date'] = meta_grouped[0]
meta_df['protocol'] = meta_grouped[1].str.replace('_', '')
meta_df['model_name'] = meta_grouped[2]
meta_df['model_type'] = 'Stacking'

# Combine both and group by model/protocol for averaging
combined_df = pd.concat([
    voting_df[['model_type', 'model_name', 'protocol', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']],
    meta_df[['model_type', 'model_name', 'protocol', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']]
])

# Average across days
table3 = combined_df.groupby(['model_type', 'model_name', 'protocol']).mean(numeric_only=True).reset_index()
table3 = table3.round(4)

# Save output
table3.to_csv("visuals/table3_meta_voting_overall_metrics.csv", index=False)
