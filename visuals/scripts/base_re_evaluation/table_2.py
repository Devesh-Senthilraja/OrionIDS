import pandas as pd

# Load the CSV file (adjust the path if needed)
df = pd.read_csv("results/base_re_evaluation/per_attack_metrics.csv")

# Remove benign rows — we're only analyzing attack-specific performance
df = df[df['attack_type'] != 'Benign']

# Group by model, feature group, protocol, and attack type
# Average across all days for each grouping
table2 = df.groupby(
    ['model', 'feature_group', 'protocol', 'attack_type']
)[['accuracy', 'precision', 'recall', 'f1']].mean().reset_index()

# Round for presentation clarity
table2 = table2.round(4)

# Save to a new CSV file
table2.to_csv("visuals/table2_full_per_attack_metrics.csv", index=False)
