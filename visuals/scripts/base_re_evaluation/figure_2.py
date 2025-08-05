import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load base model table
df = pd.read_csv("visuals/table1_base_model_overall_metrics.csv")

# Pivot to get both same_day and cross_day F1 scores
pivot = df.pivot_table(
    index=['model', 'feature_group'],
    columns='protocol',
    values='f1'
).reset_index()

# Compute degradation (same_day - cross_day)
pivot['f1_drop'] = pivot['same_day'] - pivot['cross_day']

# Create a pivot table for heatmap
heatmap_data = pivot.pivot(index='model', columns='feature_group', values='f1_drop')

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.set(font_scale=0.9)
ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={'label': 'F1 Score Drop'}
)

# Add titles and labels
plt.title("F1 Score Degradation from Same-Day to Cross-Day", fontsize=14)
plt.xlabel("Feature Group", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.tight_layout()

# Save and/or display
plt.savefig("visuals/figure2_f1_degradation_heatmap.png", dpi=300)
