import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the aggregated Table 1 file
df = pd.read_csv("visuals/table1_base_model_overall_metrics.csv")

# Filter to only cross-day results
df = df[df['protocol'] == 'same_day']

# Set plot style
sns.set(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(12, 6))
bar = sns.barplot(
    data=df,
    x='model',
    y='f1',
    hue='feature_group',
    palette='Set2'
)

# Improve formatting
bar.set_title("Same-Day F1 Score per Model across Feature Groups", fontsize=14)
bar.set_xlabel("Model", fontsize=12)
bar.set_ylabel("F1 Score", fontsize=12)
bar.legend(title="Feature Group", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save and/or show the figure
plt.savefig("visuals/figure1.2_base_model_f1_bar_chart.png", dpi=300)
