import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the corrected data
df = pd.read_csv("visuals/table3_meta_voting_overall_metrics.csv")

# Optional: sort model names by F1 score on sameday
sorted_models = (
    df[df['protocol'] == 'sameday']
    .sort_values(by='f1', ascending=False)['model_name']
    .tolist()
)
df['model_name'] = pd.Categorical(df['model_name'], categories=sorted_models, ordered=True)

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
bar = sns.barplot(
    data=df,
    x='model_name',
    y='f1',
    hue='protocol',
    palette='Set2'
)

# Labels and styling
bar.set_title("F1 Scores of Stacking and Voting Ensembles", fontsize=14)
bar.set_xlabel("Model", fontsize=12)
bar.set_ylabel("F1 Score", fontsize=12)
bar.legend(title="Protocol", loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save and show
plt.savefig("visuals/figure3_meta_vs_voting_f1_scores.png", dpi=300)
