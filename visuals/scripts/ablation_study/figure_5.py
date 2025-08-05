import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load averaged table
df = pd.read_csv("visuals/table4_ablation_summary.csv")

# Pivot to get the full-model F1 per protocol
baseline_f1 = df[df['removed_model'] == 'None'].set_index('protocol')['f1'].to_dict()

# Compute F1 drop
df = df[df['removed_model'] != 'None'].copy()
df['f1_drop'] = df.apply(lambda row: baseline_f1[row['protocol']] - row['f1'], axis=1)

# Sort bars within each protocol by F1 drop
df = df.sort_values(by=['protocol', 'f1_drop'], ascending=[True, False])

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df,
    x='removed_model',
    y='f1_drop',
    hue='protocol',
    palette='Set2'
)

# Style
ax.set_title("F1 Score Drop After Removing Each Base Model", fontsize=14)
ax.set_xlabel("Removed Base Model", fontsize=12)
ax.set_ylabel("Delta F1 Score (Drop from Full Ensemble)", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title="Protocol", loc='upper right')
plt.tight_layout()

# Save and show
plt.savefig("visuals/figure5_ablation_f1_drop.png", dpi=300)
