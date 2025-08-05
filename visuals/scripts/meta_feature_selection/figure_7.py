import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your Table 5 summary (including the 'None' baseline row)
df = pd.read_csv("visuals/table5_feature_addition_summary.csv")

# 2. Extract the baseline F1 (feature == 'None') for each protocol
baseline = (
    df[df['feature'] == 'None']
    .set_index('protocol')['f1']
    .rename('baseline_f1')
)

# 3. Filter out the actual features, merge in baseline to compute improvement
df_feats = df[df['feature'] != 'None'].copy()
df_feats['baseline_f1'] = df_feats['protocol'].map(baseline)
df_feats['f1_improvement'] = df_feats['f1'] - df_feats['baseline_f1']

# 4. Plot a boxplot of ΔF1 by protocol
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_feats,
    x='protocol',
    y='f1_improvement',
    palette='Set2'
)

# 5. Formatting
plt.title("Distribution of F1 Score Improvements", fontsize=14)
plt.xlabel("Protocol", fontsize=12)
plt.ylabel("Delta F1 Score (Feature vs. Baseline)", fontsize=12)
plt.tight_layout()

# 6. Save & show
plt.savefig("visuals/figure7_feature_addition_improvement_boxplot.png", dpi=300)

