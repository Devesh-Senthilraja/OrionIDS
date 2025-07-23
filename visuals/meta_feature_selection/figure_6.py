import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the summary table
df = pd.read_csv("table5_feature_addition_summary.csv")

# 2. Pivot to wide format: one row per feature, columns for each protocol’s F1
df_wide = df.pivot(index="feature", columns="protocol", values="f1")

# 3. Sort features by descending cross-day F1
df_wide = df_wide.sort_values(by="cross_day", ascending=False)

# 4. Create a rank index (1…78)
ranks = range(1, len(df_wide) + 1)

# 5. Plot
plt.figure(figsize=(12, 6))
plt.plot(ranks, df_wide["same_day"], label="Same-Day")
plt.plot(ranks, df_wide["cross_day"], label="Cross-Day")

# 6. Formatting
plt.title("F1 Score vs. Feature Rank (sorted by cross-day F1)", fontsize=14)
plt.xlabel("Feature Rank (by Cross-Day F1)", fontsize=12)
plt.ylabel("F1 Score", fontsize=12)
plt.legend(title="Protocol")
plt.xticks(ranks, df_wide.index, rotation=90)  # optional: show feature names if desired
plt.tight_layout()

# 7. Save and show
plt.savefig("figure6_feature_f1_vs_rank.png", dpi=300)
