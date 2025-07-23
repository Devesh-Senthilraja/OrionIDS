import pandas as pd
import ast

# 1. Load your CSV
df = pd.read_csv("meta_feature_selection_metrics.csv")

# 2. Derive protocol
df['protocol'] = df['setup'].apply(lambda s: 'same_day' if 'same_day' in s else 'cross_day')

# 3. Parse out the added feature (drop rows where raw_features is empty)
def extract_feature(raw):
    try:
        lst = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not lst:
        return 'None'       # baseline row
    return lst[0]          # the actual feature name

df['feature'] = df['raw_features'].apply(extract_feature)
df = df[df['feature'].notna()]  # keep only the 77 real features

# 4. Group & average across the 7 days
table5 = (
    df
    .groupby(['feature', 'protocol'])[
        ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    ]
    .mean()
    .reset_index()
)

# 5. Round and save
table5 = table5.round(4)
table5.to_csv("table5_feature_addition_summary.csv", index=False)

