# pipeline/sampling.py

import pandas as pd
from imblearn.over_sampling import SMOTE

def apply_smote(X, y, random_state=42):
    """
    Apply SMOTE oversampling to feature matrix X and label vector y.
    Returns the resampled (X_res, y_res) arrays.
    Prints the new sample size.
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"[SAMPLE] SMOTE applied: {len(X_res)} samples")
    return X_res, y_res

def downsample_df(df, max_samples, random_state=42):
    """
    Randomly downsample a DataFrame to max_samples rows if it exceeds that size.
    Maintains global class proportion by sampling without replacement.
    Prints the new DataFrame size.
    """
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
        print(f"[SAMPLE] Downsampled to {len(df)} rows")
    return df
