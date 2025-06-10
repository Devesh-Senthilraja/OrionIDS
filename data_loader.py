# data_loader.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import (
    # train_dataset_path,
    train_dataset_paths,
    test_dataset_path,
    general_features,
    statistical_features,
    behavioral_features,
    enabled_feature_sets,
    test_size,
    random_seed,
    use_class_weights,
    use_smote
)

def load_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna()  # Drop any NaNs to avoid model errors
    df.reset_index(drop=True, inplace=True)

    # Binary label assumed to be in a column called 'Label'
    if 'Label' not in df.columns:
        raise ValueError(f"Dataset {path} missing required 'Label' column.")
    
    return df

def load_multiple_datasets(paths):
    all_dfs = [load_dataset(p) for p in paths]
    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

def extract_features(df):
    X_dict = {}
    if enabled_feature_sets.get("General", False):
        X_dict["General"] = df[general_features]
    if enabled_feature_sets.get("Statistical", False):
        X_dict["Statistical"] = df[statistical_features]
    if enabled_feature_sets.get("Behavioral", False):
        X_dict["Behavioral"] = df[behavioral_features]

    y = df["Label"]
    return X_dict, y

def get_train_test_data():
    #train_df = load_datasets(train_dataset_path)
    train_df = load_multiple_datasets(train_dataset_paths)
    test_df = load_dataset(test_dataset_path)

    X_train_dict, y_train = extract_features(train_df)
    X_test_dict, y_test = extract_features(test_df)

    if use_smote:
        print("Before SMOTE:", y_train.value_counts())
        smote = SMOTE(random_state=random_seed)

        # Concatenate and track column slices
        feature_keys = list(X_train_dict.keys())
        feature_slices = {}
        start = 0
        X_concat_parts = []

        for key in feature_keys:
            X_part = X_train_dict[key]
            end = start + X_part.shape[1]
            feature_slices[key] = (start, end)
            X_concat_parts.append(X_part)
            start = end

        X_concat = pd.concat(X_concat_parts, axis=1)
        X_resampled, y_resampled = smote.fit_resample(X_concat, y_train)

        # Rebuild X_train_dict
        for key in feature_keys:
            s, e = feature_slices[key]
            X_train_dict[key] = pd.DataFrame(
                X_resampled[:, s:e],
                columns=X_train_dict[key].columns
            )

        y_train = pd.Series(y_resampled).reset_index(drop=True)
        print("After SMOTE:", y_train.value_counts())

    if use_class_weights:
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        scale_pos_weight = neg / pos
        class_weights = [neg / (pos + neg), pos / (pos + neg)]
    else:
        scale_pos_weight = None
        class_weights = None

    return X_train_dict, X_test_dict, y_train, y_test, train_df, test_df, scale_pos_weight, class_weights
