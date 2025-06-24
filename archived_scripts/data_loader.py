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
    add_meta_features,
    meta_features,
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
    
    if use_smote:
        smote_columns = []
        if enabled_feature_sets.get("General", False):
            smote_columns.extend(general_features)
        if enabled_feature_sets.get("Statistical", False):
            smote_columns.extend(statistical_features)
        if enabled_feature_sets.get("Behavioral", False):
            smote_columns.extend(behavioral_features)
        if add_meta_features:
            for col in meta_features:
                if col not in smote_columns:
                    smote_columns.append(col)

        smote = SMOTE(random_state=random_seed)

        X_res, y_res = smote.fit_resample(train_df[smote_columns], train_df["Label"])
        train_df = pd.DataFrame(X_res, columns=smote_columns)
        train_df["Label"] = y_res

    X_train_dict, y_train = extract_features(train_df)
    X_test_dict, y_test = extract_features(test_df)

    if use_class_weights:
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        scale_pos_weight = neg / pos
        class_weights = [neg / (pos + neg), pos / (pos + neg)]
    else:
        scale_pos_weight = None
        class_weights = None

    return X_train_dict, X_test_dict, y_train, y_test, train_df, test_df, scale_pos_weight, class_weights
