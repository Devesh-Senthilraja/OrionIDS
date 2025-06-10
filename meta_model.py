# meta_model.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from config import add_meta_features, meta_features, use_cv_meta_model, random_seed
from trainer import evaluate_model

def build_meta_inputs(predictions_dict, meta_raw, index_subset, use_meta_features):
    # Stack predictions from base models
    base_preds = np.column_stack([predictions_dict[k] for k in sorted(predictions_dict.keys())])

    if use_meta_features:
        meta_additional = meta_raw.loc[index_subset, meta_features].values
        meta_input = np.column_stack((base_preds, meta_additional))
    else:
        meta_input = base_preds

    return meta_input

def train_meta_model(pred_train, pred_test, y_train, y_test, train_df, test_df):
    # Get indices to align meta features
    train_idx = y_train.index
    test_idx = y_test.index

    # Build inputs
    X_meta_train = build_meta_inputs(pred_train, train_df, train_idx, add_meta_features)
    X_meta_test = build_meta_inputs(pred_test, test_df, test_idx, add_meta_features)

    # Define meta-model
    meta_model = RandomForestClassifier(n_estimators=100, random_state=random_seed)

    if use_cv_meta_model:
        # Safer training: use CV predictions on train set
        y_pred_train = cross_val_predict(meta_model, X_meta_train, y_train, cv=5, method='predict_proba')[:, 1]
        meta_model.fit(X_meta_train, y_train)  # Train final model for test use
    else:
        meta_model.fit(X_meta_train, y_train)
        y_pred_train = meta_model.predict_proba(X_meta_train)[:, 1]

    y_pred_test = meta_model.predict_proba(X_meta_test)[:, 1]
    y_pred_test_bin = (y_pred_test >= 0.5).astype(int)

    evaluate_model("Meta_Model", meta_model, X_meta_test, y_test)

    return y_pred_test_bin, y_pred_test
