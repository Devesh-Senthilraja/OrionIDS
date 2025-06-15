# models.py

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from config import random_seed, use_grid_search, use_class_weights, n_jobs

def get_model_configs(scale_pos_weight=None, class_weights=None):
    configs = {}

    if use_class_weights: 
        # LightGBM on General Features
        configs["LightGBM_General"] = {
            "model": LGBMClassifier(n_estimators=100, random_state=random_seed, n_jobs=n_jobs, class_weight='balanced'),
            "param_grid": {
                "learning_rate": [0.1, 0.01] if use_grid_search else [0.1],
                "num_leaves": [31, 50] if use_grid_search else [31],
                "max_depth": [10, 20] if use_grid_search else [10]
            },
            "feature_key": "General"
        }

        # XGBoost on Statistical Features
        configs["XGBoost_Statistical"] = {
            "model": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', n_jobs=n_jobs, random_state=random_seed, scale_pos_weight=scale_pos_weight),
            "param_grid": {
                "learning_rate": [0.1, 0.01] if use_grid_search else [0.1],
                "max_depth": [5, 10] if use_grid_search else [5]
            },
            "feature_key": "Statistical"
        }

        # CatBoost on Behavioral Features
        configs["CatBoost_Behavioral"] = {
            "model": CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0, random_state=random_seed, n_jobs=n_jobs, class_weights=class_weights),
            "param_grid": {
                "depth": [4, 6, 8] if use_grid_search else [6],
                "learning_rate": [0.1, 0.01] if use_grid_search else [0.1]
            },
            "feature_key": "Behavioral"
        }

    else:
        # LightGBM on General Features
        configs["LightGBM_General"] = {
            "model": LGBMClassifier(n_estimators=100, random_state=random_seed),
            "param_grid": {
                "learning_rate": [0.1, 0.01] if use_grid_search else [0.1],
                "num_leaves": [31, 50] if use_grid_search else [31],
                "max_depth": [10, 20] if use_grid_search else [10]
            },
            "feature_key": "General"
        }

        # XGBoost on Statistical Features
        configs["XGBoost_Statistical"] = {
            "model": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=random_seed),
            "param_grid": {
                "learning_rate": [0.1, 0.01] if use_grid_search else [0.1],
                "max_depth": [5, 10] if use_grid_search else [5]
            },
            "feature_key": "Statistical"
        }

        # CatBoost on Behavioral Features
        configs["CatBoost_Behavioral"] = {
            "model": CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0, random_state=random_seed),
            "param_grid": {
                "depth": [4, 6, 8] if use_grid_search else [6],
                "learning_rate": [0.1, 0.01] if use_grid_search else [0.1]
            },
            "feature_key": "Behavioral"
        }

    return configs
