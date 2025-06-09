# trainer.py

import json
import pickle
import hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from config import cache_dir, use_grid_search, use_cross_validation, n_cv_splits, random_seed, disable_cache
from logger import log_metrics, plot_confusion

cache_dir = Path(cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

def get_param_hash(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()

def train_model(model_name, model, param_grid, X_train, y_train):
    if use_grid_search:
        search = GridSearchCV(model, param_grid, cv=n_cv_splits, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
    else:
        model.set_params(**{k: v[0] for k, v in param_grid.items()})
        best_model = model.fit(X_train, y_train)
        best_params = {k: v[0] for k, v in param_grid.items()}

    return best_model, best_params

def evaluate_model(model_name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    plot_confusion(cm, model_name)

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm.tolist()
    }

    log_metrics(model_name, metrics)
    return preds, probs, metrics

def run_base_models(model_configs, X_train_dict, X_test_dict, y_train, y_test):
    predictions_train = {}
    predictions_test = {}

    for name, cfg in model_configs.items():
        X_train = X_train_dict[cfg["feature_key"]]
        X_test = X_test_dict[cfg["feature_key"]]

        # Create a unique hash for caching
        param_hash = get_param_hash(cfg["param_grid"])
        model_path = cache_dir / f"{name}_{param_hash}.pkl"

        if model_path.exists() and not disable_cache:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            model, best_params = train_model(name, cfg["model"], cfg["param_grid"], X_train, y_train)
            if not disable_cache:
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

        preds, probs, _ = evaluate_model(name, model, X_test, y_test)
        predictions_train[name] = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_train)
        predictions_test[name] = probs

    return predictions_train, predictions_test