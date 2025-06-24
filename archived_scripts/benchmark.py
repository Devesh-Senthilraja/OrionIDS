from data_loader import load_multiple_datasets, load_dataset
from trainer import train_model, evaluate_model
from models import get_model_configs
from config import (
    train_dataset_paths, test_dataset_path,
    general_features, statistical_features, behavioral_features, meta_features,
    random_seed
)
from imblearn.over_sampling import SMOTE
import json

# Define all feature set options
FEATURE_SETS = {
    "General": general_features,
    "Statistical": statistical_features,
    "Behavioral": behavioral_features,
    "All": list({*general_features, *statistical_features, *behavioral_features})
}

def prepare_features(df, features, include_meta):
    feats = features.copy()
    if include_meta:
        # Add meta features unless they're already present
        for m in meta_features:
            if m not in feats:
                feats.append(m)
    feats = list(dict.fromkeys(feats))
    return df[feats]

def run_benchmarks(smote=True, include_meta=True, results_path="results/benchmark_base_models.json"):
    train_df = load_multiple_datasets(train_dataset_paths)
    test_df = load_dataset(test_dataset_path)
    y_train = train_df["Label"]
    y_test = test_df["Label"]

    # Loop through all models and feature sets
    results = {}
    base_configs = get_model_configs()
    for model_name, cfg in base_configs.items():
        for fs_name, fs_cols in FEATURE_SETS.items():
            # Prepare features
            X_train = prepare_features(train_df, fs_cols, include_meta)
            X_test = prepare_features(test_df, fs_cols, include_meta)
            y_train_exp = y_train

            # Optionally apply SMOTE
            if smote:
                sm = SMOTE(random_state=random_seed)
                X_train, y_train_exp = sm.fit_resample(X_train, y_train)

            # Train and evaluate
            model, best_params = train_model(f"{model_name}_{fs_name}", cfg["model"], cfg["param_grid"], X_train, y_train_exp)
            preds, probs, metrics = evaluate_model(f"{model_name}_{fs_name}", model, X_test, y_test)
            tag = f"{model_name}__{fs_name}{'__meta' if include_meta else ''}{'__SMOTE' if smote else ''}"
            results[tag] = metrics
            print(f"\n{tag}\n{json.dumps(metrics, indent=2)}")

    # Save all results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {results_path}")

if __name__ == "__main__":
    # Try all model/feature combos, with meta and SMOTE enabled
    run_benchmarks(smote=True, include_meta=True)
