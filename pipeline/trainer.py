import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable, Union
import logging
import joblib # For saving/loading models, good practice for pipelines

from pipeline.data_loader import Experiment, load_and_clean, FEATURE_CATEGORIES, NEEDED_COLS
from pipeline.sampling import apply_smote, downsample_df

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError, ConvergenceWarning
import warnings

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Suppress sklearn warnings about convergence if max_iter is too small (e.g., for LogisticRegression, LinearSVC)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Define a type hint for model constructors and their grids
ModelTuple = Tuple[Callable[[], BaseEstimator], Dict[str, Any]]

MODELS: Dict[str, ModelTuple] = {
    "LogisticRegression": (
        lambda n_jobs: LogisticRegression(solver="saga", max_iter=1000, random_state=42, n_jobs=n_jobs if n_jobs != 0 else 1),
        {"C": [0.01, 0.1, 1]}
    ),
    "DecisionTree": (
        lambda n_jobs: DecisionTreeClassifier(random_state=42),
        {"max_depth": [5, 10, 20]}
    ),
    "RandomForest": (
        lambda n_jobs: RandomForestClassifier(random_state=42, n_jobs=n_jobs if n_jobs != 0 else 1),
        {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
    ),
    "ExtraTrees": (
        lambda n_jobs: ExtraTreesClassifier(random_state=42, n_jobs=n_jobs if n_jobs != 0 else 1),
        {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
    ),
    "XGBoost": (
        lambda n_jobs: XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=n_jobs if n_jobs != 0 else 1),
        {"learning_rate": [0.1, 0.01], "n_estimators": [100, 200], "max_depth": [3, 5]}
    ),
    "LightGBM": (
        lambda n_jobs: LGBMClassifier(random_state=42, n_jobs=n_jobs if n_jobs != 0 else 1),
        {"learning_rate": [0.1, 0.01], "n_estimators": [100, 200], "num_leaves": [31, 50]}
    ),
    "CatBoost": (
        # CatBoost's n_jobs is 'thread_count'
        lambda n_jobs: CatBoostClassifier(verbose=0, random_state=42, thread_count=n_jobs if n_jobs != 0 else 1),
        {"depth": [4, 6], "iterations": [100, 200], "learning_rate": [0.1, 0.01]}
    ),
    "KNN": (
        lambda n_jobs: KNeighborsClassifier(n_jobs=n_jobs if n_jobs != 0 else 1),
        {"n_neighbors": [3, 5, 7]}
    ),
    "LinearSVC": (
        # LinearSVC does not support n_jobs, or class_weight directly on init, but via set_params
        lambda n_jobs: LinearSVC(max_iter=10000, random_state=42, dual=False), # dual=False for n_features > n_samples
        {"C": [0.1, 1, 10]}
    ),
    "MLP": (
        lambda n_jobs: MLPClassifier(max_iter=500, random_state=42), # MLPClassifier does not have n_jobs
        {"hidden_layer_sizes": [(50,), (100,)], "alpha": [1e-4, 1e-3]}
    )
}

def get_model_and_grid(model_name: str, n_jobs: int) -> ModelTuple:
    """Retrieves the model constructor and parameter grid from MODELS."""
    if model_name not in MODELS:
        logger.error(f"Model '{model_name}' not defined in MODELS dictionary.")
        raise KeyError(f"Model '{model_name}' not found.")

    model_constructor, grid = MODELS[model_name]
    # Pass n_jobs to the model constructor if it accepts it
    # This requires the lambda in MODELS to accept n_jobs
    return model_constructor(n_jobs), grid


def run_training(exp: Experiment, verbose: bool = False, n_jobs: int = 1):
    """
    For one Experiment:
      1) loads & caps each training file.
      2) applies SMOTE or class weighting for imbalance handling.
      3) trains each base model ↔ feature-set combination (with optional GridSearchCV).
      4) optionally stacks predictions and trains meta-models (with optional GridSearchCV).
    Trained models are stored within the Experiment object.

    Args:
        exp (Experiment): The Experiment object to run training for.
        verbose (bool): If True, print detailed messages.
        n_jobs (int): Number of jobs to run in parallel for GridSearchCV and models.
                      -1 means use all available processors.
    """
    logger.info(f"Starting training for experiment: '{exp.name}' with n_jobs={n_jobs}")

    # --- 1) LOAD & CAP TRAINING DATA ---
    dfs: List[pd.DataFrame] = []
    for fp in exp.train_files:
        try:
            if verbose:
                logger.info(f"[TRAIN] Loading training data from '{fp.name}'")
            df = load_and_clean(fp)
            if df.empty:
                logger.warning(f"File '{fp.name}' is empty after cleaning. Skipping this file.")
                continue

            if len(df) > exp.cap_train_per_file:
                df = downsample_df(df, exp.cap_train_per_file, random_state=42)
                if verbose:
                    logger.info(f"  ↘ downsampled '{fp.name}' to {len(df)} rows.")
            dfs.append(df)
        except FileNotFoundError as e:
            logger.error(f"Training file '{fp.name}' not found: {e}. Skipping.")
        except Exception as e:
            logger.error(f"Error loading or processing training file '{fp.name}': {e}. Skipping.")

    if not dfs:
        logger.error(f"No valid training dataframes loaded for experiment '{exp.name}'. Cannot proceed with training.")
        return

    train_df = pd.concat(dfs, ignore_index=True)
    if train_df.empty:
        logger.error(f"Combined training dataframe is empty for experiment '{exp.name}'. Cannot proceed.")
        return

    # Check if 'Label' column exists and has at least two unique classes
    if "Label" not in train_df.columns:
        logger.error(f"'Label' column not found in training data for experiment '{exp.name}'. Cannot train models.")
        return
    if train_df["Label"].nunique() < 2:
        logger.error(f"Only one class found in 'Label' column for experiment '{exp.name}'. Cannot train classification models.")
        return

    X_train_raw = train_df.drop(columns=["Attack Type", "Label"], errors='ignore')
    y_train_raw = train_df["Label"]

    # --- 2) HANDLE CLASS IMBALANCE ---
    class_weight: Union[str, Dict[int, float], None] = None
    X_processed = X_train_raw
    y_processed = y_train_raw

    if exp.config["smote"]:
        if verbose:
            logger.info("[TRAIN] Applying SMOTE to training data.")
        try:
            X_processed, y_processed = apply_smote(X_train_raw, y_train_raw, random_state=42)
            logger.info(f"SMOTE applied. New sample size: {len(X_processed)}.")
        except Exception as e:
            logger.error(f"SMOTE application failed: {e}. Proceeding without SMOTE.")
            X_processed = X_train_raw
            y_processed = y_train_raw
    elif exp.config["class_weighting"]:
        class_weight = "balanced"
        if verbose:
            logger.info("[TRAIN] Using class_weight='balanced'.")

    logger.info(f"Processed training data shape: {X_processed.shape}, labels: {y_processed.shape}")
    logger.info(f"Processed training data class distribution: {y_processed.value_counts().to_dict()}")

    # --- 3) TRAIN BASE MODELS ---
    exp.trained_base = {}
    for model_name, feat_sets in exp.config["base_models"].items():
        try:
            base_model_inst, base_grid = get_model_and_grid(model_name, n_jobs)

            # Decide grouping vs. separate feature sets
            if exp.config["feature_grouping"]:
                # Concatenate all features from the specified categories
                all_features_for_group = []
                for f_cat in feat_sets:
                    if f_cat in FEATURE_CATEGORIES:
                        all_features_for_group.extend(FEATURE_CATEGORIES[f_cat])
                    else:
                        logger.warning(f"Feature category '{f_cat}' not found in FEATURE_CATEGORIES. Skipping.")

                # Remove duplicates and ensure features are present in X_processed
                unique_features = sorted(list(set(all_features_for_group)))
                actual_features = [f for f in unique_features if f in X_processed.columns]

                if not actual_features:
                    logger.warning(f"No valid features found for base model '{model_name}' with grouping '{feat_sets}'. Skipping.")
                    continue

                combos = [(model_name + "_all_features", actual_features)]
            else:
                # Train a separate model for each feature set
                combos = []
                for fset in feat_sets:
                    if fset in FEATURE_CATEGORIES:
                        features = [f for f in FEATURE_CATEGORIES[fset] if f in X_processed.columns]
                        if features:
                            combos.append((f"{model_name}_{fset}", features))
                        else:
                            logger.warning(f"No valid features found for base model '{model_name}' with feature set '{fset}'. Skipping.")
                    else:
                        logger.warning(f"Feature category '{fset}' not found in FEATURE_CATEGORIES. Skipping.")

            if not combos:
                logger.warning(f"No valid base model combos generated for model '{model_name}'. Skipping.")
                continue

            for combo_name, features in combos:
                try:
                    if verbose:
                        logger.info(f"[TRAIN] Training base model '{combo_name}' on {len(features)} features.")

                    current_model = base_model_inst.__class__(**base_model_inst.get_params()) # Create a new instance

                    # Apply class_weight if supported by the model
                    if class_weight and hasattr(current_model, 'class_weight'):
                        try:
                            current_model.set_params(class_weight=class_weight)
                            logger.debug(f"Set class_weight='{class_weight}' for '{combo_name}'.")
                        except TypeError:
                            logger.warning(f"Model '{model_name}' does not support 'class_weight' parameter. Skipping.")

                    X_base = X_processed[features]
                    y_base = y_processed

                    if X_base.empty:
                        logger.warning(f"Feature matrix for '{combo_name}' is empty. Skipping training.")
                        continue

                    if exp.config["grid_search_base"] and base_grid:
                        if verbose:
                            logger.info(f"  ↗ Applying GridSearchCV (base) for '{combo_name}'.")
                        # Use StratifiedKFold for classification tasks if 'Label' is available
                        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        grid_search = GridSearchCV(
                            current_model, base_grid,
                            cv=cv_strategy, n_jobs=n_jobs,
                            scoring="f1", verbose=0 # Suppress GridSearchCV internal verbose
                        )
                        grid_search.fit(X_base, y_base)
                        trained_model = grid_search.best_estimator_
                        logger.info(f"GridSearchCV best params for '{combo_name}': {grid_search.best_params_}, F1: {grid_search.best_score_:.3f}")
                    else:
                        if verbose:
                            logger.info(f"  ↗ Fitting base model '{combo_name}' without GridSearchCV.")
                        trained_model = current_model
                        trained_model.fit(X_base, y_base)

                    exp.trained_base[combo_name] = (trained_model, features)
                    logger.info(f"Successfully trained base model '{combo_name}'.")

                except KeyError as e:
                    logger.error(f"Missing feature in training data for '{combo_name}': {e}. Skipping.")
                except ValueError as e:
                    logger.error(f"ValueError during base model '{combo_name}' training: {e}. Check data integrity or model parameters.")
                except Exception as e:
                    logger.error(f"An unexpected error occurred during base model '{combo_name}' training: {e}. Skipping.")
        except Exception as e:
            logger.error(f"Error setting up base models for '{model_name}': {e}. Skipping this model type.")

    if not exp.trained_base:
        logger.error(f"No base models were successfully trained for experiment '{exp.name}'. Cannot proceed with meta-training or evaluation.")
        return

    # --- 4) TRAIN META MODEL (if enabled) ---
    exp.trained_meta = {}
    if exp.config["use_meta"]:
        if verbose:
            logger.info("[TRAIN] Building meta-features and training meta-models.")

        # 1) Build DataFrame of all base-model probability columns
        base_preds_df_list = []
        for name, (mdl, feats) in exp.trained_base.items():
            try:
                # Ensure the model is fitted and supports predict_proba
                if hasattr(mdl, 'predict_proba'):
                    base_preds_df_list.append(pd.Series(mdl.predict_proba(X_processed[feats])[:, 1], name=name))
                else:
                    logger.warning(f"Base model '{name}' does not support 'predict_proba'. Skipping its output as meta-feature.")
            except NotFittedError:
                logger.warning(f"Base model '{name}' is not fitted. Cannot generate predictions for meta-features. Skipping.")
            except KeyError as e:
                logger.warning(f"Features for base model '{name}' not found in X_processed for meta-feature generation: {e}. Skipping.")
            except Exception as e:
                logger.error(f"Error generating predictions for base model '{name}' for meta-features: {e}. Skipping.")

        if not base_preds_df_list:
            logger.warning("No base model predictions generated for meta-training. Skipping meta-models.")
            exp.config["use_meta"] = False # Disable meta if no predictions
            return

        base_preds_meta_df = pd.concat(base_preds_df_list, axis=1)

        # 2) Loop each meta_model → list of raw feature-categories
        for meta_name, raw_cats_config in exp.config["meta_models"].items():
            try:
                meta_model_inst, meta_grid = get_model_and_grid(meta_name, n_jobs)

                # 2a) Decide grouping vs. separate combos for meta-models
                if exp.config["feature_grouping"]:
                    combos = [(f"{meta_name}_all_meta", raw_cats_config)]
                else:
                    combos = [(f"{meta_name}_{cat}", [cat]) for cat in raw_cats_config]

                for combo_name, categories_to_include in combos:
                    try:
                        if verbose:
                            logger.info(f"[TRAIN] Training meta model '{combo_name}'.")

                        current_meta_model = meta_model_inst.__class__(**meta_model_inst.get_params())
                        if class_weight and hasattr(current_meta_model, 'class_weight'):
                            try:
                                current_meta_model.set_params(class_weight=class_weight)
                                logger.debug(f"Set class_weight='{class_weight}' for meta model '{combo_name}'.")
                            except TypeError:
                                logger.warning(f"Meta model '{meta_name}' does not support 'class_weight' parameter. Skipping.")

                        df_meta_input = base_preds_meta_df.copy()

                        # Include raw meta-features if configured
                        if exp.config["include_meta_features"]:
                            raw_features_to_add = []
                            for cat in categories_to_include:
                                if cat in FEATURE_CATEGORIES:
                                    # Filter for features actually present in the original data
                                    raw_features_to_add.extend([f for f in FEATURE_CATEGORIES[cat] if f in X_processed.columns])
                                else:
                                    logger.warning(f"Meta-feature category '{cat}' not found in FEATURE_CATEGORIES for raw feature inclusion. Skipping.")

                            # Add raw features to meta_input_df, handling potential column name conflicts
                            for col in raw_features_to_add:
                                if col in df_meta_input.columns:
                                    # If a raw feature name clashes with a base model prediction name, rename it.
                                    logger.warning(f"Raw meta-feature '{col}' conflicts with base model prediction. Renaming to '{col}_raw'.")
                                    df_meta_input[f"{col}_raw"] = X_processed[col].values
                                else:
                                    df_meta_input[col] = X_processed[col].values

                        # Use .values if model expects numpy array, else keep DataFrame
                        X_meta = df_meta_input
                        y_meta = y_processed

                        if X_meta.empty:
                            logger.warning(f"Meta-feature matrix for '{combo_name}' is empty. Skipping training.")
                            continue

                        if exp.config["grid_search_meta"] and meta_grid:
                            if verbose:
                                logger.info(f"  ↗ Applying GridSearchCV (meta) for '{combo_name}'.")
                            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                            grid_search_meta = GridSearchCV(
                                current_meta_model, meta_grid,
                                cv=cv_strategy, n_jobs=n_jobs,
                                scoring="f1", verbose=0
                            )
                            grid_search_meta.fit(X_meta, y_meta)
                            trained_meta_model = grid_search_meta.best_estimator_
                            logger.info(f"GridSearchCV best params for meta '{combo_name}': {grid_search_meta.best_params_}, F1: {grid_search_meta.best_score_:.3f}")
                        else:
                            if verbose:
                                logger.info(f"  ↗ Fitting meta model '{combo_name}' without GridSearchCV.")
                            trained_meta_model = current_meta_model
                            trained_meta_model.fit(X_meta, y_meta)

                        exp.trained_meta[combo_name] = trained_meta_model
                        logger.info(f"Successfully trained meta model '{combo_name}'.")

                    except KeyError as e:
                        logger.error(f"Missing feature for meta model '{combo_name}': {e}. Skipping.")
                    except ValueError as e:
                        logger.error(f"ValueError during meta model '{combo_name}' training: {e}. Check data or model parameters.")
                    except Exception as e:
                        logger.error(f"An unexpected error occurred during meta model '{combo_name}' training: {e}. Skipping.")
            except Exception as e:
                logger.error(f"Error setting up meta models for '{meta_name}': {e}. Skipping this meta model type.")

        if verbose:
            logger.info(f"[TRAIN] Done meta-models: {list(exp.trained_meta.keys())}\n")
    else:
        logger.info(f"Meta-models disabled for experiment '{exp.name}'. Skipping meta-training.")

    logger.info(f"[TRAIN] Finished training for experiment '{exp.name}'.")

# Example Usage (for demonstration)
if __name__ == "__main__":
    from pipeline.config_parser import load_config
    from pipeline.data_loader import prepare_experiments
    import os
    import shutil

    # Setup dummy environment for trainer demo
    test_config_path = "temp_train_config.yaml"
    dummy_cleaned_dir = Path("cleaned")

    # Create dummy config file
    dummy_config_content = """
datasets:
  - dummy_train_data_A.csv
  - dummy_train_data_B.csv
same_day: true
cross_day: false
same_day_cap: 5000 # Cap for demo
cross_day_cap: 5000
base_models:
  LogisticRegression:
    - flow_metrics
  RandomForest:
    - packet_size_stats
feature_grouping: false
grid_search_base: true
grid_search_meta: true
use_meta: true
meta_models:
  LogisticRegression:
    - meta_features
smote: true
class_weighting: false # SMOTE takes precedence
include_meta_features: true
n_jobs: 1 # Set to -1 for full parallelization if supported by your system
verbose: true
plot_heatmaps: false
plot_shap: false
save_metrics_json: false
"""
    with open(test_config_path, "w") as f:
        f.write(dummy_config_content)

    # Create dummy cleaned data directory and files
    dummy_cleaned_dir.mkdir(exist_ok=True)

    # Create dummy dataframes that include all needed columns and some with imbalanced labels
    # Use a subset of NEEDED_COLS for simpler dummy data, ensuring 'Label' and 'Attack Type' are present
    mock_needed_cols = list(NEEDED_COLS) # Convert set to list for indexing

    for ds_name in ["dummy_train_data_A.csv", "dummy_train_data_B.csv"]:
        num_samples = 60 if "A" in ds_name else 120 # Make one small, one larger for capping demo
        df_dummy = pd.DataFrame(np.random.rand(num_samples, len(mock_needed_cols)), columns=mock_needed_cols)
        # Create imbalanced labels for SMOTE testing
        df_dummy["Label"] = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
        df_dummy["Attack Type"] = np.random.choice(["Attack", "Normal"], size=num_samples)
        df_dummy.to_csv(dummy_cleaned_dir / ds_name, index=False)

    logger.info("Dummy config and data created for trainer demo.")

    try:
        cfg = load_config(test_config_path)
        experiments = prepare_experiments(cfg)

        if experiments:
            exp_to_run = experiments[0] # Take the first prepared experiment
            run_training(exp_to_run, verbose=exp_to_run.config["verbose"], n_jobs=exp_to_run.config["n_jobs"])

            # Verify if models were trained and stored
            print(f"\n--- Training Results for '{exp_to_run.name}' ---")
            print(f"Trained Base Models: {list(exp_to_run.trained_base.keys())}")
            print(f"Trained Meta Models: {list(exp_to_run.trained_meta.keys())}")

            # Optional: Test a prediction from a trained model
            if exp_to_run.trained_base:
                first_base_model_name = list(exp_to_run.trained_base.keys())[0]
                model, features = exp_to_run.trained_base[first_base_model_name]
                sample_data = pd.DataFrame(np.random.rand(1, len(features)), columns=features)
                try:
                    pred = model.predict(sample_data)
                    print(f"Sample prediction from base model '{first_base_model_name}': {pred}")
                except NotFittedError:
                    print(f"Model {first_base_model_name} not fitted for prediction test.")
                except Exception as e:
                    print(f"Error during sample prediction for base model {first_base_model_name}: {e}")

            if exp_to_run.trained_meta and exp_to_run.trained_base:
                first_meta_model_name = list(exp_to_run.trained_meta.keys())[0]
                meta_model = exp_to_run.trained_meta[first_meta_model_name]

                # Construct a dummy meta input for prediction
                dummy_base_preds = {
                    name: mdl.predict_proba(sample_data_base)[0,1] # Assume sample_data_base is compatible
                    for name, (mdl, sample_data_base) in exp_to_run.trained_base.items()
                    if hasattr(mdl, 'predict_proba')
                }
                dummy_meta_input = pd.DataFrame([dummy_base_preds])

                # Add raw meta features if applicable
                if exp_to_run.config["include_meta_features"] and first_meta_model_name in exp_to_run.config["meta_models"]:
                    for cat in exp_to_run.config["meta_models"][first_meta_model_name]:
                        if cat in FEATURE_CATEGORIES:
                            for f in FEATURE_CATEGORIES[cat]:
                                if f in X_train_raw.columns: # Check original data for existence
                                    if f in dummy_meta_input.columns:
                                        dummy_meta_input[f"{f}_raw"] = X_train_raw[f].iloc[0] # Use a sample value
                                    else:
                                        dummy_meta_input[f] = X_train_raw[f].iloc[0]
                                else:
                                    logger.warning(f"Raw feature '{f}' for meta prediction not found in original data.")

                try:
                    if not dummy_meta_input.empty:
                        meta_pred = meta_model.predict(dummy_meta_input)
                        print(f"Sample prediction from meta model '{first_meta_model_name}': {meta_pred}")
                    else:
                        print("Dummy meta input is empty. Skipping meta prediction test.")
                except NotFittedError:
                    print(f"Meta model {first_meta_model_name} not fitted for prediction test.")
                except Exception as e:
                    print(f"Error during sample prediction for meta model {first_meta_model_name}: {e}")

        else:
            logger.warning("No experiments to run for trainer demo.")

    except Exception as e:
        logger.error(f"Trainer demo failed: {e}")
    finally:
        # Clean up dummy files
        if Path(test_config_path).exists():
            Path(test_config_path).unlink()
        if dummy_cleaned_dir.exists():
            shutil.rmtree(dummy_cleaned_dir)
        logger.info("Cleaned up dummy files and directory for trainer demo.")

