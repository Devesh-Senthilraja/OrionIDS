import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.exceptions import NotFittedError, UndefinedMetricWarning
import warnings
import logging
from typing import List, Dict, Any

# Import necessary components from data_loader
from pipeline.data_loader import Experiment, load_and_clean, FEATURE_CATEGORIES

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_evaluation(exp: Experiment) -> List[Dict[str, Any]]:
    """
    For a given Experiment object (which should have trained models),
    computes evaluation metrics for each base and meta model on its test data.
    Results are saved to a JSON file in the 'results/exp_name' directory.

    Args:
        exp (Experiment): An Experiment object containing trained models and test file paths.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains
                              metrics for a specific model/combo on a specific test file.

    Raises:
        FileNotFoundError: If a test data file does not exist.
        RuntimeError: If there's an issue loading data, predicting, or saving results.
    """
    logger.info(f"Starting evaluation for experiment: '{exp.name}'")

    results = []

    if not exp.test_file or not exp.test_file.exists():
        logger.error(f"Test file for experiment '{exp.name}' not found at '{exp.test_file}'. Skipping evaluation.")
        raise FileNotFoundError(f"Test file not found: {exp.test_file}")

    try:
        if exp.config.get("verbose", False):
            logger.info(f"[EVAL] Loading test set {exp.test_file.name}")

        # Load and clean test data
        df = load_and_clean(exp.test_file)

        if df.empty:
            logger.warning(f"Test data for '{exp.test_file.name}' is empty after cleaning. Cannot evaluate.")
            return []

        y_true = df["Label"]
        if y_true.nunique() < 2:
            logger.warning(f"Test set '{exp.test_file.name}' has only one class ({y_true.iloc[0]}). Metrics may be undefined.")
            # Suppress UndefinedMetricWarning for single-class cases if that's acceptable
            # warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        # --- Evaluate Base Models ---
        if not exp.trained_base:
            logger.warning(f"No base models trained for experiment '{exp.name}'. Skipping base model evaluation.")

        for name, (mdl, features) in exp.trained_base.items():
            if not features or not all(f in df.columns for f in features):
                logger.error(f"Features for base model '{name}' are missing in test data. Skipping.")
                continue

            X_base = df[features]

            try:
                # Suppress specific warnings from sklearn metrics for single-class
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                    y_pred_base = mdl.predict(X_base)

                # Check for consistent lengths
                if len(y_true) != len(y_pred_base):
                    logger.error(f"Mismatched lengths of true ({len(y_true)}) and predicted ({len(y_pred_base)}) labels for base model '{name}'. Skipping.")
                    continue

                metrics = {
                    "experiment": exp.name,
                    "data_file": exp.test_file.name,
                    "model_type": "base",
                    "combo": name,
                    "accuracy": accuracy_score(y_true, y_pred_base),
                    "f1": f1_score(y_true, y_pred_base, zero_division=0), # zero_division=0 avoids warning if no positive samples
                    "precision": precision_score(y_true, y_pred_base, zero_division=0),
                    "recall": recall_score(y_true, y_pred_base, zero_division=0)
                }
                results.append(metrics)
                if exp.config.get("verbose", False):
                    logger.info(f"  [base] {name}: F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}")
            except NotFittedError:
                logger.error(f"Base model '{name}' is not fitted. Cannot make predictions.")
            except ValueError as e:
                logger.error(f"ValueError during prediction for base model '{name}': {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during base model '{name}' prediction or metric calculation: {e}")

        # --- Evaluate Meta Models ---
        if exp.trained_meta:
            logger.info(f"[EVAL] Evaluating meta-models for '{exp.name}'")
            # Assemble meta-DataFrame for this test df using base model predictions
            base_preds_meta_input = {}
            for nm, (mdl, feat) in exp.trained_base.items():
                try:
                    # Ensure features exist in the current test_df
                    if all(f in df.columns for f in feat):
                        # Use predict_proba for meta-features, take probability of the positive class
                        base_preds_meta_input[nm] = mdl.predict_proba(df[feat])[:, 1]
                    else:
                        logger.warning(f"Features for base model '{nm}' not found in test data for meta-feature creation. Skipping predictions for this base model.")
                except NotFittedError:
                    logger.error(f"Base model '{nm}' not fitted, cannot generate meta-features. Skipping.")
                except AttributeError: # For models like LinearSVC that don't have predict_proba
                    logger.warning(f"Base model '{nm}' does not support predict_proba. Skipping for meta-feature generation.")
                except Exception as e:
                    logger.error(f"Error generating predictions for base model '{nm}' (for meta-features): {e}")

            if not base_preds_meta_input:
                logger.warning(f"No base model predictions generated for meta-models in experiment '{exp.name}'. Skipping meta-model evaluation.")
            else:
                meta_df = pd.DataFrame(base_preds_meta_input)

                # Add raw meta-features if configured and available
                if exp.config.get("include_meta_features", False):
                    # We need to know which raw meta-features were used for the meta-model
                    # This assumes all meta-models use the same set of raw meta-features
                    # or that FEATURE_CATEGORIES["meta_features"] is the correct set.
                    # A more robust approach would save the meta-features used per meta-model during training.

                    # For now, let's assume meta_models in config specifies categories.
                    # Take categories from the first meta_model config entry
                    first_meta_model_config = next(iter(exp.config["meta_models"].values()), None)
                    if first_meta_model_config:
                        for cat in first_meta_model_config: # Iterate over categories configured for meta-models
                            if cat in FEATURE_CATEGORIES:
                                raw_cols = FEATURE_CATEGORIES[cat]
                                missing_raw_cols = [c for c in raw_cols if c not in df.columns]
                                if missing_raw_cols:
                                    logger.warning(f"Missing raw meta-features {missing_raw_cols} in test data for category '{cat}'. Skipping.")
                                else:
                                    # Ensure column names don't conflict with base model predictions
                                    for col in raw_cols:
                                        if col in meta_df.columns:
                                            logger.warning(f"Raw meta-feature '{col}' conflicts with a base model prediction column. Renaming raw feature to '{col}_raw'.")
                                            meta_df[f"{col}_raw"] = df[col].values
                                        else:
                                            meta_df[col] = df[col].values
                            else:
                                logger.warning(f"Configured meta-feature category '{cat}' not found in FEATURE_CATEGORIES.")
                    else:
                        logger.warning("No meta-model configurations found to determine raw meta-features. Skipping raw meta-feature inclusion.")

                # Check if meta_df is empty or has no features other than label (if label was added)
                if meta_df.empty:
                    logger.warning(f"Meta-features DataFrame for '{exp.name}' is empty. Skipping meta-model evaluation.")
                else:
                    for meta_name, mdl in exp.trained_meta.items():
                        try:
                            # Ensure the meta_df columns match what the meta_model was trained on
                            # This is a crucial check; if not matched, prediction will fail.
                            # The most robust way is to save feature names with the model.
                            # For simplicity, assuming meta_df has all features the model expects.
                            y_pred_meta = mdl.predict(meta_df)

                            if len(y_true) != len(y_pred_meta):
                                logger.error(f"Mismatched lengths of true ({len(y_true)}) and predicted ({len(y_pred_meta)}) labels for meta model '{meta_name}'. Skipping.")
                                continue

                            metrics = {
                                "experiment": exp.name,
                                "data_file": exp.test_file.name,
                                "model_type": "meta",
                                "combo": meta_name,
                                "accuracy": accuracy_score(y_true, y_pred_meta),
                                "f1": f1_score(y_true, y_pred_meta, zero_division=0),
                                "precision": precision_score(y_true, y_pred_meta, zero_division=0),
                                "recall": recall_score(y_true, y_pred_meta, zero_division=0)
                            }
                            results.append(metrics)
                            if exp.config.get("verbose", False):
                                logger.info(f"  [meta] {meta_name}: F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}")
                        except NotFittedError:
                            logger.error(f"Meta model '{meta_name}' is not fitted. Cannot make predictions.")
                        except ValueError as e:
                            logger.error(f"ValueError during prediction for meta model '{meta_name}': {e}. Check meta_df columns vs trained features.")
                        except Exception as e:
                            logger.error(f"An unexpected error occurred during meta model '{meta_name}' prediction or metric calculation: {e}")
        else:
            logger.info(f"No meta models trained for experiment '{exp.name}'. Skipping meta model evaluation.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation of experiment '{exp.name}': {e}")
        raise RuntimeError(f"Evaluation failed for experiment '{exp.name}': {e}")

    # --- Save Results ---
    if exp.config.get("save_metrics_json", False):
        try:
            out_dir = Path("results") / exp.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "metrics.json"

            with open(out_file, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            logger.info(f"[EVAL] Saved metrics → {out_file}")
        except Exception as e:
            logger.error(f"Failed to save evaluation metrics for '{exp.name}': {e}")
            raise RuntimeError(f"Error saving metrics for '{exp.name}': {e}")
    else:
        logger.info(f"Skipping saving metrics for '{exp.name}' as save_metrics_json is false.")

    return results

# Example Usage (for demonstration)
if __name__ == "__main__":
    from pipeline.config_parser import load_config
    from pipeline.data_loader import prepare_experiments
    import os
    import shutil
    import joblib # for saving/loading dummy models

    # Setup dummy environment
    test_config_path = "temp_eval_config.yaml"
    dummy_cleaned_dir = Path("cleaned")
    dummy_results_dir = Path("results")

    # Create dummy config
    dummy_config_content = """
datasets:
  - dummy_test_data.csv
same_day: true
cross_day: false
same_day_cap: 1000
cross_day_cap: 1000
base_models:
  LogisticRegression:
    - flow_metrics
feature_grouping: true
grid_search_base: false
grid_search_meta: false
use_meta: true
meta_models:
  LogisticRegression:
    - meta_features
smote: false
class_weighting: false
include_meta_features: true
n_jobs: 1
verbose: true
plot_heatmaps: false
plot_shap: false
save_metrics_json: true
"""
    with open(test_config_path, "w") as f:
        f.write(dummy_config_content)

    # Create dummy cleaned data directory and file
    dummy_cleaned_dir.mkdir(exist_ok=True)
    # Ensure all NEEDED_COLS are present, and some meta-features for testing
    dummy_data = {col: np.random.rand(50) for col in NEEDED_COLS if col not in ["Label", "Attack Type"]}
    dummy_data["Label"] = np.random.randint(0, 2, 50)
    dummy_data["Attack Type"] = np.random.choice(["Attack", "Normal"], 50)
    # Add specific meta_features from FEATURE_CATEGORIES to ensure they exist
    for cat in ["meta_features"]:
        for f in FEATURE_CATEGORIES[cat]:
            if f not in dummy_data:
                dummy_data[f] = np.random.rand(50)

    pd.DataFrame(dummy_data).to_csv(dummy_cleaned_dir / "dummy_test_data.csv", index=False)
    logger.info("Dummy config and data created for evaluator demo.")

    try:
        cfg = load_config(test_config_path)
        experiments = prepare_experiments(cfg)

        if experiments:
            exp_to_run = experiments[0]

            # --- Dummy Model Training ---
            # Simulate trained_base and trained_meta from trainer.py
            from sklearn.linear_model import LogisticRegression
            dummy_base_model = LogisticRegression(random_state=42)
            dummy_meta_model = LogisticRegression(random_state=42)

            # Create dummy features for base model
            base_features = FEATURE_CATEGORIES["flow_metrics"]
            # Filter dummy_data to only include base_features that exist
            valid_base_features = [f for f in base_features if f in dummy_data]
            if not valid_base_features:
                logger.error("No valid base features for dummy training.")
                raise ValueError("No valid base features for dummy training.")

            # Create dummy training data (same as test data for simplicity)
            X_dummy = pd.DataFrame(dummy_data)[valid_base_features]
            y_dummy = pd.DataFrame(dummy_data)["Label"]

            # Train dummy base model
            dummy_base_model.fit(X_dummy, y_dummy)
            exp_to_run.trained_base = {"DummyBaseLR": (dummy_base_model, valid_base_features)}
            logger.info("Dummy base model trained.")

            # Prepare meta features for dummy meta model
            base_preds_for_meta = dummy_base_model.predict_proba(X_dummy)[:, 1]
            meta_input_df = pd.DataFrame({"DummyBaseLR": base_preds_for_meta})

            # Add raw meta-features if configured
            if cfg["include_meta_features"]:
                meta_raw_features = FEATURE_CATEGORIES["meta_features"]
                for f in meta_raw_features:
                    if f in dummy_data:
                        if f in meta_input_df.columns: # Avoid conflict
                            meta_input_df[f"{f}_raw"] = pd.DataFrame(dummy_data)[f]
                        else:
                            meta_input_df[f] = pd.DataFrame(dummy_data)[f]
                    else:
                        logger.warning(f"Raw meta-feature '{f}' not found in dummy data for meta training.")

            # Train dummy meta model
            dummy_meta_model.fit(meta_input_df, y_dummy)
            exp_to_run.trained_meta = {"DummyMetaLR": dummy_meta_model}
            logger.info("Dummy meta model trained.")

            # Run evaluation
            metrics_results = run_evaluation(exp_to_run)
            print("\n--- Evaluation Results (first 3 entries) ---")
            pprint.pprint(metrics_results[:3])

            # Verify JSON output
            output_json_path = dummy_results_dir / exp_to_run.name / "metrics.json"
            if output_json_path.exists():
                logger.info(f"Metrics JSON file found at: {output_json_path}")
                with open(output_json_path, "r") as f:
                    loaded_metrics = json.load(f)
                logger.info(f"Loaded {len(loaded_metrics)} metrics entries from JSON.")
        else:
            logger.warning("No experiments to run for evaluation demo.")

    except Exception as e:
        logger.error(f"Evaluation demo failed: {e}")
    finally:
        # Clean up
        if Path(test_config_path).exists():
            Path(test_config_path).unlink()
        if dummy_cleaned_dir.exists():
            shutil.rmtree(dummy_cleaned_dir)
        if dummy_results_dir.exists():
            shutil.rmtree(dummy_results_dir)
        logger.info("Cleaned up dummy files and directories for evaluator demo.")

