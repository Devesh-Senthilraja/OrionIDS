import json
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import logging

# Import Experiment and load_and_clean from data_loader
from pipeline.data_loader import Experiment, load_and_clean, FEATURE_CATEGORIES

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_plots(exp: Experiment, plot_heatmaps: bool, plot_shap: bool, verbose: bool = False) -> None:
    """
    Orchestrates the plotting functions based on configuration.

    Args:
        exp (Experiment): The experiment object.
        plot_heatmaps (bool): Whether to plot F1 score heatmaps.
        plot_shap (bool): Whether to plot SHAP summary for meta-models.
        verbose (bool): If True, print detailed progress.
    """
    exp_name: str = exp.name
    if plot_heatmaps:
        plot_confusion_heatmaps(exp_name, verbose)
    if plot_shap:
        plot_shap_summary(exp_name, exp, verbose)


def plot_confusion_heatmaps(exp_name: str, config: dict, verbose: bool = False):
    """
    Loads metrics from 'results/exp_name/metrics.json' and plots F1 score heatmaps
    for base and meta models across different data files.

    Args:
        exp_name (str): The name of the experiment directory.
        config (dict): The configuration dictionary for the experiment, needed for verbose setting.
        verbose (bool): If True, print detailed messages. Defaults to False.
    """
    metrics_file = Path("results") / exp_name / "metrics.json"

    if not metrics_file.exists():
        logger.warning(f"Metrics file not found for '{exp_name}' at '{metrics_file}'. Skipping heatmap plotting.")
        return

    try:
        with open(metrics_file, "r", encoding='utf-8') as f:
            metrics_data = json.load(f)
        logger.info(f"Loaded metrics from '{metrics_file}' for heatmap plotting.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{metrics_file}': {e}. Skipping heatmap plotting.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading '{metrics_file}': {e}. Skipping heatmap plotting.")
        return

    df = pd.DataFrame(metrics_data)
    if df.empty:
        logger.warning(f"No metrics data found in '{metrics_file}'. Skipping heatmap plotting.")
        return

    # Check config["plot_heatmaps"]
    if not config.get("plot_heatmaps", False):
        logger.info(f"Skipping heatmap plotting for '{exp_name}' as plot_heatmaps is false in config.")
        return

    for mtype in ["base", "meta"]:
        sub_df = df[df["model_type"] == mtype]
        if sub_df.empty:
            logger.info(f"No '{mtype}' model metrics found for experiment '{exp_name}'. Skipping {mtype} heatmap.")
            continue

        try:
            # Pivot table to get F1 scores for each combo across data files
            pivot_table = sub_df.pivot_table("f1", index="data_file", columns="combo")

            # Sort columns for consistent plotting order (optional)
            pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

            plt.figure(figsize=(max(8, pivot_table.shape[1] * 1.5), max(6, pivot_table.shape[0] * 0.8)))
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, linecolor='lightgrey')
            plt.title(f"{exp_name} — {mtype.capitalize()} F1 Across Files")
            plt.xlabel("Model/Feature Combo")
            plt.ylabel("Data File")

            out_file = Path("results") / exp_name / f"{mtype}_f1_heatmap.png"
            out_file.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

            plt.tight_layout()
            plt.savefig(out_file, dpi=300)
            plt.close()
            if verbose:
                logger.info(f"[PLOT] Saved F1 heatmap for {mtype} models → {out_file}")
        except Exception as e:
            logger.error(f"Error plotting {mtype} heatmap for '{exp_name}': {e}")


def plot_shap_summary(exp: Experiment, verbose: bool = False):
    """
    Generates SHAP summary plots for the final meta-models in an experiment.
    This requires re-running predictions to get SHAP values, or loading pre-saved values.

    Args:
        exp (Experiment): The Experiment object containing trained models and config.
        verbose (bool): If True, print detailed messages. Defaults to False.
    """
    meta_models = exp.trained_meta
    if not meta_models:
        if verbose:
            logger.info("[PLOT] No meta models to SHAP for experiment '{exp.name}'. Skipping SHAP plotting.")
        return

    # Check config["plot_shap"]
    if not exp.config.get("plot_shap", False):
        logger.info(f"Skipping SHAP plotting for '{exp.name}' as plot_shap is false in config.")
        return

    if not exp.test_file:
        logger.warning(f"No test file specified for experiment '{exp.name}'. Cannot generate SHAP plots.")
        return

    try:
        # Load the first test file's data for SHAP value computation
        if verbose:
            logger.info(f"[PLOT] Loading test data from '{exp.test_file.name}' for SHAP analysis.")
        df_test = load_and_clean(exp.test_file)

        if df_test.empty:
            logger.warning(f"Test data for '{exp.test_file.name}' is empty after cleaning. Cannot generate SHAP plots.")
            return

        # Prepare meta-features DataFrame for SHAP explanation
        base_preds_for_meta = {}
        if not exp.trained_base:
            logger.warning(f"No base models found in experiment '{exp.name}'. Cannot generate base predictions for meta SHAP.")
            return

        for nm, (mdl, feat) in exp.trained_base.items():
            try:
                # Ensure features exist in the current test_df
                if all(f in df_test.columns for f in feat):
                    if hasattr(mdl, 'predict_proba'): # Check if model supports predict_proba
                        base_preds_for_meta[nm] = mdl.predict_proba(df_test[feat])[:, 1]
                    else:
                        logger.warning(f"Base model '{nm}' does not support predict_proba. Skipping for meta SHAP feature generation.")
                else:
                    logger.warning(f"Features for base model '{nm}' are missing in test data for SHAP. Skipping.")
            except Exception as e:
                logger.error(f"Error generating predictions for base model '{nm}' for SHAP: {e}")

        if not base_preds_for_meta:
            logger.warning(f"No base model predictions available for meta SHAP analysis in '{exp.name}'. Skipping.")
            return

        X_meta_shap = pd.DataFrame(base_preds_for_meta)

        # Add raw meta-features if configured and available
        if exp.config.get("include_meta_features", False):
            # As in evaluator, assume first meta-model config determines categories
            first_meta_model_config = next(iter(exp.config["meta_models"].values()), None)
            if first_meta_model_config:
                for cat in first_meta_model_config:
                    if cat in FEATURE_CATEGORIES:
                        raw_cols = FEATURE_CATEGORIES[cat]
                        missing_raw_cols = [c for c in raw_cols if c not in df_test.columns]
                        if missing_raw_cols:
                            logger.warning(f"Missing raw meta-features {missing_raw_cols} in test data for category '{cat}'. Skipping.")
                        else:
                            for col in raw_cols:
                                if col in X_meta_shap.columns:
                                    X_meta_shap[f"{col}_raw"] = df_test[col].values
                                else:
                                    X_meta_shap[col] = df_test[col].values
                    else:
                        logger.warning(f"Configured meta-feature category '{cat}' not found in FEATURE_CATEGORIES for SHAP.")
            else:
                logger.warning("No meta-model configurations found to determine raw meta-features for SHAP.")

        if X_meta_shap.empty:
            logger.warning(f"Meta-features DataFrame for SHAP analysis is empty for '{exp.name}'. Skipping SHAP plotting.")
            return

        # Generate SHAP for each meta combo
        for name, mdl in meta_models.items():
            try:
                # Some models might not be directly explainable by TreeExplainer (e.g., LogisticRegression)
                # You might need to use KernelExplainer for non-tree models, but it's much slower.
                # Here, we assume TreeExplainer compatibility for simplicity.
                # If model is not a tree ensemble, this might fail or require different explainer.
                if hasattr(mdl, 'best_estimator_'): # If GridSearchCV was used, get the best estimator
                    model_for_shap = mdl.best_estimator_
                else:
                    model_for_shap = mdl

                explainer = shap.TreeExplainer(model_for_shap)

                # Check if model type expects array or DataFrame
                # Some models might fit on arrays directly from GridSearchCV, losing column names
                if isinstance(X_meta_shap, pd.DataFrame) and hasattr(explainer.model, 'feature_names_in_') and explainer.model.feature_names_in_ is not None:
                     # Align features if possible, or convert X to numpy array for explainer.
                    # Simple check: if feature names mismatch, convert X to numpy.
                    if not list(X_meta_shap.columns) == list(explainer.model.feature_names_in_):
                        logger.warning(f"Meta model '{name}' feature names ({list(explainer.model.feature_names_in_)}) do not match SHAP input ({list(X_meta_shap.columns)}). Converting input to array.")
                        shap_vals = explainer.shap_values(X_meta_shap.values)
                        shap_input_data = X_meta_shap # Use DataFrame for plotting with original names
                    else:
                        shap_vals = explainer.shap_values(X_meta_shap)
                        shap_input_data = X_meta_shap
                else:
                    shap_vals = explainer.shap_values(X_meta_shap)
                    shap_input_data = X_meta_shap # Keep DataFrame for summary_plot labels

                # For binary classification, shap_values returns a list of two arrays.
                # We typically plot the SHAP values for the positive class (index 1).
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    shap_values_to_plot = shap_vals[1]
                else:
                    shap_values_to_plot = shap_vals # For regression or single output models

                plt.figure(figsize=(10, 6)) # Adjust figure size as needed
                shap.summary_plot(shap_values_to_plot, shap_input_data, show=False)
                plt.title(f"{exp.name} SHAP Summary — {name}")

                out_file = Path("results") / exp.name / f"shap_{name.replace(' ', '_')}.png"
                out_file.parent.mkdir(parents=True, exist_ok=True)

                plt.tight_layout()
                plt.savefig(out_file, dpi=300)
                plt.close()
                if verbose:
                    logger.info(f"[PLOT] Saved SHAP summary plot for meta model '{name}' → {out_file}")
            except Exception as e:
                logger.error(f"Error generating SHAP plot for meta model '{name}' in experiment '{exp.name}': {e}")
                # Specific error for TreeExplainer compatibility:
                if "XGBoost" not in str(type(model_for_shap)) and "LGBM" not in str(type(model_for_shap)) and "CatBoost" not in str(type(model_for_shap)) and "Tree" not in str(type(model_for_shap)):
                     logger.warning(f"Note: SHAP TreeExplainer may not be suitable for model type {type(model_for_shap)}. Consider KernelExplainer if issues persist.")

    except Exception as e:
        logger.error(f"An error occurred during SHAP plotting for experiment '{exp.name}': {e}")


# Example Usage (for demonstration)
if __name__ == "__main__":
    from pipeline.config_parser import load_config
    from pipeline.data_loader import prepare_experiments
    import os
    import shutil
    from sklearn.linear_model import LogisticRegression
    import joblib

    # Setup dummy environment
    test_config_path = "temp_plot_config.yaml"
    dummy_cleaned_dir = Path("cleaned")
    dummy_results_dir = Path("results")

    # Create dummy config
    dummy_config_content = """
datasets:
  - plot_test_data_A.csv
  - plot_test_data_B.csv
same_day: true
cross_day: false
same_day_cap: 1000
cross_day_cap: 1000
base_models:
  LogisticRegression:
    - flow_metrics
    - packet_size_stats
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
plot_heatmaps: true
plot_shap: true
save_metrics_json: true
"""
    with open(test_config_path, "w") as f:
        f.write(dummy_config_content)

    # Create dummy cleaned data files and results directory
    dummy_cleaned_dir.mkdir(exist_ok=True)
    # Define a subset of NEEDED_COLS for this dummy data to make it manageable
    demo_cols = ["Flow Duration", "SYN Flag Cnt", "Fwd IAT Mean", "Tot Fwd Pkts", "Fwd Pkt Len Max", "Label", "Attack Type"]
    for ds_name in ["plot_test_data_A.csv", "plot_test_data_B.csv"]:
        dummy_data_df = pd.DataFrame(np.random.rand(50, len(demo_cols)-2), columns=[c for c in demo_cols if c not in ["Label", "Attack Type"]])
        dummy_data_df["Label"] = np.random.randint(0, 2, 50)
        dummy_data_df["Attack Type"] = np.random.choice(["Attack", "Normal"], 50)
        dummy_data_df.to_csv(dummy_cleaned_dir / ds_name, index=False)

    logger.info("Dummy config and data created for plotter demo.")

    try:
        cfg = load_config(test_config_path)
        experiments = prepare_experiments(cfg)

        if experiments:
            exp_to_run = experiments[0] # Take the first experiment

            # Simulate base models being trained
            dummy_base_lr_model = LogisticRegression(random_state=42, solver='liblinear')
            dummy_rf_model = LogisticRegression(random_state=42, solver='liblinear') # Use LR for simplicity here too

            # Create dummy data for training (same as test for demo)
            df_for_training = load_and_clean(exp_to_run.test_file)
            if df_for_training.empty:
                logger.error("Dummy training data is empty, skipping demo.")
                raise ValueError("Empty dummy training data.")

            X_train = df_for_training.drop(columns=["Attack Type", "Label"], errors='ignore')
            y_train = df_for_training["Label"]

            # Ensure columns for feature categories exist in X_train
            flow_features = [f for f in FEATURE_CATEGORIES["flow_metrics"] if f in X_train.columns]
            packet_features = [f for f in FEATURE_CATEGORIES["packet_size_stats"] if f in X_train.columns]

            if not flow_features or not packet_features:
                logger.error("Not enough features in dummy data for base model training.")
                raise ValueError("Missing dummy features.")

            dummy_base_lr_model.fit(X_train[flow_features], y_train)
            dummy_rf_model.fit(X_train[packet_features], y_train)

            exp_to_run.trained_base = {
                "LogisticRegression_flow_metrics": (dummy_base_lr_model, flow_features),
                "LogisticRegression_packet_size_stats": (dummy_rf_model, packet_features) # using LR for both for simplicity
            }

            # Simulate meta model training
            meta_input_df = pd.DataFrame({
                "LogisticRegression_flow_metrics": dummy_base_lr_model.predict_proba(X_train[flow_features])[:, 1],
                "LogisticRegression_packet_size_stats": dummy_rf_model.predict_proba(X_train[packet_features])[:, 1]
            })

            # Add raw meta features for meta model training
            if cfg["include_meta_features"]:
                meta_raw_features = [f for f in FEATURE_CATEGORIES["meta_features"] if f in X_train.columns]
                for f in meta_raw_features:
                    if f in meta_input_df.columns:
                        meta_input_df[f"{f}_raw"] = X_train[f]
                    else:
                        meta_input_df[f] = X_train[f]

            dummy_meta_model = LogisticRegression(random_state=42, solver='liblinear')
            dummy_meta_model.fit(meta_input_df, y_train)
            exp_to_run.trained_meta = {"LogisticRegression_meta": dummy_meta_model}
            logger.info("Dummy models trained and assigned to experiment object.")


            # Simulate metrics.json file for heatmap plotting
            dummy_metrics = [
                {"experiment": exp_to_run.name, "data_file": "plot_test_data_A.csv", "model_type": "base", "combo": "LogisticRegression_flow_metrics", "f1": 0.75, "accuracy": 0.70},
                {"experiment": exp_to_run.name, "data_file": "plot_test_data_A.csv", "model_type": "base", "combo": "LogisticRegression_packet_size_stats", "f1": 0.68, "accuracy": 0.65},
                {"experiment": exp_to_run.name, "data_file": "plot_test_data_A.csv", "model_type": "meta", "combo": "LogisticRegression_meta", "f1": 0.82, "accuracy": 0.80},
                {"experiment": exp_to_run.name, "data_file": "plot_test_data_B.csv", "model_type": "base", "combo": "LogisticRegression_flow_metrics", "f1": 0.72, "accuracy": 0.68},
                {"experiment": exp_to_run.name, "data_file": "plot_test_data_B.csv", "model_type": "base", "combo": "LogisticRegression_packet_size_stats", "f1": 0.70, "accuracy": 0.66},
                {"experiment": exp_to_run.name, "data_file": "plot_test_data_B.csv", "model_type": "meta", "combo": "LogisticRegression_meta", "f1": 0.85, "accuracy": 0.83},
            ]
            exp_results_dir = dummy_results_dir / exp_to_run.name
            exp_results_dir.mkdir(parents=True, exist_ok=True)
            with open(exp_results_dir / "metrics.json", "w") as f:
                json.dump(dummy_metrics, f)
            logger.info("Dummy metrics.json created for heatmap demo.")

            # Run plotting functions
            print("\n--- Running Plotting Functions ---")
            plot_confusion_heatmaps(exp_to_run.name, cfg, verbose=True)
            plot_shap_summary(exp_to_run, verbose=True)

        else:
            logger.warning("No experiments to run for plotter demo.")

    except Exception as e:
        logger.error(f"Plotter demo failed: {e}")
    finally:
        # Clean up
        if Path(test_config_path).exists():
            Path(test_config_path).unlink()
        if dummy_cleaned_dir.exists():
            shutil.rmtree(dummy_cleaned_dir)
        if dummy_results_dir.exists():
            shutil.rmtree(dummy_results_dir)
        logger.info("Cleaned up dummy files and directories for plotter demo.")

