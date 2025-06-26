from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Union, Set
import pandas as pd
import numpy as np
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Assuming FEATURE_CATEGORIES and NEEDED_COLS are defined globally or imported from a central module
# For this example, I will define them here, mimicking their presence in trainer.py
# In a real pipeline, these would likely be in a shared 'constants.py' or 'features.py'
FEATURE_CATEGORIES = {
    "flow_metrics":      ["Flow Duration","Flow Byts/s","Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min"],
    "packet_size_stats": ["Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min",
                          "Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std",
                          "Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","Pkt Size Avg","Fwd Seg Size Avg",
                          "Bwd Seg Size Avg"],
    "timing_iat":        ["Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot","Bwd IAT Mean",
                          "Bwd IAT Std","Bwd IAT Max","Bwd IAT Min"],
    "flags_and_protocol":["FIN Flag Cnt","SYN Flag Cnt","RST Flag Cnt","PSH Flag Cnt","ACK Flag Cnt","URG Flag Cnt","CWE Flag Count",
                          "ECE Flag Cnt","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Protocol","Dst Port",
                          "Init Fwd Win Byts","Init Bwd Win Byts"],
    "rates_and_ratios":  ["Down/Up Ratio","Fwd Pkts/s","Bwd Pkts/s","Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg",
                          "Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg"],
    "connection_activity":["Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts","Fwd Act Data Pkts",
                           "Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"],
    "meta_features":     ["Flow Duration","SYN Flag Cnt","Fwd IAT Mean"] # Example meta-features
}

NEEDED_COLS: Set[str] = set().union(*FEATURE_CATEGORIES.values()) | {"Attack Type", "Label"}


@dataclass
class Experiment:
    """
    Holds all information for a single train/test experiment.
    """
    name: str
    train_files: List[Path]
    test_file: Path
    cap_train_per_file: int
    cap_test_per_file: int
    config: Dict[str, Any]
    # These fields will be populated during the pipeline run
    trained_base: Dict[str, Any] = field(default_factory=dict)
    trained_meta: Dict[str, Any] = field(default_factory=dict)


def load_and_clean(filepath: Path) -> pd.DataFrame:
    """
    Reads a CSV file, selects only necessary columns, handles infinite/NaN values,
    and downcasts numeric types to reduce memory usage.

    Args:
        filepath (Path): The path to the CSV file to load.

    Returns:
        pd.DataFrame: A cleaned and optimized Pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        KeyError: If essential columns (from NEEDED_COLS) are missing in the file.
        Exception: For other unexpected errors during file loading or processing.
    """
    if not filepath.exists():
        logger.error(f"Data file not found: {filepath}")
        raise FileNotFoundError(f"Data file not found at '{filepath}'")

    try:
        # Check if all needed columns exist in the file without loading everything
        # This is an optimization. `read_csv` with `usecols` will raise if a col is missing.
        # A more robust check might involve reading only header first.
        # For simplicity, we rely on `read_csv`'s behavior for now.
        df = pd.read_csv(filepath, usecols=list(NEEDED_COLS), low_memory=False)
        logger.debug(f"Loaded {len(df)} rows from {filepath.name}.")
    except pd.errors.EmptyDataError:
        logger.warning(f"File '{filepath.name}' is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=list(NEEDED_COLS)) # Return empty DF with expected columns
    except KeyError as e:
        logger.error(f"Missing one or more required columns in {filepath.name}: {e}")
        raise KeyError(f"Missing required column(s) in '{filepath.name}': {e}")
    except Exception as e:
        logger.error(f"Failed to load or read '{filepath.name}': {e}")
        raise RuntimeError(f"Error loading data from '{filepath.name}': {e}")

    # Replace infinities with NaN and then drop rows with any NaN
    original_rows = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    if len(df) < original_rows:
        logger.warning(f"Dropped {original_rows - len(df)} rows with NaN/inf values from {filepath.name}.")

    # Downcast numerics to reduce memory footprint
    for c in df.select_dtypes(include=["float64", "int64"]).columns:
        try:
            df[c] = pd.to_numeric(df[c], downcast="float")
        except Exception as e:
            logger.warning(f"Could not downcast column '{c}' in {filepath.name}: {e}")
            # Keep original type if downcasting fails

    logger.debug(f"Cleaned and processed {len(df)} rows from {filepath.name}.")
    return df


def prepare_experiments(cfg: Dict[str, Any]) -> List[Experiment]:
    """
    Based on same_day / cross_day toggles and dataset list from the config,
    constructs Experiment objects with appropriate train/test splits and per-file caps.

    Args:
        cfg (Dict[str, Any]): The configuration dictionary.

    Returns:
        List[Experiment]: A list of Experiment objects, each representing a distinct
                          train/test scenario.

    Raises:
        ValueError: If 'datasets' list is empty or if no experiments can be prepared
                    based on the config.
    """
    datasets = cfg.get("datasets")
    if not datasets:
        logger.error("Configuration 'datasets' list is empty or not found.")
        raise ValueError("Cannot prepare experiments: 'datasets' list is empty.")

    cleaned_dir = Path("cleaned")
    experiments: List[Experiment] = []

    # SAME-DAY experiments
    if cfg["same_day"]:
        for ds in datasets:
            test_path = cleaned_dir / ds
            # Check if file exists, although actual loading is done later in trainer/evaluator
            if not test_path.exists():
                logger.warning(f"Same-day dataset '{test_path}' not found. Skipping experiment.")
                continue
            exp = Experiment(
                name=f"same_day_{ds.replace('.csv', '')}",
                train_files=[test_path],
                test_file=test_path,
                cap_train_per_file=cfg["same_day_cap"],
                cap_test_per_file=cfg["same_day_cap"],
                config=cfg
            )
            experiments.append(exp)
        logger.info(f"Prepared {len([e for e in experiments if 'same_day' in e.name])} same-day experiments.")

    # CROSS-DAY experiments (leave-one-out)
    if cfg["cross_day"]:
        for ds in datasets:
            test_path = cleaned_dir / ds
            if not test_path.exists():
                logger.warning(f"Cross-day test dataset '{test_path}' not found. Skipping experiments using it as test.")
                continue

            train_paths = [
                cleaned_dir / train_ds
                for train_ds in datasets
                if train_ds != ds
            ]

            # Filter out non-existent training files for cross-day experiments
            existing_train_paths = [p for p in train_paths if p.exists()]
            if not existing_train_paths:
                logger.warning(f"No valid training files found for cross-day experiment leaving out '{ds}'. Skipping.")
                continue

            exp = Experiment(
                name=f"cross_day_leave_{ds.replace('.csv', '')}",
                train_files=existing_train_paths,
                test_file=test_path,
                cap_train_per_file=cfg["cross_day_cap"],
                cap_test_per_file=cfg["cross_day_cap"],
                config=cfg
            )
            experiments.append(exp)
        logger.info(f"Prepared {len([e for e in experiments if 'cross_day' in e.name])} cross-day experiments.")

    if not experiments:
        logger.error("No experiments could be prepared. Check 'datasets' list and 'same_day'/'cross_day' toggles in config.")
        raise ValueError("No experiments prepared. Ensure valid config and existing data files.")

    # Verbose listing
    if cfg.get("verbose", False):
        logger.info(f"\nPrepared {len(experiments)} experiments:")
        for e in experiments:
            mode = "SAME" if e.name.startswith("same_day") else "CROSS"
            train_files_names = [f.name for f in e.train_files]
            logger.info(f"  [{mode}] {e.name}: train_files={train_files_names}, test_file={e.test_file.name}, "
                        f"cap_train={e.cap_train_per_file}, cap_test={e.cap_test_per_file}")

    return experiments

# Demo load and prepare if run directly
if __name__ == "__main__":
    import pprint
    # Create a dummy config file and dummy cleaned files for testing
    from pipeline.config_parser import load_config

    config_file_path = "temp_exp_config.yaml"
    dummy_config_content = """
datasets:
  - dummy_dataset_A.csv
  - dummy_dataset_B.csv
  - dummy_dataset_C.csv
same_day: true
cross_day: true
same_day_cap: 10000
cross_day_cap: 5000
base_models: {} # Not used in data_loader demo
feature_grouping: false
grid_search_base: false
grid_search_meta: false
use_meta: false
meta_models: {} # Not used in data_loader demo
smote: false
class_weighting: false
include_meta_features: false
n_jobs: 1
verbose: true
plot_heatmaps: false
plot_shap: false
save_metrics_json: false
"""
    Path("cleaned").mkdir(exist_ok=True)
    with open(config_file_path, "w") as f:
        f.write(dummy_config_content)

    # Create dummy data files
    dummy_df_content = pd.DataFrame(np.random.rand(100, len(NEEDED_COLS)), columns=list(NEEDED_COLS))
    dummy_df_content["Label"] = np.random.randint(0, 2, 100)
    dummy_df_content["Attack Type"] = np.random.choice(["Attack", "Normal"], 100)

    for ds_name in ["dummy_dataset_A.csv", "dummy_dataset_B.csv", "dummy_dataset_C.csv"]:
        dummy_df_content.to_csv(Path("cleaned") / ds_name, index=False)
    logger.info("Created dummy config and data files for demo.")

    try:
        cfg = load_config(config_file_path)
        exps = prepare_experiments(cfg)
        print("\n--- Prepared Experiments ---")
        pprint.pprint(exps)

        # Demo load_and_clean
        print("\n--- Testing load_and_clean ---")
        test_file = Path("cleaned") / "dummy_dataset_A.csv"
        df_cleaned = load_and_clean(test_file)
        print(f"Loaded and cleaned DataFrame shape: {df_cleaned.shape}")
        print(f"Columns: {df_cleaned.columns.tolist()}")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
    finally:
        # Clean up dummy files
        Path(config_file_path).unlink(missing_ok=True)
        for ds_name in ["dummy_dataset_A.csv", "dummy_dataset_B.csv", "dummy_dataset_C.csv"]:
            (Path("cleaned") / ds_name).unlink(missing_ok=True)
        Path("cleaned").rmdir()
        logger.info("Cleaned up dummy files and directory.")

