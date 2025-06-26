import yaml
import os
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
# For standalone execution or simple demonstration:
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_config(path: str) -> dict:
    """
    Load and validate experiment configuration from a YAML file.

    Args:
        path (str): The file path to the YAML configuration.

    Returns:
        dict: The loaded and validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is not valid YAML.
        KeyError: If any required configuration key is missing.
        ValueError: If any configuration value is invalid or malformed.
    """
    if not os.path.exists(path):
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(f"Config file not found at '{path}'")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from '{path}'")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file '{path}': {e}")
        raise yaml.YAMLError(f"Invalid YAML format in '{path}': {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while opening or reading '{path}': {e}")
        raise IOError(f"Could not read config file '{path}': {e}")

    if not isinstance(cfg, dict):
        logger.error(f"Config file content is not a dictionary. Got: {type(cfg)}")
        raise ValueError("Config file content must be a dictionary.")

    # Required keys
    required_keys = [
        "datasets", "same_day", "cross_day",
        "same_day_cap", "cross_day_cap",
        "base_models", "feature_grouping",
        "grid_search_base", "grid_search_meta",
        "use_meta", "meta_models",
        "smote", "class_weighting", "include_meta_features",
        "n_jobs", "verbose",
        "plot_heatmaps", "plot_shap", "save_metrics_json"
    ]
    missing_keys = [k for k in required_keys if k not in cfg]
    if missing_keys:
        logger.error(f"Missing required configuration keys: {missing_keys}")
        raise KeyError(f"Missing required config keys: {missing_keys}")

    logger.debug("All required configuration keys are present.")

    # Type and value validations
    if not isinstance(cfg["datasets"], list) or not cfg["datasets"]:
        logger.error("`datasets` must be a non-empty list of dataset filenames.")
        raise ValueError("`datasets` must be a non-empty list of dataset filenames.")
    if not isinstance(cfg["same_day"], bool) or not isinstance(cfg["cross_day"], bool):
        logger.error("`same_day` and `cross_day` must be boolean values.")
        raise ValueError("`same_day` and `cross_day` must be boolean values.")
    if not cfg["same_day"] and not cfg["cross_day"]:
        logger.error("At least one of `same_day` or `cross_day` must be true.")
        raise ValueError("At least one of `same_day` or `cross_day` must be true.")

    if not isinstance(cfg["same_day_cap"], int) or cfg["same_day_cap"] < 1:
        logger.error(f"`same_day_cap` must be a positive integer. Got: {cfg['same_day_cap']}")
        raise ValueError("`same_day_cap` must be a positive integer.")
    if not isinstance(cfg["cross_day_cap"], int) or cfg["cross_day_cap"] < 1:
        logger.error(f"`cross_day_cap` must be a positive integer. Got: {cfg['cross_day_cap']}")
        raise ValueError("`cross_day_cap` must be a positive integer.")

    # Validate base_models mapping
    if not isinstance(cfg["base_models"], dict) or not cfg["base_models"]:
        logger.error("`base_models` must be a non-empty dictionary of model → feature sets.")
        raise ValueError("`base_models` must be a non-empty dict of model → feature sets.")

    # Validate meta_models if use_meta is true
    if cfg["use_meta"]:
        if not isinstance(cfg["meta_models"], dict) or not cfg["meta_models"]:
            logger.error("`meta_models` must be provided and be a non-empty dictionary when `use_meta` is true.")
            raise ValueError("`meta_models` must be provided when `use_meta` is true.")
    elif "meta_models" in cfg and cfg["meta_models"]:
        logger.warning("`meta_models` is provided but `use_meta` is false. Meta models will not be used.")

    if not isinstance(cfg["feature_grouping"], bool):
        logger.error("`feature_grouping` must be a boolean.")
        raise ValueError("`feature_grouping` must be a boolean.")
    if not isinstance(cfg["grid_search_base"], bool) or not isinstance(cfg["grid_search_meta"], bool):
        logger.error("`grid_search_base` and `grid_search_meta` must be booleans.")
        raise ValueError("`grid_search_base` and `grid_search_meta` must be booleans.")
    if not isinstance(cfg["smote"], bool) or not isinstance(cfg["class_weighting"], bool):
        logger.error("`smote` and `class_weighting` must be booleans.")
        raise ValueError("`smote` and `class_weighting` must be booleans.")
    if cfg["smote"] and cfg["class_weighting"]:
        logger.warning("Both `smote` and `class_weighting` are enabled. SMOTE will take precedence.")

    if not isinstance(cfg["include_meta_features"], bool):
        logger.error("`include_meta_features` must be a boolean.")
        raise ValueError("`include_meta_features` must be a boolean.")

    # n_jobs
    if not isinstance(cfg["n_jobs"], int) or cfg["n_jobs"] < -1 or cfg["n_jobs"] == 0:
        logger.error(f"`n_jobs` must be a positive integer or -1. Got: {cfg['n_jobs']}")
        raise ValueError("`n_jobs` must be a positive integer or -1 (for all CPUs).")

    if not isinstance(cfg["verbose"], bool):
        logger.error("`verbose` must be a boolean.")
        raise ValueError("`verbose` must be a boolean.")

    if not isinstance(cfg["plot_heatmaps"], bool) or not isinstance(cfg["plot_shap"], bool) or not isinstance(cfg["save_metrics_json"], bool):
        logger.error("`plot_heatmaps`, `plot_shap`, and `save_metrics_json` must be booleans.")
        raise ValueError("`plot_heatmaps`, `plot_shap`, and `save_metrics_json` must be booleans.")

    logger.info("Configuration validated successfully.")
    return cfg

# For demonstration, load and display
if __name__ == "__main__":
    import pprint
    # Create a dummy config file for testing
    dummy_config_content = """
datasets:
  - dataset1.csv
  - dataset2.csv
same_day: true
cross_day: true
same_day_cap: 100000
cross_day_cap: 50000
base_models:
  LogisticRegression:
    - flow_metrics
    - packet_size_stats
  RandomForest:
    - all_features # This will trigger a KeyError for 'all_features' if not in FEATURE_CATEGORIES
feature_grouping: true
grid_search_base: true
grid_search_meta: true
use_meta: true
meta_models:
  LogisticRegression:
    - meta_features
smote: false
class_weighting: true
include_meta_features: true
n_jobs: 1
verbose: true
plot_heatmaps: true
plot_shap: true
save_metrics_json: true
"""
    config_file_path = "temp_exp_config.yaml"
    with open(config_file_path, "w") as f:
        f.write(dummy_config_content)

    logger.info(f"Created a dummy config file at '{config_file_path}' for demonstration.")

    try:
        config = load_config(config_file_path)
        print("\n--- Loaded and Validated Configuration ---")
        pprint.pprint(config)
    except (FileNotFoundError, yaml.YAMLError, KeyError, ValueError) as e:
        print(f"\n--- Configuration Loading Failed ---")
        print(e)
    finally:
        # Clean up the dummy config file
        if os.path.exists(config_file_path):
            os.remove(config_file_path)
            logger.info(f"Removed dummy config file at '{config_file_path}'.")

    # Example of a missing file error
    print("\n--- Testing FileNotFoundError ---")
    try:
        load_config("non_existent_config.yaml")
    except FileNotFoundError as e:
        print(e)

    # Example of invalid YAML (if you uncomment and run, it will fail)
    # print("\n--- Testing Invalid YAML ---")
    # with open(config_file_path, "w") as f:
    #     f.write("key: - [") # Invalid YAML
    # try:
    #     load_config(config_file_path)
    # except yaml.YAMLError as e:
    #     print(e)
    # finally:
    #     if os.path.exists(config_file_path):
    #         os.remove(config_file_path)

