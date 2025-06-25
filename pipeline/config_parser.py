import yaml
import os

def load_config(path):
    """
    Load and validate experiment configuration from a YAML file.
    Returns config dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Required keys
    required = [
        "datasets", "same_day", "cross_day",
        "same_day_cap", "cross_day_cap",
        "base_models", "feature_grouping",
        "grid_search_base", "grid_search_meta",
        "use_meta", "meta_models",
        "smote", "class_weighting", "include_meta_features",
        "n_jobs", "verbose",
        "plot_heatmaps", "plot_shap", "save_metrics_json"
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    # Type validations
    if not isinstance(cfg["datasets"], list) or not cfg["datasets"]:
        raise ValueError("`datasets` must be a non-empty list of dataset filenames.")
    if not cfg["same_day"] and not cfg["cross_day"]:
        raise ValueError("At least one of `same_day` or `cross_day` must be true.")
    # Validate model mappings
    if not isinstance(cfg["base_models"], dict) or not cfg["base_models"]:
        raise ValueError("`base_models` must be a non-empty dict of model → feature sets.")
    if cfg["use_meta"] and (not isinstance(cfg["meta_models"], dict) or not cfg["meta_models"]):
        raise ValueError("`meta_models` must be provided when `use_meta` is true.")

    # n_jobs
    if not isinstance(cfg["n_jobs"], int) or cfg["n_jobs"] < 1:
        raise ValueError("`n_jobs` must be a positive integer.")

    return cfg

# For demonstration, load and display
if __name__ == "__main__":
    import pprint
    config = load_config("exp_config.yaml")
    pprint.pprint(config)
