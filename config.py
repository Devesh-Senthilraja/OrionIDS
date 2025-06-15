# config.py

# --- Dataset Configuration ---
#train_dataset_path = "Datasets/processed_thursday_dataset.csv"
train_dataset_paths = [
    "datasets/02-14-2018-processed.csv",
    "datasets/02-15-2018-processed.csv",
    "datasets/02-16-2018-processed.csv"
]
test_dataset_path = "datasets/02-21-2018-processed.csv"  # Can be same as train for within-day test

# --- Feature Groups ---
general_features = ['Flow Duration', 'Protocol_6', 'Protocol_17', 'Pkt Len Min', 'Pkt Len Max']
statistical_features = ['Tot Fwd Pkts', 'TotLen Fwd Pkts', 'Flow IAT Mean', 'Fwd IAT Std', 'Bwd Pkt Len Mean']
behavioral_features = ['SYN Flag Cnt', 'ACK Flag Cnt', 'Fwd Pkts/s', 'Bwd Pkts/s', 'RST Flag Cnt', 'Active Mean', 'Idle Mean', 'Down/Up Ratio']
meta_features = ['Dst Port', 'Pkt Len Min', 'Pkt Len Max']  # These are packet-level features used in meta-model

# Toggle which feature groups to use
enabled_feature_sets = {
    "General": True,
    "Statistical": True,
    "Behavioral": True
}

# --- Model Toggles ---
disable_cache = True  # Set to True to force retraining
use_grid_search = False
use_cv_base_model = False  # For base models, unused
use_cv_meta_model = True      # Use cross_val_predict() for meta-model, unused
add_meta_features = True      # Whether to add raw packet features to meta-model

use_class_weights = False   # Enable manual weighting 
use_smote = False  # Enable SMOTE

use_shap = False #Enable SHAP

# --- Model Settings ---
random_seed = 42
test_size = 0.3
n_cv_splits = 5  # Number of CV folds
n_jobs = -1 # Use all CPU cores

# --- Output Paths ---
results_dir = "results/"
cache_dir = "cached_models/"
plots_dir = f"{results_dir}/plots/"
metrics_log_path = f"{results_dir}/metrics_log.json"
shap_plot_path = f"{plots_dir}/meta_model_shap.png"
