# config.py

# --- Dataset Configuration ---
#train_dataset_path = "Datasets/processed_thursday_dataset.csv"
train_dataset_paths = [
    "Datasets/processed_wednesday_dataset.csv",
    "Datasets/processed_thursday_dataset.csv",
    "Datasets/processed_friday_dataset.csv"
]
test_dataset_path = "Datasets/processed_wednesday_dataset_.csv"  # Can be same as train for within-day test

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
use_cross_validation = False  # For base models
use_cv_meta_model = True      # Use cross_val_predict() for meta-model
add_meta_features = True      # Whether to add raw packet features to meta-model

use_smote = True            # Enable oversampling
use_class_weights = False   # Enable manual weighting 

# --- Model Settings ---
random_seed = 42
test_size = 0.3
n_cv_splits = 5  # Number of CV folds

# --- Output Paths ---
results_dir = "Results/"
cache_dir = "Cached_Models/"
plots_dir = f"{results_dir}/plots/"
metrics_log_path = f"{results_dir}/metrics_log.json"
