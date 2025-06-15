# main.py

from data_loader import get_train_test_data
from models import get_model_configs
from trainer import run_base_models
from meta_model import train_meta_model
from unsupervised import run_isolation_forest, run_one_class_svm
from config import results_dir

import os

def main():
    print("Starting Meta-Model Experiment Pipeline")

    # Make sure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Load data
    print("Loading datasets...")
    X_train_dict, X_test_dict, y_train, y_test, train_df, test_df, scale_pos_weight, class_weights = get_train_test_data()

    """
    # Step 2: Load base model configs
    print("Setting up base models...")
    model_configs = get_model_configs(scale_pos_weight, class_weights)

    # Step 3: Train base models and get predictions
    print("Training base models...")
    pred_train_dict, pred_test_dict = run_base_models(model_configs, X_train_dict, X_test_dict, y_train, y_test)

    # Step 4: Train meta-model using base predictions
    print("Training meta-model...")
    _ = train_meta_model(pred_train_dict, pred_test_dict, y_train, y_test, train_df, test_df, class_weights)
    """
    
    print("Running Isolation Forest baseline...")
    run_isolation_forest(X_train_dict, X_test_dict, y_train, y_test)

    print("Running One-Class SVM baseline...")
    run_one_class_svm(X_train_dict, X_test_dict, y_train, y_test)

    print("All done! Results saved to:", results_dir)

if __name__ == "__main__":
    main()
