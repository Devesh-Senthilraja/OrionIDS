
import sys
from pipeline.config_parser import load_config
from pipeline.data_loader   import prepare_experiments
from pipeline.trainer       import run_training
from pipeline.evaluator     import run_evaluation
from pipeline.plotter       import plot_confusion_heatmaps, plot_shap_summary

def main(config_path="exp_config.yaml"):
    # 1. Load and validate config
    cfg = load_config(config_path)

    # 2. Prepare experiments (each yields train/test split, caps, toggles)
    experiments = prepare_experiments(cfg)

    # 3. Loop through each experiment
    for exp in experiments:
        if cfg["verbose"]:
            print(f"\n=== Running Experiment: {exp.name} ===")

        # 3a. Train base and optional meta models
        run_training(exp, verbose=cfg["verbose"], n_jobs=cfg["n_jobs"])

        # 3b. Evaluate on hold-out test set
        run_evaluation(
            exp,
            save_json=cfg["save_metrics_json"],
            verbose=cfg["verbose"]
        )

        # 3c. Generate plots if requested
        if cfg["plot_heatmaps"] or cfg["plot_shap"]:
            run_plots(
                exp,
                plot_heatmaps=cfg["plot_heatmaps"],
                plot_shap=cfg["plot_shap"],
                verbose=cfg["verbose"]
            )

    print("\nAll experiments complete.")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "exp_config.yaml"
    main(config_file)
