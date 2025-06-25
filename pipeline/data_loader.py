from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import os
import yaml

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

def load_config(path: str) -> Dict[str, Any]:
    """
    Simple wrapper to load YAML config.
    """
    with open(path) as f:
        return yaml.safe_load(f)

def prepare_experiments(cfg: Dict[str, Any]) -> List[Experiment]:
    """
    Based on same_day / cross_day toggles and dataset list,
    construct Experiment objects with appropriate train/test splits
    and per-file caps.
    """
    datasets = cfg["datasets"]
    cleaned_dir = Path("cleaned")
    experiments: List[Experiment] = []

    # SAME-DAY experiments
    if cfg["same_day"]:
        for ds in datasets:
            test_path = cleaned_dir / ds
            exp = Experiment(
                name=f"same_day_{ds}",
                train_files=[test_path],
                test_file=test_path,
                cap_train_per_file=cfg["same_day_cap"],
                cap_test_per_file=cfg["same_day_cap"],
                config=cfg
            )
            experiments.append(exp)

    # CROSS-DAY experiments (leave-one-out)
    if cfg["cross_day"]:
        for ds in datasets:
            test_path = cleaned_dir / ds
            train_paths = [
                cleaned_dir / train_ds
                for train_ds in datasets
                if train_ds != ds
            ]
            exp = Experiment(
                name=f"cross_day_leave_{ds}",
                train_files=train_paths,
                test_file=test_path,
                cap_train_per_file=cfg["cross_day_cap"],
                cap_test_per_file=cfg["cross_day_cap"],
                config=cfg
            )
            experiments.append(exp)

    # Verbose listing
    if cfg.get("verbose", False):
        print(f"\nPrepared {len(experiments)} experiments:")
        for e in experiments:
            mode = "SAME" if e.name.startswith("same_day") else "CROSS"
            print(f"  [{mode}] {e.name}: train={len(e.train_files)} file(s), test={e.test_file.name}, "
                  f"cap_train={e.cap_train_per_file}, cap_test={e.cap_test_per_file}")

    return experiments

# Demo load and prepare if run directly
if __name__ == "__main__":
    import pprint
    cfg = load_config("exp_config.yaml")
    exps = prepare_experiments(cfg)
    pprint.pprint(exps)
