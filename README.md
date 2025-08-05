# ORIONIDS

ORIONIDS is a lightweight framework for running intrusion detection experiments on the CIC-IDS-2018 dataset using base and meta-model configurations.

## Setup

### 1) Clone the repository
```bash
git clone https://github.com/<your-username>/ORIONIDS.git
cd ORIONIDS
```

### 2) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# OR
.\.venv\Scriptsctivate    # Windows
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Data

This project uses the **CIC-IDS-2018** dataset. Download the raw CSVs from Kaggle and place them under `data/raw/2018/`:

- Kaggle: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv?resource=download

Expected structure after download:
```
ORIONIDS/
  data/
    raw/
      2018/
        <CSV files from Kaggle>
  results/
  scripts/
  visuals/
```

> Note: Large datasets and generated results are not tracked in Git. Only directory placeholders are stored.

## Running experiments

All experiment entry points are in `scripts/experiments/`. Each script will create output subfolders under `results/` if needed.

Examples:
```bash
# Base models
python scripts/experiments/base_model_study.py

# Meta model
python scripts/experiments/meta_model_study.py

# Ablation
python scripts/experiments/meta_model_ablation_study.py

# Feature-set study
python scripts/experiments/meta_model_feature_set_study.py
```

## Paths

Scripts resolve paths relative to the repository root. If you want to override defaults, most scripts accept CLI flags and/or respect environment variables (e.g., `--results_dir` or `RESULTS_DIR`).

## Reproducibility

- Python version: see `requirements.txt`
- Random seeds and other options can be set via script arguments where applicable.
- Ensure the dataset directory structure matches the paths above.
