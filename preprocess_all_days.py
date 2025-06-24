# preprocess_all_days.py

# Cleans raw CIC-IDS-2018 CSVs by selecting a curated set of features,
# renaming the multiclass 'Label' to 'Attack Type', creating a binary 'Label',
# enforcing numeric types, and writing cleaned files.
# Toggle processing mode at the top of the script.

import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration: Toggle Processing Mode ---
# Set PROCESS_FIRST_ONLY to True to process only the first file (alphabetical).
# Otherwise, all files in RAW_DIR will be processed.
PROCESS_FIRST_ONLY = False

# Optionally skip specific filenames
SKIP_FILES = ['02-20-2018.csv']  # e.g., ['02-15-2018.csv']

# Directories
RAW_DIR = Path("CIC-IDS-2018")    # Folder containing original CIC-IDS-2018 .csv files
CLEAN_DIR = Path("cleaned")        # Output: cleaned CSVs
CLEAN_DIR.mkdir(exist_ok=True)

# Manually categorized CICFlowMeter-V3 features
FEATURE_CATEGORIES = {
    "flow_metrics": [
        "Flow Duration", "Flow Byts/s", "Flow Pkts/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"
    ],
    "packet_size_stats": [
        "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
        "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
        "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
        "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
        "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg"
    ],
    "timing_iat": [
        "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min"
    ],
    "flags_and_protocol": [
        "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
        "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
        "Protocol", "Dst Port", "Init Fwd Win Byts", "Init Bwd Win Byts"
    ],
    "rates_and_ratios": [
        "Down/Up Ratio", "Fwd Pkts/s", "Bwd Pkts/s",
        "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
        "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg"
    ],
    "connection_activity": [
        "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
        "Fwd Act Data Pkts", "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
    ]
}

# Flatten all feature lists
ALL_FEATURES = [feat for feats in FEATURE_CATEGORIES.values() for feat in feats]


def main():
    # Discover raw CSV files
    all_files = sorted(RAW_DIR.glob("*.csv"))
    if not all_files:
        print(f"No CSV files found in {RAW_DIR}")
        return

    # Skip files if configured
    if SKIP_FILES:
        all_files = [f for f in all_files if f.name not in SKIP_FILES]

    # Optionally process only the first file
    files_to_process = all_files[:1] if PROCESS_FIRST_ONLY else all_files

    for csv_path in files_to_process:
        print(f"Processing {csv_path.name}...")
        df = pd.read_csv(csv_path, low_memory=False)

        # Rename multiclass label to Attack Type
        df.rename(columns={'Label': 'Attack Type'}, inplace=True)
        # Create binary Label column
        df['Label'] = df['Attack Type'].apply(lambda x: 0 if str(x).lower().startswith('benign') else 1)

        # Select curated features present in this file
        selected_feats = [c for c in ALL_FEATURES if c in df.columns]
        selected_cols = selected_feats + ['Attack Type', 'Label']

        # Coerce to numeric (except Attack Type)
        df_clean = df[selected_cols].copy()
        df_clean.loc[:, selected_feats + ['Label']] = (
            df_clean.loc[:, selected_feats + ['Label']]
            .apply(pd.to_numeric, errors='coerce')
        )

        # Replace infinities with NaN, then drop all rows with NaN in any column
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean.dropna(inplace=True)
        df_clean.reset_index(drop=True, inplace=True)

        # Save cleaned CSV
        out_path = CLEAN_DIR / csv_path.name
        df_clean.to_csv(out_path, index=False)
        print(f" -> Cleaned file: {out_path.name} | Features: {len(selected_feats)} | Rows: {df_clean.shape[0]}")

    print(f"Finished processing {len(files_to_process)} file(s). Cleaned CSVs in '{CLEAN_DIR}' directory.")

if __name__ == '__main__':
    main()
