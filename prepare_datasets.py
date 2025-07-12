import logging
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIGURE LOGGING -----
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('PrepareDatasets')

# --- Processing Mode Flags -----
PROCESS_FIRST_ONLY = False  # If True, only process the first file alphabetically
SKIP_FILES = ['02-20-2018.csv', '02-28-2018.csv', '03-01-2018.csv']

# --- Directories -----
RAW_DIR = Path("raw/CIC-IDS2018")      # Input raw CIC-IDS-2018 CSVs
CLEAN_DIR = Path("cleaned/CIC-IDS2018") # Output cleaned CSVs
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature Categories -----
FEATURE_CATEGORIES = {
    "flow_metrics": ["Flow Duration", "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"],
    "packet_size_stats": ["Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg"],
    "timing_iat": ["Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min"],
    "flags_and_protocol": ["FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Protocol", "Dst Port", "Init Fwd Win Byts", "Init Bwd Win Byts"],
    "rates_and_ratios": ["Down/Up Ratio", "Fwd Pkts/s", "Bwd Pkts/s", "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg"],
    "connection_activity": ["Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts", "Fwd Act Data Pkts", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
}

ALL_FEATURES = [feat for feats in FEATURE_CATEGORIES.values() for feat in feats]

def main():
    all_files = sorted(RAW_DIR.glob('*.csv'))
    if not all_files:
        logger.error(f"No CSV files found in {RAW_DIR}")
        return

    # Apply skip list
    files = [f for f in all_files if f.name not in SKIP_FILES]
    if PROCESS_FIRST_ONLY:
        files = files[:1]

    logger.info(f"Processing {len(files)} file(s): {[f.name for f in files]}")
    for csv_path in files:
        logger.info(f"Processing {csv_path.name}")
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            logger.error(f"Failed to read {csv_path.name}: {e}")
            continue

        # Rename and encode labels
        df.rename(columns={'Label': 'Attack Type'}, inplace=True)
        df['Label'] = df['Attack Type'].apply(
            lambda x: 0 if str(x).lower().startswith('benign') else 1
        )

        # Select features present in this file
        selected_feats = [c for c in ALL_FEATURES if c in df.columns]
        cols = selected_feats + ['Attack Type', 'Label']
        df_clean = df[cols].copy()

        # Coerce to numeric and drop invalid rows
        df_clean[selected_feats + ['Label']] = df_clean[selected_feats + ['Label']].apply(
            pd.to_numeric, errors='coerce'
        )
        df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        before = len(df_clean)
        df_clean.dropna(inplace=True)
        df_clean.reset_index(drop=True, inplace=True)
        after = len(df_clean)

        # Save cleaned CSV
        out_path = CLEAN_DIR / csv_path.name
        try:
            df_clean.to_csv(out_path, index=False)
            logger.info(
                f"Saved {out_path.name}: features={len(selected_feats)}, "
                f"rows before={before}, after={after}"
            )
        except Exception as e:
            logger.error(f"Failed to save cleaned CSV {out_path.name}: {e}")

    logger.info("Dataset preparation complete.")

if __name__ == '__main__':
    main()
