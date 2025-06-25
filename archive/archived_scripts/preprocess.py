import pandas as pd
import os

# Replace with your actual downloaded CSV path
input_path = "CIC-IDS-2018/03-02-2018.csv"
output_path = "datasets/03-02-2018-processed.csv"

# Load and clean
df = pd.read_csv(input_path, low_memory=False)

# Relabel multi-class labels to binary
df["Label"] = df["Label"].apply(lambda x: 0 if "Benign" in str(x) else 1)

# Create protocol flags for TCP (6) and UDP (17)
df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce")
df["Protocol_6"] = (df["Protocol"] == 6).astype(int)
df["Protocol_17"] = (df["Protocol"] == 17).astype(int)

# Save cleaned dataset
core_features = [
    'Flow Duration', 'Protocol_6', 'Protocol_17', 'Pkt Len Min', 'Pkt Len Max',
    'Tot Fwd Pkts', 'TotLen Fwd Pkts', 'Flow IAT Mean', 'Fwd IAT Std', 'Bwd Pkt Len Mean',
    'SYN Flag Cnt', 'ACK Flag Cnt', 'Fwd Pkts/s', 'Bwd Pkts/s', 'RST Flag Cnt',
    'Active Mean', 'Idle Mean', 'Down/Up Ratio',
    'Dst Port'
]

# Add Label
core_features.append("Label")

# Only keep those features
df = df[[col for col in core_features if col in df.columns]]

# Convert to numeric, coerce junk to NaN
df = df.apply(pd.to_numeric, errors="coerce")

# Drop rows with NaNs
df = df.dropna()

# Save cleaned dataset
df.to_csv(output_path, index=False)
print(f"Saved preprocessed dataset to {output_path}")
print(df.dtypes)
print(df["Label"].value_counts())