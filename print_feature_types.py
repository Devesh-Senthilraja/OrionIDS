import pandas as pd
from pathlib import Path

# Directory containing cleaned CSVs
CLEAN_DIR = Path("cleaned")

# Iterate over all cleaned dataset files
for csv_file in sorted(CLEAN_DIR.glob("*.csv")):
    df = pd.read_csv(csv_file)
    print(f"\nFeature types in {csv_file.name}:")
    # Print column name and its dtype
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
