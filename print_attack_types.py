import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("cleaned")

attack_types = set()
for csv_file in sorted(CLEAN_DIR.glob("*.csv")):
    df = pd.read_csv(csv_file)
    attack_types.update(df["Attack Type"].unique())

print("Attack types across all days:")
for at in sorted(attack_types):
    print(f"  {at}")