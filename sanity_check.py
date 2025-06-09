import pandas as pd

df = pd.read_csv("Datasets/processed_wednesday_dataset.csv", low_memory=False)
print(df.dtypes)
print(df["Label"].value_counts())