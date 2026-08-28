import pandas as pd
import numpy as np

# Load the full experiment data
df = pd.read_csv("data/main_experiment_luna.csv")

print(f"Total rows: {len(df)}")
print(f"Epsilon values: {sorted(df['epsilon'].unique())}")
print(f"Seeds per epsilon: {df.groupby('epsilon')['seed'].nunique().to_dict()}")
print()

# Basic descriptive stats per epsilon
print("=== Deviation statistics by epsilon ===")
summary = df.groupby("epsilon")["deviation"].agg(["mean", "std", "min", "max"])
print(summary)
print()

# Preview the first few rows
print("=== First 5 rows ===")
print(df.head())