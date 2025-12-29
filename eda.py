import pandas as pd

df = pd.read_csv("Student_Performance.csv")

print("\n--- Shape ---")
print(df.shape)

print("\n--- Columns & dtypes ---")
print(df.dtypes)

print("\n--- Missing values ---")
print(df.isna().sum().sort_values(ascending=False))

print("\n--- Describe numeric ---")
print(df.describe())

print("\n--- First 5 rows ---")
print(df.head())