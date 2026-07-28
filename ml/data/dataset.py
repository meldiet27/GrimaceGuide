import pandas as pd
df = pd.read_csv("grimaceguide/data/CAT_DATASET_meta.xml")  # adjust path as needed
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("First 3 rows:")
print(df.head(3))
print("\nSample of unique values per column:")
for col in df.columns:
    unique_vals = df[col].unique()
    if len(unique_vals) < 20:
        print(f"  {col}: {unique_vals}")
    else:
        print(f"  {col}: {len(unique_vals)} unique values, e.g. {unique_vals[:5]}")