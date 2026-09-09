import pandas as pd
df = pd.read_parquet(r"c:\Users\Dell\OneDrive\Desktop\BTP\data\raw\stock_prices.parquet")
print("Columns:")
for c in df.columns:
    print(repr(c))
print("\nHead:")
print(df.head())
