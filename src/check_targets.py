import pandas as pd
from features import build_full_dataset

df = build_full_dataset()
for h in [24, 48, 72]:
    col = f"target_{h}h"
    print(f"{col}: {df[col].isna().sum()} nulls (expect ~{h}), "
          f"{df[col].notna().sum()} valid values")
    
gaps = df["datetime"].diff().dropna()
non_hourly = gaps[gaps != pd.Timedelta(hours=1)]
print(f"Non-hourly gaps found: {len(non_hourly)}")
print(non_hourly)

print(df["aqi_epa"].isna().sum())   