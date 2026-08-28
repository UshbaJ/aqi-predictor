from features import load_features
df, _ = load_features()
print(df[["temp", "pressure"]].corr())