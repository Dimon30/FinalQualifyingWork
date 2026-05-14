# merge_datasets.py (запустить из корня проекта)
import pandas as pd, glob

parts = sorted(glob.glob("code_app/ml/data/dataset_large_s*.csv"))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle
out = "code_app/ml/data/dataset_large.csv"
df.to_csv(out, index=False)
print(f"Merged {len(df)} rows from {len(parts)} files → {out}")