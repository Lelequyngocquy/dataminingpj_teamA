import pandas as pd
import os

def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Loaded data from {path} — shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return pd.DataFrame()

def save_csv(df: pd.DataFrame, filename: str, folder: str = "data/processed"):
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)
    df.to_csv(full_path, index=False)
    print(f"[INFO] Saved processed data to {full_path}")
