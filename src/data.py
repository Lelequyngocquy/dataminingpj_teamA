import pandas as pd
import os


def load_data(path: str) -> pd.DataFrame:
    try:
        if path.endswith(".xlsx"):
            df = pd.read_excel(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path, delimiter=";", decimal=",")
        else:
            raise ValueError("Unsupported file format")

        print(f"[INFO] Loaded data from {path} - shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}")
        return pd.DataFrame()


def save_data(df: pd.DataFrame, filename: str, folder: str = "data/processed"):
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)

    if filename.endswith(".xlsx"):
        df.to_excel(full_path, index=False)
    else:
        df.to_csv(full_path, index=False)

    print(f"[INFO] Saved processed data to {full_path}")
