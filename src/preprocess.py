import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Parse DateTime column
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df["Hour"] = df["DateTime"].dt.hour
        df["Day"] = df["DateTime"].dt.day
        df["Month"] = df["DateTime"].dt.month
        df.drop(columns=["DateTime"], inplace=True)

    # Chuyển dấu phẩy thành dấu chấm (nếu chưa được chuyển)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "."), errors="coerce"
        )

    # Loại bỏ dòng toàn NaN
    df.dropna(axis=0, how="all", inplace=True)

    # Điền khuyết bằng trung bình
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Target mặc định là cột đầu tiên
    y = df[df.columns[0]]
    X = df.drop(columns=[df.columns[0]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return X_train, X_test, y_train, y_test, df
