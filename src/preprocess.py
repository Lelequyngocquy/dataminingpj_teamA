import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
    - Remove columns with >95% missing values
    - Handle missing values (impute or drop)
    - Remove duplicates
    - Parse weather values
    - Create TMP_C from TMP
    - Encode categoricals
    - Normalize numeric features
    - Create new features
    - Return processed df and train/test split
    """

    df = df.copy()

    
    threshold = 0.95 * len(df)
    df = df.loc[:, df.isnull().sum() < threshold]

    
    df.drop_duplicates(inplace=True)

    
    def parse_weather_value(s):
        try:
            if isinstance(s, str):
                parts = s.split(",")
                return float(parts[0]) if parts[0] not in ["+9999", "+999"] else np.nan
            return np.nan
        except:
            return np.nan

    
    if "TMP" in df.columns:
        df["TMP_C"] = df["TMP"].apply(parse_weather_value)
        df.drop(columns=["TMP"], inplace=True)
    if "TMP_C" in df.columns:
        df["TMP_CLASS"] = (df["TMP_C"] >= 0).astype(int)

    
    for col in ["DEW", "SLP", "MW1", "VIS"]:
        if col in df.columns:
            df[col + "_VAL"] = df[col].apply(parse_weather_value)
            df.drop(columns=[col], inplace=True)

    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    
    if 'REPORT_TYPE' in df.columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[['REPORT_TYPE']])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['REPORT_TYPE']))
        df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
        df.drop(columns=['REPORT_TYPE'], inplace=True)

    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    
    if 'DATE' in df.columns:
        df['HOUR'] = pd.to_datetime(df['DATE'], errors='coerce').dt.hour
        df.drop(columns=['DATE'], inplace=True)

    
    df = df.select_dtypes(include=[np.number])
    df.dropna(inplace=True)

    
    if 'TMP_C' in df.columns:
        y = df['TMP_C']
        X = df.drop(columns=['TMP_C'])
    else:
        y = df[df.columns[0]]
        X = df.drop(columns=[df.columns[0]])

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, df
