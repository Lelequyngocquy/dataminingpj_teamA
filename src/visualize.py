import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tmp_distribution(df: pd.DataFrame):
    if "TMP_C" not in df.columns:
        print("[WARN] No 'TMP_C' column to plot.")
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(df["TMP_C"].dropna(), kde=True, color="skyblue")
    plt.title("Temperature Distribution (°C)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color="steelblue", alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--")
    plt.xlabel("Actual TMP_C")
    plt.ylabel("Predicted TMP_C")
    plt.title("Actual vs Predicted TMP_C")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
