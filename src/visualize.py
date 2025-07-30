import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_target_distribution(df: pd.DataFrame, target_name: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if target_name not in df.columns:
        print(f"[WARN] Column '{target_name}' not found in DataFrame.")
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_name].dropna(), kde=True, color="skyblue")
    plt.title(f"Distribution of {target_name} - The last trained")
    plt.xlabel(target_name)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, target_name="Target"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color="steelblue", alpha=0.6)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="red",
        linestyle="--",
    )
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Actual vs Predicted {target_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
