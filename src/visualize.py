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

    data = df[target_name].dropna()

    plt.figure(figsize=(10, 7))

    # Subplot 1: Histogram + KDE
    plt.subplot(2, 1, 1)
    sns.histplot(data, kde=True, color="skyblue")
    plt.title(f"Distribution of {target_name}")
    plt.xlabel("")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.xlabel(target_name)
    # Subplot 2: Boxplot
    plt.subplot(2, 1, 2)
    sns.boxplot(x=data, color="lightcoral")

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
