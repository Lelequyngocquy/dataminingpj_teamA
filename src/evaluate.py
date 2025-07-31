from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.utils.multiclass import type_of_target
import numpy as np
import pandas as pd


def evaluate(
    y_true, y_pred, task_type: str = "regression", X_test: pd.DataFrame = None
) -> dict:
    if task_type == "regression":
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²": r2_score(y_true, y_pred),
        }

    elif task_type == "classification":
        y_true_type = type_of_target(y_true)
        y_pred_type = type_of_target(y_pred)

        if y_pred_type.startswith("continuous"):
            if len(np.unique(y_true)) == 2:

                y_pred = (y_pred >= 0.5).astype(int)
            else:

                if y_pred.ndim == 2:
                    y_pred = np.argmax(y_pred, axis=1)
                else:

                    pass

        if y_true_type not in ["binary", "multiclass"]:
            return {
                "Error": "y_true must be label (binary or multiclass), not continuous."
            }

        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "ROC AUC": (
                roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else "N/A"
            ),
        }

    elif task_type == "clustering":
        if len(np.unique(y_pred)) <= 1 or len(np.unique(y_pred)) >= len(X_test):
            return {"Error": f"Invalid number of clusters: {len(np.unique(y_pred))}"}

        return {
            "Silhouette Score": silhouette_score(X_test, y_pred),
            "Davies-Bouldin Score": davies_bouldin_score(X_test, y_pred),
        }

    else:
        return {"Error": "Unsupported task type"}
