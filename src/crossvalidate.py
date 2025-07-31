from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def cross_validate_model(X, y, task_type="regression", cv=5):
    """
    Perform k-fold cross validation on the given data.

    Parameters:
        X: pd.DataFrame - features
        y: pd.Series - target
        task_type: "regression" or "classification"
        cv: number of folds

    Returns:
        List of scores (1 per fold)
    """
    if task_type == "regression":
        model = LinearRegression()
        scoring = "neg_root_mean_squared_error"
    elif task_type == "classification":
        model = RandomForestClassifier()
        scoring = "accuracy"
    else:
        raise ValueError("Unsupported task_type for cross-validation")

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # For regression, convert negative RMSE to positive
    return -scores if task_type == "regression" else scores
