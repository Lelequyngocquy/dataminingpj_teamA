from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, X_test, task_type="regression"):
    """
    Train a model for regression or classification.

    Parameters:
        X_train, y_train: training data
        X_test: input to predict
        task_type: 'regression' or 'classification'

    Returns:
        model, y_pred
    """
    if task_type == "regression":
        model = LinearRegression()
    elif task_type == "classification":
        model = RandomForestClassifier()
    else:
        raise ValueError("Unsupported task_type. Use 'regression' or 'classification'.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred
