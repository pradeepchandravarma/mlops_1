from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def evaluate_model(model, X_test, y_test):
    # Keras evaluation
    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    # Predictions
    predictions = model.predict(X_test).ravel()

    # Sklearn metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
