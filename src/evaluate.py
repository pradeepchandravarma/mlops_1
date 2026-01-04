
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "mae": mae,
        "mse": mse,
        "r2": r2
    }
