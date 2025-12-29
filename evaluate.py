import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import Config


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": rmse, "mae": float(mae), "r2": float(r2)}


def main():
    cfg = Config()

    df = pd.read_csv(cfg.data_path)
    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    model = joblib.load(cfg.model_path)
    y_pred = model.predict(X_test)

    metrics = regression_metrics(y_test, y_pred)

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name="evaluate_saved_model"):
        mlflow.log_param("model_path", str(cfg.model_path))
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})

    print("Evaluation complete.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
