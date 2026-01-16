from datetime import datetime
import os

import joblib
import mlflow
import mlflow.sklearn

from src.train import train_regression
from src.evaluate import model_evaluate


def generate_model_name(your_name: str, model_name: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{your_name}-{model_name}-{timestamp}"


def main():
    # 1) Get MLflow tracking URI from environment (AWS CodeBuild will set this)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI env var is not set")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("student-performance")

    with mlflow.start_run(run_name="sgd_regression_rishika"):
        model, scaler, X_test, y_test = train_regression()

        print("----------Gradient Descent Model Trained Successfully------------")

        os.makedirs("models", exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler}, "models/student_performance_sgd.joblib")

        registered_model_latest = generate_model_name(
            your_name="suganthy",
            model_name="sgd_student_performance"
        )
        print("Registering model as:", registered_model_latest)

        # 2) Correct MLflow API: artifact_path (NOT name)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_latest
        )

        metrics = model_evaluate(model, X_test, y_test)
        print("\n------------Model Metrics------------")
        print(metrics)

        mlflow.log_metric("mse", metrics["mse"])
        mlflow.log_metric("r2", metrics["r2"])

        mlflow.log_param("model_type", "SGDRegressor")
        mlflow.log_param("loss", "squared_error")
        mlflow.log_param("learning_rate", "constant")
        mlflow.log_param("eta0", 0.01)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)


if __name__ == "__main__":
    main()