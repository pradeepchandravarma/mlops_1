from dotenv import load_dotenv
load_dotenv()

import os
import joblib
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

from .evaluate import evaluate_model


def get_next_run_name(experiment_name, base_name="regression_NN_pradeep"):
    """
    Generate an auto-incrementing MLflow run name.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)

    if exp is None:
        return f"{base_name}_1"

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=1000
    )
    return f"{base_name}_{len(runs) + 1}"


def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    scaler,
    epochs,
    batch_size,
    validation_split,
    artifacts_dir="artifacts",
    experiment_name="student-performance"
):
    # ---------------------------
    # MLflow setup (DO NOT force artifact location)
    # ---------------------------
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)

    run_name = get_next_run_name(experiment_name)

    with mlflow.start_run(run_name=run_name):

        # ---------------------------
        # Log parameters
        # ---------------------------
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("validation_split", validation_split)
        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_param("model_type", "NeuralNetwork")
        mlflow.log_param("problem_type", "Regression")

        # ---------------------------
        # Train model
        # ---------------------------
        history = model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # ---------------------------
        # Evaluate model (TEST SET)
        # ---------------------------
        metrics = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        # ---------------------------
        # Save artifacts LOCALLY (do not delete old ones)
        # ---------------------------
        os.makedirs(artifacts_dir, exist_ok=True)

        model_path = os.path.join(artifacts_dir, "model.h5")
        scaler_path = os.path.join(artifacts_dir, "scaler.joblib")

        model.save(model_path, overwrite=True)
        joblib.dump(scaler, scaler_path)

        # ---------------------------
        # Log artifacts to MLflow (server decides storage)
        # ---------------------------
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        mlflow.tensorflow.log_model(
            model,
            artifact_path="tf_model"
        )

        # Helpful tags (non-breaking)
        mlflow.set_tag("run_display_name", run_name)
        mlflow.set_tag("artifact_source", "local_fallback")

        print(f"MLflow run '{run_name}' logged successfully")
        print(f"Local artifacts available at: {artifacts_dir}/")

    return history
