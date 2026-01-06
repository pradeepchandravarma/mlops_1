import joblib
import mlflow
import os
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlflow.models import infer_signature

from config import Config
from dotenv import load_dotenv
import os

load_dotenv()

def build_pipeline(X: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    model = Lasso(max_iter=20000)

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipe, num_cols, cat_cols


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": rmse, "mae": float(mae), "r2": float(r2)}


def main():
    cfg = Config()
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # MLflow tracking + experiment
    #mlflow.set_tracking_uri("http://localhost:5000")
    
    #mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlops-mlflow-server-1077737027.eu-west-2.elb.amazonaws.com"))
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://mlops-mlflow-server-1077737027.eu-west-2.elb.amazonaws.com"
    )
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")

    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLFLOW_TRACKING_URI = {mlflow.get_tracking_uri()}", flush=True)


    mlflow.set_experiment("student-performance")

    df = pd.read_csv(cfg.data_path)

    if cfg.target_col not in df.columns:
        raise ValueError(
            f"Target column '{cfg.target_col}' not found. Available: {list(df.columns)}"
        )

    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    pipe, num_cols, cat_cols = build_pipeline(X_train)

    # Alpha grid for Lasso
    param_grid = {
        "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
    }

    with mlflow.start_run(run_name="lasso_regression_vidya"):
        mlflow.set_tag("mlflow.user", "Vidya")
        mlflow.log_param("target_col", cfg.target_col)
        mlflow.log_param("test_size", cfg.test_size)
        mlflow.log_param("random_state", cfg.random_state)
        mlflow.log_param("numeric_cols", ",".join(num_cols))
        mlflow.log_param("categorical_cols", ",".join(cat_cols))

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="r2",
            cv=5,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)

        best_model: Pipeline = gs.best_estimator_
        best_params = gs.best_params_
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        # Evaluate on holdout test split
        y_pred = best_model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        # Save model locally (optional, for your own use)
        joblib.dump(best_model, cfg.model_path)

        # ---- KEY CHANGE: Log model in MLflow model format (like your teammate) ----
        signature = infer_signature(X_train, best_model.predict(X_train))
        model_name = os.getenv("MODEL_NAME")

        if not model_name:
            raise RuntimeError("MODEL_NAME env var not set")

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=X_train.head(5),
        )
        run_id = mlflow.active_run().info.run_id

        #mlflow.register_model(model_uri=f"runs:/{run_id}/model",name="student_lasso_model")

        #mlflow.log_artifacts(str(cfg.artifacts_dir), artifact_path="artifacts")

        print("Training complete.")
        print("Best params:", best_params)
        print("Holdout test metrics:", metrics)
        print(f"Saved model locally to: {cfg.model_path}")
        


if __name__ == "__main__":
    main()

print("RUNNING UPDATED train_lasso.py")