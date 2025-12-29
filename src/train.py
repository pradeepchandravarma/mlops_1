import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = "data/Student_Performance.csv"
TARGET_CANDIDATES = ["Performance Index", "Performance_Index", "performance_index"]


def detect_target(columns):
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    return columns[-1]  # fallback


def main():
    # 1) Connect to shared MLflow server (via SSH tunnel)
    mlflow.set_tracking_uri("http://localhost:5000")

    # 2) Everyone uses the SAME experiment name
    mlflow.set_experiment("student-performance")

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    target = detect_target(list(df.columns))

    y = df[target]
    X = df.drop(columns=[target])

    # Columns
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    # Preprocess
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])

    # Split
    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Run name = your model + your name
    run_name = "linear_regression_maryam"

    with mlflow.start_run(run_name=run_name):
        # Tags (easy filtering in UI)
        mlflow.set_tag("member", "maryam")
        mlflow.set_tag("dataset", "Student_Performance")
        mlflow.set_tag("problem_type", "regression")

        # Params
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("target", target)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Train + predict
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Metrics
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Save local model (optional)
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/model.joblib")

        # Log model artifact to MLflow (may require S3 creds on client)
        try:
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            print("✅ Model artifact logged to MLflow (S3).")
        except Exception as e:
            print("⚠️ Metrics logged, but model artifact upload skipped (S3 creds missing).")
            print("   Error:", e)

        print("Target column:", target)
        print(f"RMSE = {rmse:.4f}")
        print(f"R2   = {r2:.4f}")
        print("✅ Run logged:", run_name)


if __name__ == "__main__":
    main()
