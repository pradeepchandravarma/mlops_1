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
LOCAL_MODEL_DIR = "models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.joblib")

# Target detection (robust)
TARGET_CANDIDATES = [
    "Performance Index",
    "Performance_Index",
    "performance_index",
]


def detect_target(columns):
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    # fallback: last column if not found
    return columns[-1]


def main():
    # ---------- MLflow: point to shared server via tunnel ----------
    # IMPORTANT: tunnel must be up: ssh -L 5000:localhost:5000 ...
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("student-performance")

    # ---------- Load data ----------
    df = pd.read_csv(DATA_PATH)
    target = detect_target(list(df.columns))

    y = df[target]
    X = df.drop(columns=[target])

    # Identify numeric/categorical columns
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    # Preprocessing pipelines
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

    model = LinearRegression()

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # Train/test split
    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ---------- MLflow run ----------
    run_name = "linear_regression_maryam"

    with mlflow.start_run(run_name=run_name):
        # Tags (useful for filtering/comparison)
        mlflow.set_tag("member", "maryam")
        mlflow.set_tag("dataset", "student-performance")
        mlflow.set_tag("problem_type", "regression")

        # Params
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("target", target)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("num_features", len(num_cols))
        mlflow.log_param("cat_features", len(cat_cols))

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        preds = pipeline.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Save locally
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        joblib.dump(pipeline, LOCAL_MODEL_PATH)

        # Try to log model artifact to MLflow (may fail if S3 creds missing)
        try:
            mlflow.sklearn.log_model(pipeline, name="model")
        except Exception as e:
            print("⚠️ Skipping MLflow model artifact upload (likely S3 creds missing):", e)

        print("Target column:", target)
        print(f"RMSE = {rmse:.4f}")
        print(f"R2   = {r2:.4f}")
        print("Model saved locally to:", LOCAL_MODEL_PATH)
        print("✅ Run logged to MLflow with name:", run_name)


if __name__ == "__main__":
    main()
