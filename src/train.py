yimport os
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
MODEL_OUT = "models/model.joblib"

FEATURE_COLUMNS = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]
TARGET_COL = "Performance Index"


def ensure_dirs():
    os.makedirs("models", exist_ok=True)


def main():
    ensure_dirs()

    # ✅ IMPORTANT: Do NOT hardcode tracking URI here.
    # Use env var instead. If none is set, default is local sqlite.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    # Use the shared experiment name
    mlflow.set_experiment("student-performance")

    print(f"MLFLOW_TRACKING_URI = {mlflow.get_tracking_uri()}", flush=True)

    df = pd.read_csv(DATA_PATH)

    # enforce correct columns
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].copy()

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="linear_regression_maryam"):
        mlflow.log_param("model", "linear_regression")
        mlflow.log_param("member", "maryam")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # always save locally
        joblib.dump(pipeline, MODEL_OUT)

        # try logging model artifact; if server uses S3 and creds missing, don't block your run
        try:
            mlflow.sklearn.log_model(pipeline, name="model")
        except Exception as e:
            print("⚠️ Model artifact upload skipped:", str(e), flush=True)

        print("Target column:", TARGET_COL, flush=True)
        print(f"RMSE = {rmse:.4f}", flush=True)
        print(f"R2   = {r2:.4f}", flush=True)
        print(f"✅ Model saved to {MODEL_OUT}", flush=True)
        print("✅ Run logged: linear_regression_maryam", flush=True)


if __name__ == "__main__":
    main()
