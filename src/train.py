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

TARGET_CANDIDATES = [
    "Performance Index",
    "Performance_Index",
    "performance_index",
]


def detect_target(columns):
    for c in TARGET_CANDIDATES:
        if c in columns:
            return c
    return columns[-1]  # fallback


def main():
    df = pd.read_csv(DATA_PATH)
    target = detect_target(list(df.columns))

    y = df[target]
    X = df.drop(columns=[target])

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ]
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("student-performance")

    with mlflow.start_run(run_name="maryam-lr-v1"):
        mlflow.log_param("model", "linear_regression")
        mlflow.log_param("target", target)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("member", "maryam")

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        joblib.dump(pipeline, "models/model.joblib")
        mlflow.sklearn.log_model(pipeline, name="model")

        print("Target column:", target)
        print(f"RMSE = {rmse:.4f}")
        print(f"R2   = {r2:.4f}")
        print("Model saved to models/model.joblib")


if __name__ == "__main__":
    main()
