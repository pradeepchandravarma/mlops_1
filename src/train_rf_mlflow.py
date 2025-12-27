import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


# CONFIG
DATA_PATH = "/mnt/data/Student_Performance.csv"
TARGET_COL = "Performance Index"

EXPERIMENT_NAME = "student-performance-regression"
RUN_NAME = "rf-regressor-v1"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# RF hyperparams (good starting point)
N_ESTIMATORS = 500
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "auto"   # can try "sqrt"


# LOAD
df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found.\nAvailable columns: {list(df.columns)}"
    )

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


# PREPROCESS
# - OneHotEncode categoricals
# - Pass numeric through
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)


# MODEL + PIPELINE
rf = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features=MAX_FEATURES,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", rf),
])


# MLFLOW: EXPERIMENT + RUN
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME):

    # ---- Step 1: Log Params (reproducibility) ----
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)

    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", str(MAX_DEPTH))
    mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
    mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
    mlflow.log_param("max_features", str(MAX_FEATURES))

    mlflow.log_param("categorical_cols", ",".join(cat_cols) if cat_cols else "None")
    mlflow.log_param("numeric_cols", ",".join(num_cols))

    # Step 2: Train
    pipeline.fit(X_train, y_train)

    # Step 3: Predict 
    preds = pipeline.predict(X_test)

    # Step 4: Compute Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    # Step 5: Log Metrics (compare runs) ----
    mlflow.log_metric("mae", float(mae))
    mlflow.log_metric("rmse", float(rmse))
    mlflow.log_metric("r2", float(r2))

    # ---- Step 6: Log Artifacts (plots/files) ----
    os.makedirs("artifacts", exist_ok=True)

    # Residual plot
    residuals = y_test - preds
    plt.figure()
    plt.scatter(preds, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot (RandomForestRegressor)")
    residual_path = "artifacts/residual_plot.png"
    plt.savefig(residual_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(residual_path)

    # Feature importance (after OHE expansion)
    feature_names = []
    if cat_cols:
        ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
        feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    feature_names.extend(num_cols)

    importances = pipeline.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(30)
    fi_path = "artifacts/top_feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path)

    # Step 7: Log Model (deployable)
    signature = infer_signature(X_train, pipeline.predict(X_train))
    input_example = X_train.head(3)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
    )

    print("✅ Logged run to MLflow")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")