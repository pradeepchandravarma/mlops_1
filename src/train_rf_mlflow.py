import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend, safe for scripts
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


# =====================================================
# CONFIG
# =====================================================
DATA_PATH = r"C:/Users/sugan/OneDrive/Desktop/ITC/MLFlow/mlops_1/data/Student_Performance.csv"
TARGET_COL = "Performance Index"

EXPERIMENT_NAME = "student-performance"
RUN_NAME = "rf-regressor-suganthy"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# RF hyperparams
N_ESTIMATORS = 500
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "sqrt"  # ✅ safer than "auto" for newer sklearn

# Artifacts output (local) - used only for temporary files before logging
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
LOCAL_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# MLFLOW (REMOTE TRACKING SERVER)
# IMPORTANT: set_tracking_uri BEFORE set_experiment
# =====================================================
mlflow.set_tracking_uri("http://localhost:5000")

# Optional but useful: ensure experiment exists and set a stable artifact location
# If your MLflow server already manages artifact location, you can remove this block.
#exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
#if exp is None:
    # Put artifacts into a stable folder (file://) to avoid weird defaults.
    # This does NOT use the broken ./mlruns folder.
#    artifact_location = (PROJECT_ROOT / "mlflow_artifacts").resolve()
#    artifact_location.mkdir(parents=True, exist_ok=True)
#    mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=f"file:///{artifact_location.as_posix()}")
mlflow.set_experiment(EXPERIMENT_NAME)


# =====================================================
# LOAD
# =====================================================
df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found.\nAvailable columns: {list(df.columns)}"
    )

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# =====================================================
# SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


# =====================================================
# PREPROCESS
# =====================================================
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)


# =====================================================
# MODEL + PIPELINE
# =====================================================
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


# =====================================================
# RUN
# =====================================================
with mlflow.start_run(run_name=RUN_NAME):

    # ---- Params ----
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)

    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", str(MAX_DEPTH))
    mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
    mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
    mlflow.log_param("max_features", str(MAX_FEATURES))

    mlflow.log_param("categorical_cols", ",".join(cat_cols) if cat_cols else "None")
    mlflow.log_param("numeric_cols", ",".join(num_cols) if num_cols else "None")

    # ---- Train ----
    pipeline.fit(X_train, y_train)

    # ---- Predict ----
    preds = pipeline.predict(X_test)

    # ---- Metrics ----
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("mae", float(mae))
    mlflow.log_metric("rmse", float(rmse))
    mlflow.log_metric("r2", float(r2))

    # ---- Artifacts ----
    # Residual plot
    residuals = y_test - preds
    plt.figure()
    plt.scatter(preds, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot (RandomForestRegressor)")

    residual_path = LOCAL_ARTIFACT_DIR / "residual_plot.png"
    plt.savefig(residual_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(residual_path))

    # Feature importance (after OHE expansion)
    feature_names = []
    if cat_cols:
        ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
        feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    feature_names.extend(num_cols)

    importances = pipeline.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(30)

    fi_path = LOCAL_ARTIFACT_DIR / "top_feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    mlflow.log_artifact(str(fi_path))

    # ---- Model ----
    signature = infer_signature(X_train, pipeline.predict(X_train))
    input_example = X_train.head(3)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
    )

    print("✅ Logged run to MLflow tracking server at http://localhost:5000")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")