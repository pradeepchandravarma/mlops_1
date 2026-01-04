import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


# =====================================================
# Helpers
# =====================================================
def env_required(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"❌ Missing required env var: {key}")
    return value


def atomic_joblib_dump(obj, path: Path):
    """
    Prevents partial / corrupted model files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


# =====================================================
# REQUIRED CONFIG (FAIL FAST)
# =====================================================
MLFLOW_TRACKING_URI = env_required("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = env_required("EXPERIMENT_NAME")
RUN_NAME = env_required("RUN_NAME")

DATA_PATH = env_required("DATA_PATH")
TARGET_COL = env_required("TARGET_COL")

MODEL_PATH = Path(env_required("MODEL_PATH"))
READY_PATH = Path(os.getenv("MODEL_READY_PATH", "/app/models/model.ready"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "/app/artifacts"))

# =====================================================
# OPTIONAL CONFIG
# =====================================================
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "500"))
MAX_DEPTH_RAW = os.getenv("MAX_DEPTH", "")
MAX_DEPTH = None if MAX_DEPTH_RAW == "" else int(MAX_DEPTH_RAW)
MIN_SAMPLES_SPLIT = int(os.getenv("MIN_SAMPLES_SPLIT", "2"))
MIN_SAMPLES_LEAF = int(os.getenv("MIN_SAMPLES_LEAF", "1"))
MAX_FEATURES = os.getenv("MAX_FEATURES", "sqrt")


# =====================================================
# MLflow Init
# =====================================================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("🚀 Training started")
print(f"• Experiment : {EXPERIMENT_NAME}")
print(f"• Run name   : {RUN_NAME}")
print(f"• Data path : {DATA_PATH}")


# =====================================================
# Load Data
# =====================================================
df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features=MAX_FEATURES,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)


# =====================================================
# Train + Log
# =====================================================
if READY_PATH.exists():
    READY_PATH.unlink()

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

with mlflow.start_run(run_name=RUN_NAME):

    # Params
    mlflow.log_params({
        "model": "RandomForestRegressor",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "max_features": MAX_FEATURES,
        "categorical_cols": ",".join(cat_cols) or "None",
        "numeric_cols": ",".join(num_cols),
    })

    # Train
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metrics({
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    })

    # Residual plot
    plt.figure()
    plt.scatter(preds, y_test - preds)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plot_path = ARTIFACT_DIR / "residuals.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(str(plot_path))

    # Log model to MLflow
    signature = infer_signature(X_train, pipeline.predict(X_train))
    """mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(3),
    )"""

    REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME")  # e.g. student-performance-rf

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(3),
        registered_model_name=REGISTERED_MODEL_NAME if REGISTERED_MODEL_NAME else None,
    )


    # Save local model for API
    atomic_joblib_dump(pipeline, MODEL_PATH)
    READY_PATH.write_text("ok")

    print("✅ Training complete")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print(f"📦 Model saved → {MODEL_PATH}")