import os
import time
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient


def env_required(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"Missing required env var: {key}")
    return v


MLFLOW_TRACKING_URI = env_required("MLFLOW_TRACKING_URI")
REGISTERED_MODEL_NAME = env_required("REGISTERED_MODEL_NAME")

# which alias to export (champion/staging)
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# where to export joblib
EXPORT_PATH = Path(os.getenv("EXPORT_MODEL_PATH", "/app/models/model.joblib"))
READY_PATH = Path(os.getenv("MODEL_READY_PATH", "/app/models/model.ready"))

# local fallback (what train writes)
LOCAL_FALLBACK_PATH = Path(os.getenv("LOCAL_FALLBACK_PATH", "/app/models/model.joblib"))


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"📦 Exporting model from registry alias: models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")

    # 1) find which version champion points to
    alias_info = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    version = alias_info.version
    run_id = alias_info.run_id

    print(f"✅ Alias {MODEL_ALIAS} -> version {version}, run_id {run_id}")

    # 2) try to pull model from run artifacts (preferred)
    model_uri = f"runs:/{run_id}/model"

    try:
        print(f"⬇️  Trying to load from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        tmp = EXPORT_PATH.with_suffix(".tmp")
        joblib.dump(model, tmp)
        os.replace(tmp, EXPORT_PATH)

        READY_PATH.write_text("ok")
        print(f"✅ Exported model to: {EXPORT_PATH}")
        print(f"✅ Ready flag written: {READY_PATH}")
        return

    except Exception as e:
        print("❌ Failed to export from MLflow artifacts (likely missing S3 artifacts).")
        print(f"Reason: {e}")
        print("➡️ Falling back to local shared model file...")

    # 3) fallback
    if not LOCAL_FALLBACK_PATH.exists() or LOCAL_FALLBACK_PATH.stat().st_size < 1024:
        raise RuntimeError(
            f"Fallback model not available at {LOCAL_FALLBACK_PATH}. "
            f"Train may not have produced it."
        )

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = EXPORT_PATH.with_suffix(".tmp")
    joblib.dump(joblib.load(LOCAL_FALLBACK_PATH), tmp)
    os.replace(tmp, EXPORT_PATH)

    READY_PATH.write_text("ok")
    print(f"✅ Fallback export OK -> {EXPORT_PATH}")


if __name__ == "__main__":
    main()