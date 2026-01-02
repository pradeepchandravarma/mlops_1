# src/utils/get_latest_model_version.py
import os
from mlflow.tracking import MlflowClient

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
model_name = os.getenv("MODEL_NAME")

if not tracking_uri:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set")

if not model_name:
    raise RuntimeError("MODEL_NAME is not set")

client = MlflowClient(tracking_uri=tracking_uri)

versions = client.search_model_versions(f"name='{model_name}'")
latest = max((int(v.version) for v in versions), default=None)

if latest is None:
    raise RuntimeError(f"No versions found for model '{model_name}'")

print(latest)
