import os
from mlflow.tracking import MlflowClient

# Read required env vars
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME", "student-performance-model")
MODEL_VERSION = os.getenv("MODEL_VERSION")  # e.g. "3"

if not TRACKING_URI or not MODEL_VERSION:
    raise SystemExit(
        "❌ Set MLFLOW_TRACKING_URI and MODEL_VERSION environment variables first."
    )

client = MlflowClient(tracking_uri=TRACKING_URI)

# Set aliases
client.set_registered_model_alias(MODEL_NAME, "staging", MODEL_VERSION)
client.set_registered_model_alias(MODEL_NAME, "maryam", MODEL_VERSION)

print("✅ Aliases set successfully:")
print(f"Model: {MODEL_NAME}")
print(f"Version: {MODEL_VERSION}")
print("Aliases: staging, maryam")
