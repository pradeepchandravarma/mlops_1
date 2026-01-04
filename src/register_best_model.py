import os
import time
import mlflow
from mlflow.tracking import MlflowClient


def env_required(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise RuntimeError(f"Missing env var: {key}")
    return v

print("MLFLOW_TRACKING_URI =", os.getenv("MLFLOW_TRACKING_URI"))
MLFLOW_TRACKING_URI = env_required("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = env_required("EXPERIMENT_NAME")

REGISTERED_MODEL_NAME = env_required("REGISTERED_MODEL_NAME")
SELECTION_MODE = os.getenv("SELECTION_MODE", "combo")  # r2 | rmse | combo
MODEL_ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "model")  # usually "model"
ALIAS_NAME = os.getenv("ALIAS_NAME", "suganthy")


def select_best(runs_df, mode: str):
    if runs_df.empty:
        raise RuntimeError("No runs found in the experiment.")

    # ensure required metrics exist
    for col in ["metrics.r2", "metrics.rmse"]:
        if col not in runs_df.columns:
            raise RuntimeError(f"Missing {col} in runs. Columns: {list(runs_df.columns)}")

    if mode == "r2":
        return runs_df.sort_values("metrics.r2", ascending=False).iloc[0]
    if mode == "rmse":
        return runs_df.sort_values("metrics.rmse", ascending=True).iloc[0]
    if mode == "combo":
        return runs_df.sort_values(["metrics.r2", "metrics.rmse"], ascending=[False, True]).iloc[0]

    raise ValueError("SELECTION_MODE must be: r2 | rmse | combo")


def wait_ready(client: MlflowClient, name: str, version: str, timeout_s: int = 300):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        mv = client.get_model_version(name, version)
        if mv.status == "READY":
            return
        time.sleep(2)
    raise RuntimeError(f"Model version not READY after {timeout_s}s: {name} v{version}")


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="pandas")
    best = select_best(runs, SELECTION_MODE)

    run_id = best["run_id"]
    r2 = float(best["metrics.r2"])
    rmse = float(best["metrics.rmse"])

    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"

    print("Best run selected:")
    print(f"  run_id: {run_id}")
    print(f"  r2    : {r2}")
    print(f"  rmse  : {rmse}")
    print(f"  model : {model_uri}")

    #mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    mv = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=REGISTERED_MODEL_NAME)
    version = str(mv.version)

    wait_ready(client, REGISTERED_MODEL_NAME, version)

    # ✅ ALWAYS repoint aliases to this newly created version
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "staging", version)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_NAME, version)

    # helpful tags
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "selection_mode", SELECTION_MODE)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "r2", str(r2))
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "rmse", str(rmse))
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "source_run_id", run_id)

    print(f"✅ Registered model: {REGISTERED_MODEL_NAME} v{version}")
    print(f"✅ Aliases updated -> staging, champion, {ALIAS_NAME} all point to v{version}")


if __name__ == "__main__":
    main()