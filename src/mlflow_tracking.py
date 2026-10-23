"""
MLflow Experiment Tracking Module for EMIPredict AI.

Provides unified logging for classification and regression models,
hyperparameters, evaluation metrics, and model artifacts with SQLite backend.
"""

from pathlib import Path
import os
import sys

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import mlflow
import mlflow.sklearn

# Silence agent tracing hint
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

# Portable SQLite tracking URI
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
EXPERIMENT_NAME = "EMIPredict-AI"


def init_mlflow():
    """Initialize MLflow tracking URI and experiment."""
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return True
    except Exception as e:
        print(f"[MLflow Warning] Could not initialize MLflow: {e}")
        return False


def log_experiment(model_name, model, metrics, params=None, tags=None, artifacts=None):
    """
    Log model parameters, metrics, tags, and model artifact to MLflow.
    Returns the run_id on success, or None on failure.
    """
    if not init_mlflow():
        return None

    try:
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id

            if tags:
                mlflow.set_tags(tags)

            if params:
                safe_params = {
                    k: (str(v) if not isinstance(v, (int, float, bool, str)) else v)
                    for k, v in params.items()
                }
                mlflow.log_params(safe_params)

            if metrics:
                mlflow.log_metrics({
                    key: float(value)
                    for key, value in metrics.items()
                    if value is not None and not (isinstance(value, float) and (value != value))
                })

            if model is not None:
                try:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        name="model",
                        serialization_format="cloudpickle",
                    )
                except Exception as model_err:
                    print(f"[MLflow Warning] Could not log sklearn model: {model_err}")

            if artifacts:
                for file_path in artifacts:
                    p = Path(file_path)
                    if p.exists():
                        mlflow.log_artifact(str(p))

            print(f"Logged experiment: '{model_name}' (Run ID: {run_id})")
            return run_id
    except Exception as e:
        print(f"[MLflow Warning] Failed to log experiment '{model_name}': {e}")
        return None


def get_experiment_runs(limit=20):
    """Retrieve logged runs from the EMIPredict-AI experiment as a DataFrame."""
    try:
        init_mlflow()
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            return None
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"],
        )
        data = []
        for r in runs:
            row = {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
            }
            row.update(r.data.metrics)
            data.append(row)
        import pandas as pd
        df = pd.DataFrame(data)
        if not df.empty and "start_time" in df.columns:
            df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
        return df
    except Exception as e:
        print(f"[MLflow Warning] Could not fetch runs: {e}")
        return None