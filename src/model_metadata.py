"""
Model Metadata and Persistence Module for EMIPredict AI.

Handles saving and loading model metadata, feature schemas, evaluation benchmarks,
and providing helpful diagnostics when model artifacts are missing.
"""

from datetime import datetime
import json
from pathlib import Path
import joblib

from src.data_preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    VALID_CATEGORIES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METADATA_FILE = ARTIFACTS_DIR / "model_metadata.json"
CLASSIFICATION_PIPELINE_FILE = ARTIFACTS_DIR / "classification_pipeline.joblib"
REGRESSION_PIPELINE_FILE = ARTIFACTS_DIR / "regression_pipeline.joblib"


def get_default_metadata():
    """Return default schema metadata."""
    return {
        "model_version": "1.0.0",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "emi_prediction_dataset.csv",
        "dataset_records": 404800,
        "input_features_count": len(NUMERICAL_COLUMNS) + len(CATEGORICAL_COLUMNS),
        "numerical_features": NUMERICAL_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "valid_categories": VALID_CATEGORIES,
        "classification": {
            "target": "emi_eligibility",
            "classes": ["Eligible", "High_Risk", "Not_Eligible"],
            "selected_model": "Weighted XGBoost Classifier",
            "metrics": {
                "accuracy": 0.8950,
                "macro_precision": 0.7580,
                "macro_recall": 0.7720,
                "macro_f1": 0.7580,
                "high_risk_recall": 0.8570,
            },
        },
        "regression": {
            "target": "max_monthly_emi",
            "selected_model": "XGBoost Regressor",
            "metrics": {
                "MAE": 647.64,
                "RMSE": 1014.81,
                "R2": 0.9826,
                "MAPE": 11.24,
            },
        },
    }


def save_metadata(metadata: dict):
    """Save metadata dictionary to artifacts/model_metadata.json."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {METADATA_FILE}")


def load_metadata() -> dict:
    """Load metadata dictionary from artifacts/model_metadata.json or fallback to defaults."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to read {METADATA_FILE}: {e}")
    return get_default_metadata()


def load_artifacts():
    """
    Load fitted classification and regression pipelines along with metadata.
    Raises FileNotFoundError with instructions if artifacts are missing.
    """
    if not CLASSIFICATION_PIPELINE_FILE.exists() or not REGRESSION_PIPELINE_FILE.exists():
        missing = []
        if not CLASSIFICATION_PIPELINE_FILE.exists():
            missing.append(str(CLASSIFICATION_PIPELINE_FILE))
        if not REGRESSION_PIPELINE_FILE.exists():
            missing.append(str(REGRESSION_PIPELINE_FILE))
        raise FileNotFoundError(
            f"Model artifacts not found: {missing}. "
            "Please run 'python -m src.train_all' to train models and generate artifacts."
        )

    clf_pipeline = joblib.load(CLASSIFICATION_PIPELINE_FILE)
    reg_pipeline = joblib.load(REGRESSION_PIPELINE_FILE)
    metadata = load_metadata()

    return {
        "classification_pipeline": clf_pipeline,
        "regression_pipeline": reg_pipeline,
        "metadata": metadata,
    }
