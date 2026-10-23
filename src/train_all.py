"""
Unified Training Pipeline for EMIPredict AI.

Executes both classification and regression model pipelines, evaluates all candidates,
logs runs to MLflow, and serializes best models and metadata into artifacts/.
"""

import argparse
from datetime import datetime
from pathlib import Path
import time
import pandas as pd

from src.data_preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    VALID_CATEGORIES,
    prepare_train_test_data,
)
from src.model_metadata import save_metadata
from src.train_classification import save_best_classifier, train_all_classifiers
from src.train_regression import save_best_regressor, train_all_regressors


def run_training_pipeline(
    data_path="data/emi_prediction_dataset.csv",
    sample_size=None,
    track_mlflow=True,
):
    print("=" * 80)
    print("EMIPREDICT AI — END-TO-END MODEL TRAINING & BENCHMARKING")
    print("=" * 80)
    t_start = time.time()

    print(f"\n1. Loading and cleaning dataset from '{data_path}'...")
    data = prepare_train_test_data(csv_path=data_path, sample_size=sample_size)
    print(f"   Train set size: {len(data['X_train']):,} records")
    print(f"   Test set size:  {len(data['X_test']):,} records")

    # Train classification models
    print("\n2. Training Classification Models (EMI Eligibility)...")
    clf_pipelines, clf_evals, clf_summary = train_all_classifiers(
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=data["y_class_train"],
        y_test=data["y_class_test"],
        track_mlflow=track_mlflow,
    )
    best_clf = save_best_classifier(clf_evals)

    # Train regression models
    print("\n3. Training Regression Models (Maximum Monthly EMI)...")
    reg_pipelines, reg_evals, reg_summary = train_all_regressors(
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=data["y_reg_train"],
        y_test=data["y_reg_test"],
        track_mlflow=track_mlflow,
    )
    best_reg = save_best_regressor(reg_evals)

    # Build and save unified metadata
    print("\n4. Generating and saving model_metadata.json...")
    metadata = {
        "model_version": "1.0.0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": Path(data_path).name,
        "dataset_records": len(data["X_train"]) + len(data["X_test"]),
        "input_features_count": len(NUMERICAL_COLUMNS) + len(CATEGORICAL_COLUMNS),
        "numerical_features": NUMERICAL_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "valid_categories": VALID_CATEGORIES,
        "classification": {
            "target": "emi_eligibility",
            "classes": ["Eligible", "High_Risk", "Not_Eligible"],
            "selected_model": best_clf["name"],
            "metrics": best_clf["metrics"],
            "all_models": clf_summary.to_dict(orient="records"),
        },
        "regression": {
            "target": "max_monthly_emi",
            "selected_model": best_reg["name"],
            "metrics": best_reg["metrics"],
            "all_models": reg_summary.to_dict(orient="records"),
        },
    }
    save_metadata(metadata)

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE IN {elapsed:.1f} SECONDS!")
    print("=" * 80)
    print(f"Selected Classifier: {best_clf['name']} (Macro F1: {best_clf['metrics']['macro_f1']:.4f})")
    print(f"Selected Regressor:  {best_reg['name']} (RMSE: ₹{best_reg['metrics']['RMSE']:,.2f}, R²: {best_reg['metrics']['R2']:.4f})")
    print("Artifacts saved to 'artifacts/'")
    print("Reports & Figures saved to 'reports/' and 'reports/figures/'")


def main():
    parser = argparse.ArgumentParser(description="Run complete EMIPredict AI training pipeline.")
    parser.add_argument("--data-path", type=str, default="data/emi_prediction_dataset.csv")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size (leave empty for full dataset).")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data_path,
        sample_size=args.sample_size,
        track_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
