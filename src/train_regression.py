"""
Regression Model Training Pipeline for EMIPredict AI.

Trains, evaluates, compares, logs to MLflow, and serializes maximum monthly EMI regressors:
1. Linear Regression (baseline)
2. Random Forest Regressor
3. XGBoost Regressor
With non-negative guarantee (min EMI threshold clip >= Rs. 500).
"""

import argparse
from pathlib import Path
import sys

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data_preprocessing import build_preprocessor, prepare_train_test_data
from src.mlflow_tracking import log_experiment


class SafeRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper that ensures predicted EMI amounts are finite
    and bounded above the minimum sensible threshold (default Rs. 500).
    """

    def __init__(self, base_regressor=None, min_emi=500.0):
        self.base_regressor = base_regressor
        self.min_emi = min_emi

    def fit(self, X, y):
        if self.base_regressor is not None:
            self.model_ = self.base_regressor
        else:
            self.model_ = LinearRegression()
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        raw_preds = self.model_.predict(X)
        # Ensure finite numbers and apply non-negative minimum constraint
        safe_preds = np.nan_to_num(raw_preds, nan=self.min_emi, posinf=self.min_emi, neginf=self.min_emi)
        return np.clip(safe_preds, a_min=self.min_emi, a_max=None)


def evaluate_regressor(name, model, X_test, y_test):
    """Evaluate a trained regressor and return metrics dict and predictions."""
    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(mean_squared_error(y_test, preds) ** 0.5)
    r2 = float(r2_score(y_test, preds))

    # Calculate Mean Absolute Percentage Error (MAPE) safely
    y_test_arr = np.array(y_test)
    non_zero = y_test_arr > 0
    mape = float(np.mean(np.abs((y_test_arr[non_zero] - preds[non_zero]) / y_test_arr[non_zero])) * 100)

    print(f"\n{'=' * 60}")
    print(f"REGRESSOR: {name}")
    print(f"{'=' * 60}")
    print(f"MAE:  Rs. {mae:,.2f}")
    print(f"RMSE: Rs. {rmse:,.2f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }

    return {
        "name": name,
        "metrics": metrics,
        "predictions": preds,
        "model": model,
    }


def plot_regression_diagnostics(
    best_name,
    y_true,
    y_pred,
    actual_pred_path="reports/figures/actual_vs_predicted_emi.png",
    residuals_path="reports/figures/regression_residuals.png",
):
    """Generate and save publication-grade regression diagnostic plots."""
    Path(actual_pred_path).parent.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    residuals = y_true_arr - y_pred_arr

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    sample_idx = np.random.choice(len(y_true_arr), size=min(5000, len(y_true_arr)), replace=False)
    plt.scatter(y_true_arr[sample_idx], y_pred_arr[sample_idx], alpha=0.35, color="#1f77b4", edgecolors="none", s=25)

    max_val = max(np.percentile(y_true_arr, 99.5), np.percentile(y_pred_arr, 99.5))
    plt.plot([0, max_val], [0, max_val], color="#d62728", linestyle="--", linewidth=2, label="Perfect Fit (y = x)")

    plt.title(f"Actual vs Predicted Maximum Monthly EMI — {best_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Max Monthly EMI (Rs.)", fontsize=11)
    plt.ylabel("Predicted Max Monthly EMI (Rs.)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(actual_pred_path, dpi=180)
    plt.close()
    print(f"Saved Actual vs Predicted plot to: {actual_pred_path}")

    # Plot 2: Residual Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals[sample_idx], kde=True, color="#2ca02c", bins=50)
    plt.axvline(0, color="#d62728", linestyle="--", linewidth=1.5, label="Zero Error")
    plt.title(f"Residual Distribution (Actual - Predicted) — {best_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Prediction Error (Rs.)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(residuals_path, dpi=180)
    plt.close()
    print(f"Saved Residuals plot to: {residuals_path}")


def train_all_regressors(
    X_train,
    X_test,
    y_train,
    y_test,
    track_mlflow=True,
):
    """Train, evaluate, and compare Linear Regression, Random Forest, and XGBoost Regressor."""
    evaluations = []
    trained_pipelines = {}

    # 1. Linear Regression
    print("\n[1/3] Training Linear Regression...")
    linear_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", SafeRegressor(LinearRegression(), min_emi=500.0)),
    ])
    linear_pipe.fit(X_train, y_train)
    res_lin = evaluate_regressor("Linear Regression", linear_pipe, X_test, y_test)
    evaluations.append(res_lin)
    trained_pipelines["linear_regression"] = linear_pipe

    if track_mlflow:
        log_experiment(
            model_name="Linear Regression",
            model=linear_pipe,
            metrics=res_lin["metrics"],
            params={"model_type": "LinearRegression", "min_emi_constraint": 500.0},
            tags={"task": "regression", "framework": "scikit-learn"},
        )

    # 2. Random Forest Regressor
    print("\n[2/3] Training Random Forest Regressor...")
    rf_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", SafeRegressor(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            ),
            min_emi=500.0,
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    res_rf = evaluate_regressor("Random Forest Regressor", rf_pipe, X_test, y_test)
    evaluations.append(res_rf)
    trained_pipelines["random_forest"] = rf_pipe

    if track_mlflow:
        log_experiment(
            model_name="Random Forest Regressor",
            model=rf_pipe,
            metrics=res_rf["metrics"],
            params={"n_estimators": 150, "max_depth": 18, "min_emi_constraint": 500.0},
            tags={"task": "regression", "framework": "scikit-learn"},
        )

    # 3. XGBoost Regressor
    print("\n[3/3] Training XGBoost Regressor...")
    xgb_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", SafeRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=250,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
            min_emi=500.0,
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    res_xgb = evaluate_regressor("XGBoost Regressor", xgb_pipe, X_test, y_test)
    evaluations.append(res_xgb)
    trained_pipelines["xgboost"] = xgb_pipe

    if track_mlflow:
        log_experiment(
            model_name="XGBoost Regressor",
            model=xgb_pipe,
            metrics=res_xgb["metrics"],
            params={
                "n_estimators": 250,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_emi_constraint": 500.0,
            },
            tags={"task": "regression", "framework": "xgboost"},
        )

    # Build comparison summary DataFrame sorted by lowest RMSE
    summary_rows = []
    for ev in evaluations:
        m = ev["metrics"]
        summary_rows.append({
            "model": ev["name"],
            "MAE": round(m["MAE"], 2),
            "RMSE": round(m["RMSE"], 2),
            "R2": round(m["R2"], 4),
            "MAPE": round(m["MAPE"], 2),
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(by="RMSE", ascending=True)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("REGRESSION MODEL BENCHMARK COMPARISON")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Find best model
    best_eval = min(evaluations, key=lambda x: x["metrics"]["RMSE"])

    # Plot diagnostic figures for best model
    plot_regression_diagnostics(
        best_name=best_eval["name"],
        y_true=y_test,
        y_pred=best_eval["predictions"],
    )

    # Save comparison CSV
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    summary_df.to_csv(reports_dir / "regression_model_comparison.csv", index=False)

    return trained_pipelines, evaluations, summary_df


def save_best_regressor(evaluations, output_dir="artifacts"):
    """Identify best regressor (lowest RMSE) and save to artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    best_eval = min(evaluations, key=lambda x: x["metrics"]["RMSE"])

    pipeline_path = out_dir / "regression_pipeline.joblib"
    joblib.dump(best_eval["model"], pipeline_path)
    print(f"\n[Saved Best Regressor] '{best_eval['name']}' saved to {pipeline_path}")

    return best_eval


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate maximum monthly EMI regressors.")
    parser.add_argument("--data-path", type=str, default="data/emi_prediction_dataset.csv", help="Path to CSV dataset.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample size for fast testing.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking.")
    args = parser.parse_args()

    print("Loading and preparing dataset for regression...")
    data = prepare_train_test_data(csv_path=args.data_path, sample_size=args.sample_size)

    trained_pipelines, evaluations, summary_df = train_all_regressors(
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=data["y_reg_train"],
        y_test=data["y_reg_test"],
        track_mlflow=not args.no_mlflow,
    )

    save_best_regressor(evaluations)


if __name__ == "__main__":
    main()
