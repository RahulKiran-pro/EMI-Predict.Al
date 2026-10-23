"""
Classification Model Training Pipeline for EMIPredict AI.

Trains, evaluates, compares, logs to MLflow, and serializes EMI eligibility classifiers:
1. Logistic Regression (balanced)
2. Random Forest Classifier (balanced)
3. Weighted XGBoost Classifier
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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.data_preprocessing import build_preprocessor, prepare_train_test_data
from src.mlflow_tracking import log_experiment


class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for XGBoost multiclass classifier.
    Handles internal integer label encoding and decoding transparently,
    allowing drop-in pipeline serialization and string label prediction.
    """

    def __init__(
        self,
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight

    def fit(self, X, y):
        self.classes_ = np.array(sorted(list(set(y))))
        self.label_map_ = {c: i for i, c in enumerate(self.classes_)}
        self.inv_label_map_ = {i: c for i, c in enumerate(self.classes_)}

        y_encoded = np.array([self.label_map_[val] for val in y])

        if self.class_weight == "balanced":
            sample_weights = compute_sample_weight("balanced", y_encoded)
        else:
            sample_weights = None

        self.model_ = XGBClassifier(
            objective="multi:softprob",
            num_class=len(self.classes_),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y_encoded, sample_weight=sample_weights)
        return self

    def predict(self, X):
        preds_encoded = self.model_.predict(X)
        return np.array([self.inv_label_map_[int(i)] for i in preds_encoded])

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def evaluate_classifier(name, model, X_test, y_test, labels=None):
    """Evaluate a trained classifier and return metrics dictionary and confusion matrix."""
    if labels is None:
        labels = ["Eligible", "High_Risk", "Not_Eligible"]

    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_precision = float(
        precision_score(y_test, y_pred, average="macro", zero_division=0)
    )
    macro_recall = float(
        recall_score(y_test, y_pred, average="macro", zero_division=0)
    )
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    class_recalls = recall_score(
        y_test, y_pred, labels=labels, average=None, zero_division=0
    )
    class_precisions = precision_score(
        y_test, y_pred, labels=labels, average=None, zero_division=0
    )
    class_f1s = f1_score(
        y_test, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, lbl in enumerate(labels):
        per_class_metrics[f"{lbl.lower()}_recall"] = float(class_recalls[i])
        per_class_metrics[f"{lbl.lower()}_precision"] = float(class_precisions[i])
        per_class_metrics[f"{lbl.lower()}_f1"] = float(class_f1s[i])

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    print(f"\n{'=' * 65}")
    print(f"CLASSIFIER: {name}")
    print(f"{'=' * 65}")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Macro Precision:  {macro_precision:.4f}")
    print(f"Macro Recall:     {macro_recall:.4f}")
    print(f"Macro F1-score:   {macro_f1:.4f}")
    print(f"High-Risk Recall: {per_class_metrics.get('high_risk_recall', 0.0):.4f}")
    print(f"High-Risk F1:     {per_class_metrics.get('high_risk_f1', 0.0):.4f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "high_risk_recall": per_class_metrics.get("high_risk_recall", 0.0),
        "high_risk_precision": per_class_metrics.get("high_risk_precision", 0.0),
        "high_risk_f1": per_class_metrics.get("high_risk_f1", 0.0),
        "eligible_recall": per_class_metrics.get("eligible_recall", 0.0),
        "not_eligible_recall": per_class_metrics.get("not_eligible_recall", 0.0),
    }

    return {
        "name": name,
        "metrics": metrics,
        "confusion_matrix": cm,
        "report": report,
        "model": model,
    }


def plot_confusion_matrices(results, labels, output_path="reports/figures/classification_confusion_matrices.png"):
    """Plot side-by-side confusion matrices for all evaluated classifiers."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"{res['name']}\nMacro F1: {res['metrics']['macro_f1']:.3f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved confusion matrices figure to: {output_path}")


def train_all_classifiers(
    X_train,
    X_test,
    y_train,
    y_test,
    track_mlflow=True,
):
    """Train, evaluate, and compare Logistic Regression, Random Forest, and XGBoost."""
    labels = ["Eligible", "High_Risk", "Not_Eligible"]
    evaluations = []
    trained_pipelines = {}

    # 1. Logistic Regression
    print("\n[1/3] Training Logistic Regression...")
    log_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])
    log_pipe.fit(X_train, y_train)
    res_log = evaluate_classifier("Logistic Regression", log_pipe, X_test, y_test, labels)
    evaluations.append(res_log)
    trained_pipelines["logistic_regression"] = log_pipe

    if track_mlflow:
        log_experiment(
            model_name="Logistic Regression Classifier",
            model=log_pipe,
            metrics=res_log["metrics"],
            params={"max_iter": 1500, "class_weight": "balanced", "model_type": "LogisticRegression"},
            tags={"task": "classification", "framework": "scikit-learn"},
        )

    # 2. Random Forest
    print("\n[2/3] Training Random Forest Classifier...")
    rf_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    res_rf = evaluate_classifier("Random Forest Classifier", rf_pipe, X_test, y_test, labels)
    evaluations.append(res_rf)
    trained_pipelines["random_forest"] = rf_pipe

    if track_mlflow:
        log_experiment(
            model_name="Random Forest Classifier",
            model=rf_pipe,
            metrics=res_rf["metrics"],
            params={"n_estimators": 150, "max_depth": 18, "class_weight": "balanced"},
            tags={"task": "classification", "framework": "scikit-learn"},
        )

    # 3. Weighted XGBoost
    print("\n[3/3] Training Weighted XGBoost Classifier...")
    xgb_pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifierWrapper(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    res_xgb = evaluate_classifier("Weighted XGBoost Classifier", xgb_pipe, X_test, y_test, labels)
    evaluations.append(res_xgb)
    trained_pipelines["xgboost"] = xgb_pipe

    if track_mlflow:
        log_experiment(
            model_name="Weighted XGBoost Classifier",
            model=xgb_pipe,
            metrics=res_xgb["metrics"],
            params={
                "n_estimators": 250,
                "max_depth": 6,
                "learning_rate": 0.1,
                "class_weight": "balanced",
            },
            tags={"task": "classification", "framework": "xgboost"},
        )

    # Build comparison summary DataFrame
    summary_rows = []
    for ev in evaluations:
        m = ev["metrics"]
        summary_rows.append({
            "model": ev["name"],
            "accuracy": round(m["accuracy"], 4),
            "macro_precision": round(m["macro_precision"], 4),
            "macro_recall": round(m["macro_recall"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "high_risk_recall": round(m["high_risk_recall"], 4),
            "high_risk_precision": round(m["high_risk_precision"], 4),
            "high_risk_f1": round(m["high_risk_f1"], 4),
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(by="macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 80)
    print("CLASSIFICATION MODEL BENCHMARK COMPARISON")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Plot confusion matrices
    plot_confusion_matrices(evaluations, labels)

    # Save comparison CSV
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    summary_df.to_csv(reports_dir / "classification_model_comparison.csv", index=False)

    return trained_pipelines, evaluations, summary_df


def save_best_classifier(evaluations, output_dir="artifacts"):
    """
    Identify best classifier (prioritizing macro F1 and high-risk recall)
    and save to artifacts/.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Sort primarily by macro F1 score
    best_eval = max(
        evaluations,
        key=lambda x: (x["metrics"]["macro_f1"] * 0.7 + x["metrics"]["accuracy"] * 0.3),
    )

    pipeline_path = out_dir / "classification_pipeline.joblib"
    joblib.dump(best_eval["model"], pipeline_path)
    print(f"\n[Saved Best Classifier] '{best_eval['name']}' saved to {pipeline_path}")

    return best_eval


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate EMI eligibility classifiers.")
    parser.add_argument("--data-path", type=str, default="data/emi_prediction_dataset.csv", help="Path to CSV dataset.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample size for fast testing.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking.")
    args = parser.parse_args()

    print("Loading and preparing stratified dataset...")
    data = prepare_train_test_data(csv_path=args.data_path, sample_size=args.sample_size)

    trained_pipelines, evaluations, summary_df = train_all_classifiers(
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=data["y_class_train"],
        y_test=data["y_class_test"],
        track_mlflow=not args.no_mlflow,
    )

    save_best_classifier(evaluations)


if __name__ == "__main__":
    main()
