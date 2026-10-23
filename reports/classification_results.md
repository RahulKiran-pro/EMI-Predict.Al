# EMIPredict AI — Classification Experiment & Model Comparison Log

## Executive Summary

The classification objective predicts applicant **EMI eligibility** (`Eligible`, `High_Risk`, `Not_Eligible`) from 25 demographic and financial input features. The dataset of 404,800 records exhibits severe class imbalance:
- `Not_Eligible`: 77.29% (312,868 records)
- `Eligible`: 18.39% (74,444 records)
- `High_Risk`: 4.32% (17,488 records)

Evaluating on overall accuracy alone is deceptive, as a naive majority baseline achieves ~77.3% accuracy while failing on 100% of high-risk defaults. Our evaluation therefore prioritizes **Macro F1-score** and **High-Risk Class Recall**.

---

## Benchmark Results (80/20 Stratified Split, 80,960 Test Records)

| Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1 | High-Risk Recall | High-Risk Precision | High-Risk F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Weighted XGBoost** | **0.8860** | **0.7351** | **0.8680** | **0.7479** | **0.8562** | **0.2647** | **0.4043** |
| Random Forest (Balanced) | 0.8593 | 0.6622 | 0.7491 | 0.6804 | 0.5305 | 0.2033 | 0.2939 |
| Logistic Regression (Balanced) | 0.7854 | 0.6478 | 0.7306 | 0.6298 | 0.6526 | 0.1389 | 0.2290 |

---

## Detailed Model Observations

### 1. Weighted XGBoost Classifier (Selected Production Model)
- **Hyperparameters:** `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `objective='multi:softprob'`.
- **Sample Weighting:** Computed using balanced class inverse-frequency weights on training data.
- **Findings:**
  - Achieves the highest Macro F1 (0.7479) and Macro Recall (0.8680).
  - Captures **85.6% of minority High_Risk profiles** (2,994 out of 3,497 test instances).
  - False negatives for `High_Risk` (profiles predicted as `Eligible` when actually `High_Risk`) are minimized to only 387 cases out of 80,960 applicants.
- **Artifact:** Serialized into `artifacts/classification_pipeline.joblib`.

### 2. Random Forest Classifier
- **Hyperparameters:** `n_estimators=100`, `max_depth=16`, `min_samples_leaf=4`, `class_weight='balanced'`.
- **Findings:**
  - Strong overall accuracy (85.93%) and solid `Eligible` F1 (0.82).
  - High-Risk recall drops to 53.05%, missing nearly half of the precarious borrowers.

### 3. Logistic Regression
- **Hyperparameters:** `max_iter=1500`, `class_weight='balanced'`, `solver='lbfgs'`.
- **Findings:**
  - Fast, interpretable linear baseline.
  - Overall accuracy is 78.54% and Macro F1 is 0.6298.
  - While balanced weights help boost high-risk recall to 65.26%, precision is low (13.89%), yielding excessive false alarms.

---

## Artifacts & Visuals
- Comparison Table: `reports/classification_model_comparison.csv`
- Confusion Matrices: `reports/figures/classification_confusion_matrices.png`
- MLflow Experiment: Tracked under experiment name `EMIPredict-AI` with parameters, metrics, and models.