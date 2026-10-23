# EMIPredict AI — Intelligent Financial Risk Assessment Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-F7931E.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-PyTest%20(16%20Passed)-4EAA25.svg?logo=pytest&logoColor=white)](https://pytest.org/)


---

## 1. Project Overview

**EMIPredict AI** is an intelligent, end-to-end financial risk assessment platform designed to automate and standardize retail loan evaluation. Built using a 404,800-record financial dataset across 5 lending scenarios, the system leverages dual machine learning models to simultaneously predict applicant **EMI eligibility** (multiclass classification) and determine a customer's **maximum safe monthly EMI ceiling** in rupees (regression with non-negative guarantee).

The system integrates reproducible preprocessing, feature engineering of financial health indices, automated MLflow experiment tracking with an embedded SQLite database, a responsive fintech Streamlit dashboard, and a 100% passing automated test suite.

> **Educational & Demonstration Notice:** This platform is developed strictly for internship evaluation and academic study. It does not replace professional underwriting or provide legally binding loan approvals.

---

## 2. Problem Statement

Nowadays, retail borrowers frequently struggle to service monthly loan obligations due to inadequate financial planning and fragmented credit risk assessments. Traditional retail underwriting often relies solely on historical CIBIL/credit scores, neglecting overall living expenditure burdens, family dependency ratios, and emergency savings adequacy. This leads to loan defaults, excessive debt-to-income (DTI) burdens, and protracted manual loan approval cycles.

**EMIPredict AI solves this critical issue by providing data-driven, real-time risk insights:**
- Automates eligibility scoring across 5 major EMI lending categories.
- Computes safe maximum repayment thresholds to prevent borrower over-indebtedness.
- Delivers transparent financial health ratios (DTI, ETI, emergency fund coverage).

---

## 3. Project Objectives

1. **Dual Machine Learning Formulation:**
   - Multiclass Classification: Predict `emi_eligibility` (`Eligible`, `High_Risk`, `Not_Eligible`).
   - Continuous Regression: Estimate `max_monthly_emi` in rupees (INR ₹).
2. **Robust, Leak-Free Preprocessing:** Clean malformed numeric entries, normalize categorical inputs, and fit transformations strictly on training partitions.
3. **Domain Feature Engineering:** Derive actionable debt-to-income, expense-to-income, and affordability ratios.
4. **MLOps & Tracking:** Track model parameters, metrics, and serialized artifacts with MLflow.
5. **Interactive Fintech Dashboard:** Deliver a multi-page Streamlit application with instant demo profiles and real-time inference.
6. **Rigorous Quality Assurance:** 100% passing automated pytest test suite and executable Jupyter EDA notebooks.

---

## 4. Dataset Description & Schema Clarification

The source of truth dataset is located at `data/emi_prediction_dataset.csv`, comprising **404,800 records** and **27 columns**.

### Schema Discrepancy Note: 22 vs 25 Input Variables
Early project briefs referenced "22 Variables". Our audit clarifies that the document listed 22 demographic, income, housing, and financial history parameters under one heading, followed by **3 Loan Application Details** (`emi_scenario`, `requested_amount`, `requested_tenure`). Together with the 2 target variables, this totals **25 input features and 2 targets (27 columns total)**.

### Lending Scenarios (404,800 Balanced Records)
- **Home Appliances EMI:** 80,988 records (Amount: ₹20K–₹300K, Tenure: 6–36 months)
- **Personal Loan EMI:** 80,980 records (Amount: ₹50K–₹1,000K, Tenure: 12–60 months)
- **E-commerce Shopping EMI:** 80,948 records (Amount: ₹10K–₹200K, Tenure: 3–24 months)
- **Education EMI:** 80,942 records (Amount: ₹50K–₹500K, Tenure: 6–48 months)
- **Vehicle EMI:** 80,942 records (Amount: ₹80K–₹1,500K, Tenure: 12–84 months)

---

## 5. Input Features & Target Variables

| Feature Category | Variable Name | Type | Description |
|---|---|---|---|
| **Personal Demographics** | `age` | Numeric | Applicant age (years) |
| | `gender` | Categorical | Normalized to `Male`, `Female` |
| | `marital_status` | Categorical | `Married`, `Single` |
| | `education` | Categorical | `Graduate`, `Post Graduate`, `High School`, `Professional` |
| **Employment & Income** | `monthly_salary` | Numeric | Gross monthly salary (₹) |
| | `employment_type` | Categorical | `Private`, `Government`, `Self-employed` |
| | `company_type` | Categorical | `Large Indian`, `MNC`, `Mid-size`, `Startup`, `Small` |
| | `years_of_employment` | Numeric | Work experience duration (years) |
| **Housing & Household** | `house_type` | Categorical | `Rented`, `Own`, `Family` |
| | `monthly_rent` | Numeric | Rental expense (₹) |
| | `family_size` | Numeric | Total household members |
| | `dependents` | Numeric | Number of financial dependents |
| **Monthly Expenses** | `school_fees` | Numeric | Child schooling fees (₹) |
| | `college_fees` | Numeric | Higher education fees (₹) |
| | `travel_expenses` | Numeric | Monthly commute / transportation (₹) |
| | `groceries_utilities` | Numeric | Essential living utilities (₹) |
| | `other_monthly_expenses`| Numeric | Miscellaneous discretionary spending (₹) |
| **Financial & Credit** | `existing_loans` | Categorical | Active existing loans (`Yes`, `No`) |
| | `current_emi_amount` | Numeric | Existing monthly EMI commitments (₹) |
| | `credit_score` | Numeric | Creditworthiness score (300–850) |
| | `bank_balance` | Numeric | Liquid savings balance (₹) |
| | `emergency_fund` | Numeric | Dedicated emergency fund reserves (₹) |
| **Loan Application** | `emi_scenario` | Categorical | Lending category (5 scenarios) |
| | `requested_amount` | Numeric | Desired loan principal (₹) |
| | `requested_tenure` | Numeric | Desired repayment tenure (months) |
| **Classification Target** | `emi_eligibility` | Target | `Eligible`, `High_Risk`, `Not_Eligible` |
| **Regression Target** | `max_monthly_emi` | Target | Continuous safe EMI capacity (₹500 – ₹91,040) |

---

## 6. Project Architecture

```text
EMI-Predict.Al/
├── app.py                      # Multi-page interactive Streamlit web dashboard
├── check_dataset.py            # Diagnostic dataset integrity checker
├── requirements.txt            # Locked project dependencies
├── README.md                   # Comprehensive project documentation
├── mlflow.db                   # SQLite database for MLflow experiment tracking
├── data/
│   └── emi_prediction_dataset.csv  # 404,800 records raw financial dataset
├── src/
│   ├── data_preprocessing.py   # Cleaning, normalization, pipeline builder
│   ├── feature_engineering.py  # Financial ratios (DTI, ETI, burden ratios)
│   ├── train_classification.py # Classification benchmarking & persistence
│   ├── train_regression.py     # Regression benchmarking & non-negative bounds
│   ├── train_all.py            # End-to-end dual training orchestrator
│   ├── mlflow_tracking.py      # SQLite MLflow logging & runs query API
│   └── model_metadata.py       # Artifact loader & metadata schema manager
├── artifacts/
│   ├── classification_pipeline.joblib  # Fitted Weighted XGBoost pipeline
│   ├── regression_pipeline.joblib      # Fitted Safe XGBoost regressor pipeline
│   └── model_metadata.json             # Serialized metrics, hyperparameters & schema
├── notebooks/
│   ├── 01_data_audit.ipynb     # Interactive data audit & anomaly inspection
│   └── 02_eda.ipynb            # Visual exploratory data analysis & correlations
├── reports/
│   ├── classification_results.md           # Detailed classification log
│   ├── classification_model_comparison.csv # Classification benchmark table
│   ├── regression_model_comparison.csv     # Regression benchmark table
│   └── figures/                            # Publication-grade figures
│       ├── 1_target_distributions.png
│       ├── 2_correlation_heatmap.png
│       ├── 3_max_emi_by_scenario.png
│       ├── 4_salary_and_dti_by_eligibility.png
│       ├── classification_confusion_matrices.png
│       ├── actual_vs_predicted_emi.png
│       └── regression_residuals.png
├── tests/
│   ├── test_pipeline.py        # Automated test suite (16 passing unit/integration tests)
│   └── test_app_smoke.py       # Streamlit app syntax & compilation smoke test
└── docs/
    └── demo_script.md          # 10-minute presentation script & recording checklist
```

---

## 7. Data Preprocessing & Leakage Prevention

1. **String Artifact Resolution:** Resolved malformed numeric values such as `58.0.0` and `18000.0.0.0` by stripping repeated `.0` suffixes and replacing `nan.0` strings with nulls.
2. **Categorical Normalization:** Standardized casing variations in `gender` (`MALE`, `M`, `female`, `F`) into `Male` and `Female`.
3. **Leakage Prevention:**
   - Both target columns (`emi_eligibility`, `max_monthly_emi`) are strictly isolated prior to pipeline transformation.
   - All imputations (median for numeric, mode for categorical) and scaling (`StandardScaler`, `OneHotEncoder`) are fitted strictly on the training partition and evaluated on held-out test data.
4. **End-to-End Pipeline Serialization:** The trained preprocessor and estimator are bundled into unified scikit-learn `Pipeline` artifacts, ensuring training and inference feature ordering are identical.

---

## 8. Exploratory Data Analysis Key Findings

- **Severe Class Imbalance:** `Not_Eligible` represents 77.29% (312,868 records), `Eligible` represents 18.39% (74,444 records), and `High_Risk` represents only 4.32% (17,488 records).
- **Target Skewness:** `max_monthly_emi` is right-skewed with median ₹4,211 and maximum ₹91,040.
- **Key Determinants:** Net disposable income, bank balance, and current EMI burdens exhibit the highest correlation with maximum monthly EMI capacity.

---

## 9. Verified Model Evaluation Results

### Classification Benchmark (80/20 Stratified Split, 80,960 Test Records)

| Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1 | High-Risk Recall | High-Risk F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Weighted XGBoost (Selected)** | **0.8860** | **0.7351** | **0.8680** | **0.7479** | **0.8562** | **0.4043** |
| Random Forest (Balanced) | 0.8593 | 0.6622 | 0.7491 | 0.6804 | 0.5305 | 0.2939 |
| Logistic Regression (Balanced) | 0.7854 | 0.6478 | 0.7306 | 0.6298 | 0.6526 | 0.2290 |

*Justification:* Standard unweighted models achieve deceptively high accuracy by ignoring the 4.3% minority `High_Risk` class. Weighted XGBoost captures **85.6% of all high-risk borrowers**, making it vastly superior for credit safety.

### Regression Benchmark (80,960 Test Records)

| Regressor | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| **XGBoost Regressor (Selected)** | **₹602.62** | **₹1,013.12** | **0.9830** | **26.87%** |
| Random Forest Regressor | ₹859.43 | ₹1,448.23 | 0.9653 | 23.88% |
| Linear Regression | ₹2,584.59 | ₹3,915.17 | 0.7463 | 115.45% |

*Non-Negative Guarantee:* Embedded via `SafeRegressor` wrapper, guaranteeing all predicted monthly EMIs are finite and bounded above ₹500.

---

## 10. Installation & Virtual Environment Setup

### Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- Git (for repository version control)
- 4GB+ RAM recommended for model training and interactive dashboard

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd EMI-Predict.Al
```

#### 2. Create and Activate a Virtual Environment

- **Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- **Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

- **macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Locked Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Environment Configuration (Optional)
```bash
# If using custom port or MLflow server settings:
copy .env.example .env    # Windows PowerShell / CMD
# or: cp .env.example .env  # Linux / macOS
```

---

## 11. Execution Commands

All commands below are copy-paste ready:

### 1. Verify Dataset Integrity
```powershell
python check_dataset.py
```

### 2. Run Automated PyTest Suite (16 Passed)
```powershell
python -m pytest -v
```

### 3. Retrain Full Machine Learning Pipeline (All 6 Models)
```powershell
python -m src.train_all
```
*(Fast dry-run on 30k sample: `python -m src.train_all --sample-size 30000`)*

### 4. Launch Streamlit Web Application
```powershell
streamlit run app.py
```
*Access UI in your browser at:* `http://localhost:8501`

### 5. Launch MLflow Experiment Tracking Dashboard
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
*Access MLflow at:* `http://localhost:5000`

---

## 12. Known Limitations & Responsible AI

1. **Synthetic Training Distribution:** The platform is trained on a 404,800-record statistically simulated dataset. While calibrated to mimic realistic Indian retail borrower patterns across 5 loan categories, it does not connect to live CIBIL/Experian credit bureaus or open banking account aggregator APIs.
2. **Deterministic EMI Floor:** The regression pipeline enforces a strict ₹500 floor (`SafeRegressor`), guaranteeing mathematical non-negativity and alignment with commercial minimum micro-loan tenures.
3. **Macroeconomic Shifts:** The models evaluate static point-in-time financial indicators and do not account for external macroeconomic shifts such as sudden RBI repo rate hikes, unexpected inflation shocks, or industry-specific economic downturns.
4. **Advisory Decision Support Only:** Model outputs (`Eligible`, `High_Risk`, `Not_Eligible`) and maximum monthly EMI recommendations serve strictly as assistive decision support for credit underwriting officers. They do not constitute automated lending authorization or legally binding contracts.
5. **Cold-Start & Missing Bureau History:** For applicants with thin or zero credit file history (`credit_score` absent or default baseline), predictions rely heavily on income, expense ratios, and liquid savings; bureau inquiries should always supplement non-traditional profile scoring.
6. **Local Privacy & Zero Exfiltration:** Operates 100% locally on-premise. No applicant data, PII, or financial records are transmitted to third-party cloud servers.

---

## 13. Future Improvements

- [ ] Incorporate SHAP (SHapley Additive exPlanations) force plots inside Streamlit for individualized feature attribution.
- [ ] Implement Docker containerization and Kubernetes deployment manifests.
- [ ] Add PDF underwriting report generation for loan officers.

---

## 14. Deployment Verification & Checklist

- [x] **App runs locally without errors:** Verified running smoothly via Streamlit.
- [x] **All three customer profiles produce results:** Profile 1 (Salaried Tech Professional), Profile 2 (Freelance Designer), and Profile 3 (Overleveraged Retail Borrower) all trigger dual predictions cleanly.
- [x] **requirements.txt contains required dependencies:** Core DS, ML, visual, testing, and UI libraries properly declared with compatible versions.
- [x] **Model artifacts are included in the repository:** Pre-trained artifacts (`classification_pipeline.joblib`, `regression_pipeline.joblib`, and `model_metadata.json`) are present in `artifacts/`.
- [x] **Secrets, API keys, and private data are not committed:** Strict `.gitignore` rules in place; 0 secrets or private keys in the repository; `.env.example` template provided.
- [x] **README includes setup instructions and project limitations:** Multi-platform installation steps, execution commands, architecture, and comprehensive limitations documented.

---

**Developed for Internship Capstone Assessment • October 2026**

