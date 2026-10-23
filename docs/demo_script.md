# EMIPredict AI — 10-Minute Video Demonstration Script

**Platform:** EMIPredict AI — Intelligent Financial Risk Assessment Platform  
**Target Duration:** 10 Minutes (600 seconds)  
**Presenter:** Rahul Kiran Hosuru (Machine Learning Engineer / Data Scientist)  
**Domain:** FinTech, Banking & Credit Risk Management  

---

## 📋 Pre-Recording Setup Checklist

Before initiating screen recording, ensure the following are running and ready:

1. [ ] **Terminal 1 (Streamlit App):**
   ```powershell
   cd D:\Intenship_projects\EMI-Predict.Al
   .\.venv\Scripts\Activate.ps1
   streamlit run app.py
   ```
   *Browser tab open at:* `http://localhost:8501`

2. [ ] **Terminal 2 (MLflow Tracking UI):**
   ```powershell
   cd D:\Intenship_projects\EMI-Predict.Al
   .\.venv\Scripts\Activate.ps1
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
   ```
   *Browser tab open at:* `http://localhost:5000`

3. [ ] **IDE / Code Editor:** Open `D:\Intenship_projects\EMI-Predict.Al` displaying project tree, `src/`, `artifacts/`, and `reports/figures/`.
4. [ ] **Audio & Display:** 1080p resolution, audio test completed, notification banners silenced.

---

## ⏱️ Video Presentation Script & Timeline

### Segment 1: Introduction & Problem Statement [0:00 – 1:00] (60s)
- **Visual:** Project Title slide or Streamlit **Overview** page banner.
- **Spoken Script:**
  > "Hello everyone. Welcome to the project demonstration of **EMIPredict AI — Intelligent Financial Risk Assessment Platform**."
  > 
  > "In the retail lending and fintech domain, assessing loan affordability is often hampered by manual underwriting delays, over-reliance on single credit scores, and poor financial planning by borrowers. Unplanned debt-to-income burdens lead to loan delinquency and high default rates."
  > 
  > "EMIPredict AI addresses this challenge through a **dual machine learning architecture**:
  > 1. Predicting multi-class **EMI eligibility** (`Eligible`, `High_Risk`, or `Not_Eligible`), and
  > 2. Estimating a customer's **maximum safe monthly EMI ceiling** in rupees to prevent over-leveraging."

---

### Segment 2: Dataset Architecture & Data Quality Audit [1:00 – 2:15] (75s)
- **Visual:** Switch to `notebooks/01_data_audit.ipynb` or the `reports/figures/1_target_distributions.png`.
- **Spoken Script:**
  > "Let's examine the underlying data. The platform is trained on **404,800 verified records** across **27 columns**."
  > 
  > "An important finding during our initial data audit was resolving feature count discrepancies. The project brief previously grouped 22 personal and financial profile variables, followed by 3 loan application details: loan scenario, requested amount, and tenure. Together with the two target variables, this yields 27 columns."
  > 
  > "Our data audit uncovered several critical real-world data quality issues:
  > - Malformed numeric string artifacts in `age`, `monthly_salary`, and `bank_balance` with trailing `.0.0` sequences (e.g. `58.0.0` or `18000.0.0.0`) and string `nan.0`.
  > - Inconsistent categorical entries in `gender` (`MALE`, `M`, `female`, `F`).
  > - We engineered a strict, reproducible cleaning pipeline in `src/data_preprocessing.py` that resolves these anomalies, normalizes strings, and prevents target leakage by strictly separating training and test partitions."

---

### Segment 3: Exploratory Data Analysis & Class Imbalance [2:15 – 3:30] (75s)
- **Visual:** Display `notebooks/02_eda.ipynb` figures in `reports/figures/`:
  - `1_target_distributions.png`
  - `2_correlation_heatmap.png`
  - `4_salary_and_dti_by_eligibility.png`
- **Spoken Script:**
  > "Turning to Exploratory Data Analysis:
  > - The classification target `emi_eligibility` has **severe class imbalance**: 77.3% are `Not_Eligible`, 18.4% are `Eligible`, and only **4.3% are `High_Risk`**.
  > - If an algorithm simply guessed 'Not Eligible', it would achieve 77% accuracy while missing 100% of high-risk defaults! Therefore, optimizing **Macro F1** and **High-Risk Recall** is our primary evaluation priority.
  > - From our correlation analysis, monthly gross income and net disposable income show the highest positive correlation with allowable monthly EMI.
  > - In `src/feature_engineering.py`, we derived crucial financial health ratios including **Debt-to-Income (DTI)**, **Expense-to-Income (ETI)**, and **Emergency Fund Coverage Months**."

---

### Segment 4: Model Benchmarking & Performance Comparison [3:30 – 5:00] (90s)
- **Visual:** Show Streamlit **Model Benchmarks** tab and `reports/figures/classification_confusion_matrices.png`.
- **Spoken Script:**
  > "We developed, trained, and benchmarked three candidate algorithms for each problem:
  > 
  > **For Classification:**
  > - *Logistic Regression:* Achieved 78.5% accuracy and 0.630 Macro F1.
  > - *Random Forest:* Achieved 85.9% accuracy, but High-Risk recall was only 53.0%.
  > - ***Weighted XGBoost (Selected):*** By incorporating balanced sample weights, Weighted XGBoost achieved **88.6% overall accuracy**, a **Macro F1 of 0.748**, and captured **85.6% of minority High-Risk borrowers**, minimizing dangerous credit blindspots.
  > 
  > **For Regression:**
  > - *Linear Regression:* MAE of ₹2,584 and R² of 0.746.
  > - *Random Forest Regressor:* MAE of ₹859, RMSE of ₹1,448, R² of 0.965.
  > - ***XGBoost Regressor (Selected):*** Delivered an outstanding **R² of 0.9830**, **MAE of ₹602.62**, and **RMSE of ₹1,013.12**. Furthermore, we wrapped the model in a custom `SafeRegressor` to guarantee non-negative predictions above ₹500."

---

### Segment 5: Live Web Application Demonstration [5:00 – 7:30] (150s)
- **Visual:** Open Streamlit app at `http://localhost:8501`, navigate to **EMI Assessment** page.
- **Spoken Script:**
  > "Now, let's demonstrate the interactive Streamlit application in action."
  > 
  > *(Click '🟢 Low-Risk Salaried (Govt)' button)*
  > "First, we test a low-risk government salaried applicant earning ₹95,000/month with ₹0 existing loans and a CIBIL score of 790, requesting ₹1,20,000 for a Home Appliance loan over 24 months. Let's click 'Evaluate Financial Risk'."
  > *(Show output cards)*
  > - **Decision:** 🟢 **ELIGIBLE** (94% model confidence).
  > - **Max Monthly EMI:** Predicted safe ceiling of ~₹14,500.
  > - **Installment:** Requested ₹5,000/month is well within safe capacity.
  > 
  > *(Click '🔴 High-Risk Stretched Profile' button)*
  > "Next, let's test a financially stretched applicant: earning ₹24,000/month with ₹9,000 rent, 3 dependents, ₹6,500 existing EMI, and a 580 credit score, requesting a ₹3,00,000 vehicle loan over 24 months."
  > *(Click Evaluate)*
  > - **Decision:** 🔴 **NOT ELIGIBLE** (Debt-to-Income exceeds 50%, disposable cash flow is negative).
  > - **Requested Installment:** ₹12,500/month far exceeds their safe ceiling.
  > 
  > "Notice how our custom financial health indicators provide transparent, explainable context alongside model outputs."

---

### Segment 6: MLflow Experiment Tracking Demonstration [7:30 – 8:30] (60s)
- **Visual:** Switch to browser tab at `http://localhost:5000` (MLflow UI).
- **Spoken Script:**
  > "MLOps and reproducibility are core pillars of this project.
  > - Here in the **MLflow Tracking UI**, under experiment `EMIPredict-AI`, we can inspect every trained run.
  > - Each run logs hyperparameters such as `n_estimators`, `max_depth`, and `class_weight`, along with evaluation metrics like `macro_f1`, `high_risk_recall`, `RMSE`, and `R2`.
  > - Full fitted scikit-learn and XGBoost pipelines are tracked as artifacts.
  > - The SQLite backend provides portable, local storage directly inside the project root without external server overhead."

---

### Segment 7: Automated Testing & Reproducibility [8:30 – 9:15] (45s)
- **Visual:** Switch to terminal and run pytest.
  ```powershell
  & .\.venv\Scripts\python.exe -m pytest -v
  ```
- **Spoken Script:**
  > "To guarantee engineering reliability, we implemented an automated test suite with pytest.
  > - We run `pytest -v` across 11 unit and integration tests.
  > - Notice that all 11 tests pass in ~8 seconds: testing schema validation, numeric string cleaning, unknown categorical levels, non-negative EMI constraints, and pipeline inference."

---

### Segment 8: Limitations, Future Work & Closing [9:15 – 10:00] (45s)
- **Visual:** Switch to **Methodology & Disclaimers** tab in Streamlit.
- **Spoken Script:**
  > "In summary:
  > - **Limitations:** This project is designed as an educational and demonstration platform. Real-world lending involves bureau inquiry feeds, KYC verification, and dynamic macro-economic risk models.
  > - **Future Improvements:** In upcoming versions, we plan to incorporate SHAP tree explainability values directly in the UI, add multi-tenancy for banking branch officers, and deploy containerized microservices on Kubernetes.
  > 
  > Thank you very much for your time and attention!"

---

## 🎬 Recording Tips for Success
- **Pacing:** Speak clearly at a moderate, confident pace.
- **Mouse Movement:** Smooth cursor movements without rapid swirling.
- **Transitions:** Pause for 1 second between segment switches so cuts are seamless.
