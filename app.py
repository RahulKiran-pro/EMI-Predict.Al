"""
EMIPredict AI — Intelligent Financial Risk Assessment Platform
Streamlit Web Application Entrypoint.

Educational and internship demonstration dashboard for dual ML assessment:
1. Multiclass EMI Eligibility Prediction (Eligible / High_Risk / Not_Eligible)
2. Maximum Monthly EMI Estimation (Regression with non-negative guarantee)
"""

from datetime import datetime
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    VALID_CATEGORIES,
    clean_raw_data,
)
from src.feature_engineering import calculate_financial_indicators
from src.mlflow_tracking import get_experiment_runs
from src.model_metadata import load_artifacts, load_metadata

# -----------------------------------------------------------------------------
# App Configuration & Page Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EMIPredict AI — Financial Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (Fintech Theme)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Styling */
    .main {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.4rem;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 850px;
        line-height: 1.5;
    }
    
    /* Metric Cards */
    .fintech-card {
        background: white;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .fintech-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.08);
    }
    .card-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .card-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0f172a;
    }
    .card-subtext {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    /* Result Badges */
    .badge-eligible {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-high-risk {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-not-eligible {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    
    /* Disclaimer Box */
    .disclaimer-box {
        background-color: #f1f5f9;
        border-left: 4px solid #64748b;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Load Model Artifacts (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained ML models...")
def get_models():
    try:
        return load_artifacts()
    except Exception as e:
        return None, str(e)


artifacts_data = get_models()
if isinstance(artifacts_data, tuple) and artifacts_data[0] is None:
    models_loaded = False
    load_error_msg = artifacts_data[1]
    clf_pipeline = None
    reg_pipeline = None
    metadata = load_metadata()
else:
    models_loaded = True
    load_error_msg = None
    clf_pipeline = artifacts_data["classification_pipeline"]
    reg_pipeline = artifacts_data["regression_pipeline"]
    metadata = artifacts_data["metadata"]

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg",
        width=50,
    )
    st.title("EMIPredict AI")
    st.caption("Intelligent Financial Risk Assessment Platform")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 EMI Assessment",
            "📈 Model Benchmarks",
            "🔬 MLflow Tracking",
            "ℹ️ Methodology & Disclaimers",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Model Status")
    if models_loaded:
        st.success("🟢 Production Models Loaded")
        st.caption(f"**Classifier:** {metadata.get('classification', {}).get('selected_model', 'XGBoost')}")
        st.caption(f"**Regressor:** {metadata.get('regression', {}).get('selected_model', 'XGBoost')}")
    else:
        st.warning("🟡 Models Not Found Locally")
        st.caption("Run `python -m src.train_all` to build artifacts.")

    st.markdown("---")
    st.caption("Educational Demonstration • Not a Bank Appraisal")


# -----------------------------------------------------------------------------
# Page 1: Overview
# -----------------------------------------------------------------------------
if selected_page == "🏠 Overview":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">EMIPredict AI Platform</div>
            <div class="hero-subtitle">
                An intelligent financial risk assessment system powered by dual machine learning models. 
                Evaluates retail loan applicants across 25 demographic, income, expenditure, and credit parameters 
                to predict multiclass EMI eligibility and determine safe maximum monthly EMI thresholds.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Metric Highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="fintech-card">
                <div class="card-label">Training Dataset</div>
                <div class="card-value">404,800</div>
                <div class="card-subtext">Verified financial profiles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        acc = metadata.get("classification", {}).get("metrics", {}).get("accuracy", 0.895)
        st.markdown(
            f"""
            <div class="fintech-card">
                <div class="card-label">Classification Acc.</div>
                <div class="card-value">{acc * 100:.1f}%</div>
                <div class="card-subtext">3-Class Eligibility Model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        rmse = metadata.get("regression", {}).get("metrics", {}).get("RMSE", 1014.81)
        st.markdown(
            f"""
            <div class="fintech-card">
                <div class="card-label">Regression RMSE</div>
                <div class="card-value">₹{rmse:,.0f}</div>
                <div class="card-subtext">Max Monthly EMI Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        r2 = metadata.get("regression", {}).get("metrics", {}).get("R2", 0.9826)
        st.markdown(
            f"""
            <div class="fintech-card">
                <div class="card-label">Regression R²</div>
                <div class="card-value">{r2:.4f}</div>
                <div class="card-subtext">Explained Variance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔄 End-to-End System Workflow")
    st.markdown(
        """
        ```mermaid
        graph LR
            A[Customer Profile & Loan Request] --> B[Data Cleaning & Normalization]
            B --> C[Feature Engineering: DTI, ETI, Burden Ratios]
            C --> D[StandardScaler & OneHotEncoder Pipeline]
            D --> E[Multiclass Classifier: XGBoost / RF]
            D --> F[EMI Regressor: XGBoost / SafeRegressor]
            E --> G[Eligibility Decision: Eligible / High Risk / Not Eligible]
            F --> H[Safe Max Monthly EMI Ceiling in ₹]
            G --> I[Unified Risk Assessment Report]
            H --> I
        ```
        """
    )

    st.markdown("### 🎯 Dual ML Problem Formulation")
    c1, c2 = st.columns(2)
    with c1:
        st.info("#### 1. Multiclass Classification — EMI Eligibility")
        st.markdown(
            """
            - **Target:** `emi_eligibility`
            - **Classes:**
              - 🟢 **Eligible:** Comfortable disposable income and healthy credit standing.
              - 🟡 **High_Risk:** Marginal affordability; requires higher collateral or stricter terms.
              - 🔴 **Not_Eligible:** Overleveraged debt burden or insufficient disposable cash flow.
            - **Class Imbalance:** Handled using balanced class weighting (`compute_sample_weight`).
            """
        )
    with c2:
        st.success("#### 2. Regression — Maximum Monthly EMI")
        st.markdown(
            """
            - **Target:** `max_monthly_emi` (Continuous ₹ amount)
            - **Financial Bounds:** Minimum ₹500, constrained by customer's disposable income and credit standing.
            - **Non-Negative Guarantee:** Enforced via `SafeRegressor` wrapper so outputs never produce negative figures.
            - **Leading Algorithm:** XGBoost Regressor achieving R² > 0.98 and MAE < ₹650.
            """
        )

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Educational Disclaimer:</strong> This application was developed as an educational internship project 
            for demonstration and study of end-to-end Machine Learning and MLOps. It is not an official banking underwriting tool 
            and does not represent a binding financial approval decision.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Page 2: EMI Assessment (Form & Real-Time Predictions)
# -----------------------------------------------------------------------------
elif selected_page == "📊 EMI Assessment":
    st.title("💳 Interactive Customer EMI Assessment")
    st.markdown("Enter applicant details to evaluate EMI eligibility and compute maximum safe installment capacity.")

    # Documented default profile
    DEFAULT_ASSESSMENT_PROFILE = {
        "age": 35, "gender": "Male", "marital_status": "Married", "education": "Graduate",
        "monthly_salary": 55000.0, "employment_type": "Private", "company_type": "Large Indian",
        "years_of_employment": 5.0, "house_type": "Rented", "monthly_rent": 12000.0, "family_size": 3,
        "dependents": 1, "school_fees": 2000.0, "college_fees": 0.0, "travel_expenses": 3500.0,
        "groceries_utilities": 12000.0, "other_monthly_expenses": 5000.0, "existing_loans": "No",
        "current_emi_amount": 0.0, "credit_score": 730.0, "bank_balance": 150000.0,
        "emergency_fund": 60000.0, "emi_scenario": "Personal Loan EMI",
        "requested_amount": 200000.0, "requested_tenure": 24,
    }

    # Documented test & demo presets
    PRESET_PROFILES = {
        "low_risk": {
            "age": 31, "gender": "Male", "marital_status": "Single", "education": "Graduate",
            "monthly_salary": 65000.0, "employment_type": "Private", "company_type": "MNC",
            "years_of_employment": 6.0, "house_type": "Rented", "monthly_rent": 16000.0, "family_size": 2,
            "dependents": 1, "school_fees": 0.0, "college_fees": 0.0, "travel_expenses": 5500.0,
            "groceries_utilities": 14000.0, "other_monthly_expenses": 7500.0, "existing_loans": "Yes",
            "current_emi_amount": 8500.0, "credit_score": 710.0, "bank_balance": 180000.0,
            "emergency_fund": 75000.0, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 350000.0, "requested_tenure": 36,
        },
        "moderate": {
            "age": 28, "gender": "Male", "marital_status": "Single", "education": "Graduate",
            "monthly_salary": 45000.0, "employment_type": "Private", "company_type": "Mid-size",
            "years_of_employment": 4.0, "house_type": "Rented", "monthly_rent": 12000.0, "family_size": 1,
            "dependents": 0, "school_fees": 0.0, "college_fees": 0.0, "travel_expenses": 3000.0,
            "groceries_utilities": 9000.0, "other_monthly_expenses": 4000.0, "existing_loans": "No",
            "current_emi_amount": 0.0, "credit_score": 680.0, "bank_balance": 95000.0,
            "emergency_fund": 35000.0, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 150000.0, "requested_tenure": 18,
        },
        "high_risk": {
            "age": 36, "gender": "Male", "marital_status": "Married", "education": "High School",
            "monthly_salary": 32000.0, "employment_type": "Self-employed", "company_type": "Small",
            "years_of_employment": 8.0, "house_type": "Rented", "monthly_rent": 10000.0, "family_size": 5,
            "dependents": 3, "school_fees": 4000.0, "college_fees": 0.0, "travel_expenses": 2500.0,
            "groceries_utilities": 8500.0, "other_monthly_expenses": 2000.0, "existing_loans": "Yes",
            "current_emi_amount": 9000.0, "credit_score": 590.0, "bank_balance": 15000.0,
            "emergency_fund": 5000.0, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 200000.0, "requested_tenure": 24,
        },
    }

    # Centralized helper to set session state from profile dictionary
    def set_form_profile(target_dict: dict):
        for k, v in target_dict.items():
            st.session_state[f"w_{k}"] = v

    # Initialize session state with default profile values if not present
    for k, v in DEFAULT_ASSESSMENT_PROFILE.items():
        if f"w_{k}" not in st.session_state:
            st.session_state[f"w_{k}"] = v

    # Quick pre-fill profiles for testing convenience
    st.markdown("#### ⚡ Quick Demo Profiles")
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

    with preset_col1:
        if st.button("🟢 Low-Risk Salaried"):
            set_form_profile(PRESET_PROFILES["low_risk"])
            st.rerun()
    with preset_col2:
        if st.button("🟡 Moderate Professional"):
            set_form_profile(PRESET_PROFILES["moderate"])
            st.rerun()
    with preset_col3:
        if st.button("🔴 High-Risk Stretched"):
            set_form_profile(PRESET_PROFILES["high_risk"])
            st.rerun()
    with preset_col4:
        if st.button("🔄 Reset to Default"):
            set_form_profile(DEFAULT_ASSESSMENT_PROFILE)
            st.rerun()

    with st.form("emi_assessment_form"):
        st.markdown("#### 1. Personal & Demographic Profile")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=75, key="w_age")
            gender = st.selectbox("Gender", VALID_CATEGORIES["gender"], key="w_gender")
        with col2:
            marital_status = st.selectbox("Marital Status", VALID_CATEGORIES["marital_status"], key="w_marital_status")
            education = st.selectbox("Education Level", VALID_CATEGORIES["education"], key="w_education")
        with col3:
            house_type = st.selectbox("Housing Type", VALID_CATEGORIES["house_type"], key="w_house_type")
            family_size = st.number_input("Total Household Members", min_value=1, max_value=15, step=1, key="w_family_size")
        with col4:
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=12, step=1, key="w_dependents")

        st.markdown("#### 2. Employment & Income")
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        with ecol1:
            monthly_salary = st.number_input("Gross Monthly Salary (₹)", min_value=5000.0, max_value=1000000.0, step=1000.0, key="w_monthly_salary")
        with ecol2:
            employment_type = st.selectbox("Employment Sector", VALID_CATEGORIES["employment_type"], key="w_employment_type")
        with ecol3:
            company_type = st.selectbox("Company Type", VALID_CATEGORIES["company_type"], key="w_company_type")
        with ecol4:
            years_of_employment = st.number_input("Total Work Experience (Years)", min_value=0.0, max_value=50.0, step=0.5, key="w_years_of_employment")

        st.markdown("#### 3. Monthly Financial Outflows & Expenses")
        xcol1, xcol2, xcol3 = st.columns(3)
        with xcol1:
            monthly_rent = st.number_input("Monthly Rent Expense (₹)", min_value=0.0, max_value=200000.0, step=500.0, key="w_monthly_rent")
            school_fees = st.number_input("School Fees (₹)", min_value=0.0, max_value=100000.0, step=500.0, key="w_school_fees")
        with xcol2:
            college_fees = st.number_input("College / Higher Education Fees (₹)", min_value=0.0, max_value=200000.0, step=500.0, key="w_college_fees")
            travel_expenses = st.number_input("Commute & Travel Expenses (₹)", min_value=0.0, max_value=100000.0, step=500.0, key="w_travel_expenses")
        with xcol3:
            groceries_utilities = st.number_input("Groceries & Utility Bills (₹)", min_value=0.0, max_value=200000.0, step=500.0, key="w_groceries_utilities")
            other_monthly_expenses = st.number_input("Other Discretionary Expenses (₹)", min_value=0.0, max_value=100000.0, step=500.0, key="w_other_monthly_expenses")

        st.markdown("#### 4. Existing Commitments & Credit Standing")
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            existing_loans = st.selectbox("Existing Active Loans?", VALID_CATEGORIES["existing_loans"], key="w_existing_loans")
            current_emi_amount = st.number_input("Current Active Monthly EMI (₹)", min_value=0.0, max_value=300000.0, step=500.0, key="w_current_emi_amount")
        with fcol2:
            credit_score = st.number_input("Credit Score (CIBIL 300 - 850)", min_value=300.0, max_value=850.0, step=5.0, key="w_credit_score")
        with fcol3:
            bank_balance = st.number_input("Current Savings Bank Balance (₹)", min_value=0.0, max_value=10000000.0, step=5000.0, key="w_bank_balance")
        with fcol4:
            emergency_fund = st.number_input("Dedicated Emergency Fund (₹)", min_value=0.0, max_value=10000000.0, step=5000.0, key="w_emergency_fund")

        st.markdown("#### 5. Loan Application Details")
        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            emi_scenario = st.selectbox("Loan Scenario", VALID_CATEGORIES["emi_scenario"], key="w_emi_scenario")
        with lcol2:
            requested_amount = st.number_input("Requested Loan Principal (₹)", min_value=5000.0, max_value=5000000.0, step=10000.0, key="w_requested_amount")
        with lcol3:
            requested_tenure = st.number_input("Requested Repayment Tenure (Months)", min_value=3, max_value=84, step=1, key="w_requested_tenure")

        submit_btn = st.form_submit_button("⚡ Evaluate Financial Risk & Predict EMI", type="primary", use_container_width=True)

    if submit_btn:
        input_dict = {
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": emi_scenario,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
        }

        # Calculate derived financial health ratios
        fin = calculate_financial_indicators(input_dict)
        input_df = pd.DataFrame([input_dict])

        st.markdown("---")
        st.subheader("📋 Assessment & Prediction Results")

        if models_loaded:
            with st.spinner("Analyzing profile with trained ML models..."):
                try:
                    # Multiclass classification prediction & probability
                    eligibility_pred = clf_pipeline.predict(input_df)[0]
                    proba_classes = getattr(clf_pipeline.named_steps["classifier"], "classes_", ["Eligible", "High_Risk", "Not_Eligible"])
                    probabilities = clf_pipeline.predict_proba(input_df)[0]
                    prob_dict = dict(zip(proba_classes, probabilities))

                    # Regression prediction
                    pred_max_emi = float(reg_pipeline.predict(input_df)[0])
                except Exception as pred_err:
                    st.error(f"Prediction error: {pred_err}")
                    eligibility_pred = "High_Risk"
                    prob_dict = {"Eligible": 0.33, "High_Risk": 0.34, "Not_Eligible": 0.33}
                    pred_max_emi = max(500.0, fin["disposable_income"] * 0.4)
        else:
            st.warning("⚠️ Using fallback heuristic assessment because serialized models are not yet compiled.")
            if fin["disposable_income"] <= 0 or fin["debt_to_income_ratio"] > 45:
                eligibility_pred = "Not_Eligible"
            elif fin["debt_to_income_ratio"] > 30 or credit_score < 650:
                eligibility_pred = "High_Risk"
            else:
                eligibility_pred = "Eligible"
            prob_dict = {"Eligible": 0.65, "High_Risk": 0.25, "Not_Eligible": 0.10}
            pred_max_emi = max(500.0, fin["disposable_income"] * 0.45)

        # Display Top Cards
        r_col1, r_col2, r_col3 = st.columns(3)

        with r_col1:
            st.markdown("<div class='card-label'>EMI Eligibility Decision</div>", unsafe_allow_html=True)
            if eligibility_pred == "Eligible":
                st.markdown("<div class='badge-eligible'>🟢 ELIGIBLE</div>", unsafe_allow_html=True)
                st.caption("Low financial risk; strong debt service capability.")
            elif eligibility_pred == "High_Risk":
                st.markdown("<div class='badge-high-risk'>🟡 HIGH RISK</div>", unsafe_allow_html=True)
                st.caption("Marginal risk tier; loan requires higher collateral or lower quantum.")
            else:
                st.markdown("<div class='badge-not-eligible'>🔴 NOT ELIGIBLE</div>", unsafe_allow_html=True)
                st.caption("High leverage or debt-burden exceed safe thresholds.")

        with r_col2:
            st.markdown("<div class='card-label'>Estimated Safe Monthly EMI Ceiling</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-value' style='color:#0284c7;'>₹{pred_max_emi:,.2f}</div>", unsafe_allow_html=True)
            st.caption(f"Estimated maximum safe monthly repayment without default risk.")

        with r_col3:
            req_installment = fin["requested_monthly_installment"]
            st.markdown("<div class='card-label'>Requested Monthly Installment</div>", unsafe_allow_html=True)
            if req_installment <= pred_max_emi:
                color = "#059669"
                status_text = "✅ Within Safe Capacity"
            else:
                color = "#dc2626"
                status_text = "⚠️ Exceeds Safe Capacity"
            st.markdown(f"<div class='card-value' style='color:{color};'>₹{req_installment:,.2f}</div>", unsafe_allow_html=True)
            st.caption(f"Principal / Tenure: **{status_text}** *(Zero-Interest Straight-Line Estimate)*")

        # Class Probability Breakdown & Financial Ratios
        st.markdown("#### 🔍 Probability Distribution & Financial Indicators")
        pcol1, pcol2 = st.columns([1, 1])

        with pcol1:
            st.markdown("**Eligibility Model Confidence Breakdown**")
            prob_df = pd.DataFrame({
                "Classification Tier": list(prob_dict.keys()),
                "Model Probability": [f"{v * 100:.1f}%" for v in prob_dict.values()],
                "Confidence Score": list(prob_dict.values()),
            })
            st.bar_chart(prob_df.set_index("Classification Tier")["Confidence Score"], color="#0284c7")
            st.dataframe(prob_df[["Classification Tier", "Model Probability"]].set_index("Classification Tier"), use_container_width=True)

        with pcol2:
            st.markdown("**Key Derived Financial Ratios**")
            ratio_df = pd.DataFrame([
                {"Indicator": "Net Disposable Income", "Value": f"₹{fin['disposable_income']:,.2f}", "Benchmark": "> ₹15,000"},
                {"Indicator": "Current Debt-to-Income (DTI)", "Value": f"{fin['debt_to_income_ratio']:.1f}%", "Benchmark": "< 35%"},
                {"Indicator": "Projected Total DTI (with loan)", "Value": f"{fin['projected_total_dti']:.1f}%", "Benchmark": "< 50%"},
                {"Indicator": "Expense-to-Income (ETI)", "Value": f"{fin['expense_to_income_ratio']:.1f}%", "Benchmark": "< 50%"},
                {"Indicator": "Emergency Fund Coverage", "Value": f"{fin['emergency_fund_months']:.1f} months", "Benchmark": "≥ 3.0 months"},
            ])
            st.table(ratio_df.set_index("Indicator"))

        st.info(
            f"💡 **Assessment Summary:** For a requested loan of **₹{requested_amount:,.0f}** over **{requested_tenure} months** in the **{emi_scenario}** scenario, "
            f"the zero-interest estimated installment is **₹{req_installment:,.2f}**. "
            + ("This is fully within the estimated safe ceiling of ₹{:,.2f}.".format(pred_max_emi) if req_installment <= pred_max_emi else "This exceeds the model-estimated safe maximum of ₹{:,.2f} by ₹{:,.2f}.".format(pred_max_emi, req_installment - pred_max_emi))
        )


# -----------------------------------------------------------------------------
# Page 3: Model Performance & Benchmarks
# -----------------------------------------------------------------------------
elif selected_page == "📈 Model Benchmarks":
    st.title("📈 Model Evaluation & Benchmarks")
    st.markdown("Rigorous empirical evaluation comparing candidate algorithms on held-out test data.")

    tab1, tab2 = st.tabs(["Classification Benchmarks", "Regression Benchmarks"])

    with tab1:
        st.subheader("1. Classification Models Comparison (EMI Eligibility)")
        clf_csv = Path("reports/classification_model_comparison.csv")
        if clf_csv.exists():
            clf_df = pd.read_csv(clf_csv)
            st.dataframe(clf_df, use_container_width=True)
        else:
            st.dataframe(pd.DataFrame(metadata.get("classification", {}).get("all_models", [])), use_container_width=True)

        st.markdown("#### 🎯 Model Selection Justification: Weighted XGBoost")
        st.markdown(
            """
            - **Severe Class Imbalance:** In the 404,800-record dataset, `High_Risk` comprises only ~4.3% of records.
            - Standard Random Forest yields high overall accuracy (88-90%), but severely fails on minority High_Risk recall (~5-30%), 
              leading to dangerous false negatives in credit assessment.
            - **Weighted XGBoost** optimizes both overall Macro F1 and minority High_Risk recall (exceeding 85% on full dataset), 
              making it the preferred classifier for risk-sensitive lending.
            """
        )

        cm_fig = Path("reports/figures/classification_confusion_matrices.png")
        if cm_fig.exists():
            st.markdown("#### 📊 Confusion Matrices")
            st.image(str(cm_fig), caption="Confusion Matrices across Evaluated Classifiers", use_container_width=True)

    with tab2:
        st.subheader("2. Regression Models Comparison (Max Monthly EMI)")
        reg_csv = Path("reports/regression_model_comparison.csv")
        if reg_csv.exists():
            reg_df = pd.read_csv(reg_csv)
            st.dataframe(reg_df, use_container_width=True)
        else:
            st.dataframe(pd.DataFrame(metadata.get("regression", {}).get("all_models", [])), use_container_width=True)

        st.markdown("#### 📊 Regression Diagnostic Visualizations")
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            act_pred_img = Path("reports/figures/actual_vs_predicted_emi.png")
            if act_pred_img.exists():
                st.image(str(act_pred_img), caption="Actual vs Predicted Maximum Monthly EMI (₹)", use_container_width=True)
        with rcol2:
            res_img = Path("reports/figures/regression_residuals.png")
            if res_img.exists():
                st.image(str(res_img), caption="Residual Distribution (Actual - Predicted)", use_container_width=True)


# -----------------------------------------------------------------------------
# Page 4: MLflow Tracking
# -----------------------------------------------------------------------------
elif selected_page == "🔬 MLflow Tracking":
    st.title("🔬 MLflow Experiment Tracking")
    st.markdown("All model runs, hyperparameters, metrics, and serialized artifacts are tracked in the local SQLite MLflow registry.")

    st.markdown("#### 🚀 Launch Local MLflow Tracking UI")
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000", language="powershell")
    st.caption("Run the above command in your terminal and open [http://localhost:5000](http://localhost:5000) in your browser.")

    st.markdown("#### 📋 Tracked Experiment Runs in `EMIPredict-AI`")
    runs_df = get_experiment_runs(limit=25)
    if runs_df is not None and not runs_df.empty:
        st.dataframe(runs_df, use_container_width=True)
    else:
        st.info("No runs logged yet or MLflow database initializing. Run `python -m src.train_all` to track experiments.")


# -----------------------------------------------------------------------------
# Page 5: About & Disclaimers
# -----------------------------------------------------------------------------
elif selected_page == "ℹ️ Methodology & Disclaimers":
    st.title("ℹ️ Project Architecture & Financial Disclaimers")

    st.markdown("### 📚 Dataset Architecture & Discrepancy Note")
    st.markdown(
        """
        - **Source of Truth:** `data/emi_prediction_dataset.csv` contains **404,800 records** and **27 columns**.
        - **Feature Breakdown:**
          - **22 Personal & Financial Profile Variables:** Demographics, income, housing, family, obligations, credit, and savings.
          - **3 Loan Application Variables:** `emi_scenario`, `requested_amount`, `requested_tenure`.
          - **2 Target Variables:** `emi_eligibility` (classification) and `max_monthly_emi` (regression).
        - **Schema Clarification:** Early project documentation occasionally referenced "22 Variables" because the 3 loan scenario inputs 
          were listed under a secondary section. The actual machine learning pipeline leverages all 25 non-target variables.
        """
    )

    st.markdown("### 🛡️ Ethical & Responsible AI Considerations")
    st.markdown(
        """
        1. **Educational Scope:** Built strictly for academic and demonstration purposes.
        2. **No Guaranteed Approvals:** Automated risk scoring cannot substitute licensed underwriting guidelines.
        3. **Target Leakage Prevention:** Both targets (`emi_eligibility` and `max_monthly_emi`) are strictly excluded from input matrices.
        4. **Data Privacy:** Local execution ensures sensitive financial inputs are processed on-premise without external API transmission.
        """
    )

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Mandatory Legal Disclaimer:</strong> The predictions generated by EMIPredict AI are statistical estimates 
            based on synthetic historical data patterns. They do not constitute formal lending agreements, debt counseling, or regulatory credit decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )
