"""
Automated Test Suite for EMIPredict AI.

Validates:
- Data cleaning & malformed numeric string resolution
- Categorical normalization
- Feature engineering financial indicators
- Preprocessor transformations and unknown category handling
- Classification model inference & probability bounds
- Regression model inference & non-negative EMI constraint
- Model artifacts loading & metadata schema
- MLflow experiment tracking
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    VALID_CATEGORIES,
    build_preprocessor,
    clean_numeric_value,
    clean_raw_data,
    normalize_gender,
)
from src.feature_engineering import (
    DEFAULT_FINANCIAL_INPUTS,
    calculate_financial_indicators,
    calculate_standard_emi,
)
from src.mlflow_tracking import get_experiment_runs, log_experiment
from src.model_metadata import load_artifacts, load_metadata


@pytest.fixture
def sample_applicant_dict():
    """Valid synthetic applicant record matching dataset schema."""
    return {
        "age": 35.0,
        "gender": "Male",
        "marital_status": "Married",
        "education": "Graduate",
        "monthly_salary": 60000.0,
        "employment_type": "Private",
        "years_of_employment": 7.0,
        "company_type": "MNC",
        "house_type": "Rented",
        "monthly_rent": 14000.0,
        "family_size": 3,
        "dependents": 1,
        "school_fees": 3000.0,
        "college_fees": 0.0,
        "travel_expenses": 4000.0,
        "groceries_utilities": 12000.0,
        "other_monthly_expenses": 5000.0,
        "existing_loans": "No",
        "current_emi_amount": 0.0,
        "credit_score": 750.0,
        "bank_balance": 200000.0,
        "emergency_fund": 90000.0,
        "emi_scenario": "Personal Loan EMI",
        "requested_amount": 250000.0,
        "requested_tenure": 24,
    }


def test_numeric_cleaning_artifacts():
    """Verify malformed numeric strings with repeated .0 or nan.0 are resolved."""
    assert clean_numeric_value("58.0.0") == 58.0
    assert clean_numeric_value("18000.0.0.0") == 18000.0
    assert clean_numeric_value("32.0") == 32.0
    assert pd.isna(clean_numeric_value("nan.0"))
    assert pd.isna(clean_numeric_value("null"))
    assert pd.isna(clean_numeric_value(np.nan))


def test_gender_normalization():
    """Verify gender abbreviations and casing are standardized."""
    assert normalize_gender("MALE") == "Male"
    assert normalize_gender("m") == "Male"
    assert normalize_gender("female") == "Female"
    assert normalize_gender("F") == "Female"
    assert normalize_gender("Male") == "Male"


def test_clean_raw_data_dataframe():
    """Verify DataFrame batch cleaning on noisy inputs."""
    noisy_df = pd.DataFrame([
        {
            "age": "45.0.0",
            "gender": "M",
            "monthly_salary": "75000.0.0",
            "bank_balance": "nan.0",
            "credit_score": 720.0,
        }
    ])
    cleaned = clean_raw_data(noisy_df)
    assert cleaned.loc[0, "age"] == 45.0
    assert cleaned.loc[0, "gender"] == "Male"
    assert cleaned.loc[0, "monthly_salary"] == 75000.0
    assert pd.isna(cleaned.loc[0, "bank_balance"])


def test_feature_engineering_ratios(sample_applicant_dict):
    """Verify financial indicators calculation (DTI, ETI, disposable income)."""
    fin = calculate_financial_indicators(sample_applicant_dict)

    # total_monthly_expenses: 14000 + 3000 + 0 + 4000 + 12000 + 5000 = 38000
    assert fin["total_monthly_expenses"] == 38000.0
    # disposable_income: 60000 - 38000 - 0 = 22000
    assert fin["disposable_income"] == 22000.0
    # DTI: current_emi (0) / salary = 0%
    assert fin["debt_to_income_ratio"] == 0.0
    # ETI: 38000 / 60000 = ~63.33%
    assert abs(fin["expense_to_income_ratio"] - 63.33) < 0.1
    # Requested monthly installment: 250000 / 24 = ~10416.67
    assert abs(fin["requested_monthly_installment"] - 10416.67) < 1.0


def test_preprocessor_fitting_and_unknown_categories():
    """Verify preprocessing pipeline handles missing values and unknown categories."""
    preprocessor = build_preprocessor()

    # Synthetic training dataframe
    train_data = pd.DataFrame({
        col: [30.0, 40.0, np.nan] for col in NUMERICAL_COLUMNS
    })
    for col in CATEGORICAL_COLUMNS:
        train_data[col] = ["Male", "Female", None]

    preprocessor.fit(train_data)

    # Test with unseen category values
    test_data = pd.DataFrame({
        col: [35.0] for col in NUMERICAL_COLUMNS
    })
    for col in CATEGORICAL_COLUMNS:
        test_data[col] = ["UNSEEN_CATEGORY_XYZ"]

    transformed = preprocessor.transform(test_data)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_saved_model_artifacts_and_metadata():
    """Verify saved production pipelines and metadata exist and are valid."""
    artifacts = load_artifacts()
    assert artifacts["classification_pipeline"] is not None
    assert artifacts["regression_pipeline"] is not None

    meta = artifacts["metadata"]
    assert "classification" in meta
    assert "regression" in meta
    assert meta["dataset_records"] > 0
    assert "numerical_features" in meta
    assert "categorical_features" in meta


def test_classification_single_record_inference(sample_applicant_dict):
    """Verify classification pipeline generates valid class and probabilities."""
    artifacts = load_artifacts()
    clf = artifacts["classification_pipeline"]

    df = pd.DataFrame([sample_applicant_dict])
    pred = clf.predict(df)[0]
    assert pred in ["Eligible", "High_Risk", "Not_Eligible"]

    probas = clf.predict_proba(df)[0]
    assert len(probas) == 3
    assert abs(sum(probas) - 1.0) < 1e-4
    assert all(0.0 <= p <= 1.0 for p in probas)


def test_regression_single_record_inference(sample_applicant_dict):
    """Verify regression pipeline predicts non-negative finite monthly EMI."""
    artifacts = load_artifacts()
    reg = artifacts["regression_pipeline"]

    df = pd.DataFrame([sample_applicant_dict])
    pred = reg.predict(df)[0]

    assert np.isfinite(pred)
    assert pred >= 500.0  # SafeRegressor minimum non-negative constraint


def test_regression_edge_case_negative_profile(sample_applicant_dict):
    """Verify SafeRegressor bounds extreme negative capacity profiles at >= Rs. 500."""
    artifacts = load_artifacts()
    reg = artifacts["regression_pipeline"]

    # Extreme over-burdened profile
    stretched = sample_applicant_dict.copy()
    stretched["monthly_salary"] = 1000.0
    stretched["current_emi_amount"] = 80000.0
    stretched["monthly_rent"] = 50000.0

    df = pd.DataFrame([stretched])
    pred = reg.predict(df)[0]

    assert np.isfinite(pred)
    assert pred >= 500.0


def test_mlflow_logging_smoke():
    """Verify MLflow log_experiment executes and registers a run."""
    from sklearn.linear_model import LinearRegression
    dummy = LinearRegression().fit([[1], [2]], [1, 2])
    run_id = log_experiment(
        model_name="PyTest Smoke Run",
        model=dummy,
        metrics={"test_metric": 1.23},
        params={"test_param": "val"},
        tags={"source": "pytest"},
    )
    assert run_id is not None


def test_financial_indicators_missing_dict_keys():
    """Verify calculate_financial_indicators gracefully handles empty/partial dicts without crashing."""
    empty_result = calculate_financial_indicators({})
    assert isinstance(empty_result, dict)
    assert empty_result["total_monthly_expenses"] == 0.0
    assert empty_result["disposable_income"] == 0.0
    assert empty_result["debt_to_income_ratio"] == 0.0
    assert empty_result["requested_monthly_installment"] == 0.0

    partial_result = calculate_financial_indicators({
        "monthly_salary": 50000.0,
        "monthly_rent": 15000.0,
        "current_emi_amount": 5000.0,
    })
    assert partial_result["total_monthly_expenses"] == 15000.0
    assert partial_result["disposable_income"] == 30000.0
    assert abs(partial_result["debt_to_income_ratio"] - 10.0) < 1e-3


def test_standard_emi_formula():
    """Verify standard reducing-balance EMI formula and zero-interest boundary conditions."""
    # Zero interest rate should equal straight-line amortization
    assert calculate_standard_emi(120000.0, 0.0, 12) == 10000.0
    # Zero or negative tenure should safely return 0.0
    assert calculate_standard_emi(120000.0, 12.0, 0) == 0.0
    # Standard 12% annual interest on 100,000 for 12 months (monthly r = 0.01)
    # EMI = 100000 * 0.01 * (1.01)^12 / ((1.01)^12 - 1) = 8884.8788
    emi = calculate_standard_emi(100000.0, 12.0, 12)
    assert abs(emi - 8884.88) < 0.05


def test_exact_user_profiles_math():
    """Verify exact financial indicators math across the 3 audited customer profiles."""
    # Profile 1: Low-Risk Salaried Customer
    p1 = {
        "monthly_salary": 65000.0,
        "monthly_rent": 16000.0,
        "school_fees": 0.0,
        "college_fees": 0.0,
        "groceries_utilities": 14000.0,
        "travel_expenses": 5500.0,
        "other_monthly_expenses": 7500.0,
        "current_emi_amount": 8500.0,
        "emergency_fund": 75000.0,
        "requested_amount": 350000.0,
        "requested_tenure": 36,
    }
    fin1 = calculate_financial_indicators(p1)
    assert fin1["total_monthly_expenses"] == 43000.0
    assert fin1["disposable_income"] == 13500.0
    assert round(fin1["debt_to_income_ratio"], 1) == 13.1
    assert round(fin1["expense_to_income_ratio"], 1) == 66.2
    assert round(fin1["emergency_fund_months"], 1) == 1.5
    assert round(fin1["requested_monthly_installment"], 2) == 9722.22
    assert round(fin1["projected_total_dti"], 1) == 28.0

    # Profile 2: Moderate-Income Professional
    p2 = {
        "monthly_salary": 45000.0,
        "monthly_rent": 12000.0,
        "school_fees": 0.0,
        "college_fees": 0.0,
        "groceries_utilities": 9000.0,
        "travel_expenses": 3000.0,
        "other_monthly_expenses": 4000.0,
        "current_emi_amount": 0.0,
        "emergency_fund": 35000.0,
        "requested_amount": 150000.0,
        "requested_tenure": 18,
    }
    fin2 = calculate_financial_indicators(p2)
    assert fin2["total_monthly_expenses"] == 28000.0
    assert fin2["disposable_income"] == 17000.0
    assert fin2["debt_to_income_ratio"] == 0.0
    assert round(fin2["expense_to_income_ratio"], 1) == 62.2
    assert round(fin2["emergency_fund_months"], 2) == 1.25
    assert round(fin2["requested_monthly_installment"], 2) == 8333.33
    assert round(fin2["projected_total_dti"], 1) == 18.5

    # Profile 3: High-Risk Stretched Customer
    p3 = {
        "monthly_salary": 32000.0,
        "monthly_rent": 10000.0,
        "school_fees": 4000.0,
        "college_fees": 0.0,
        "groceries_utilities": 8500.0,
        "travel_expenses": 2500.0,
        "other_monthly_expenses": 2000.0,
        "current_emi_amount": 9000.0,
        "emergency_fund": 5000.0,
        "requested_amount": 200000.0,
        "requested_tenure": 24,
    }
    fin3 = calculate_financial_indicators(p3)
    assert fin3["total_monthly_expenses"] == 27000.0
    assert fin3["disposable_income"] == -4000.0
    assert round(fin3["debt_to_income_ratio"], 1) == 28.1
    assert round(fin3["expense_to_income_ratio"], 1) == 84.4
    assert round(fin3["emergency_fund_months"], 1) == 0.1
    assert round(fin3["requested_monthly_installment"], 2) == 8333.33
    assert round(fin3["projected_total_dti"], 1) == 54.2


def test_model_inference_on_three_profiles():
    """Verify model inference integrity, probability simplex, and bounds on all 3 profiles."""
    artifacts = load_artifacts()
    clf = artifacts["classification_pipeline"]
    reg = artifacts["regression_pipeline"]

    test_profiles = [
        # Test 1: Low-Risk Salaried
        {
            "age": 31.0, "gender": "Male", "marital_status": "Single", "education": "Graduate",
            "monthly_salary": 65000.0, "employment_type": "Private", "company_type": "MNC",
            "years_of_employment": 6.0, "house_type": "Rented", "monthly_rent": 16000.0, "family_size": 2,
            "dependents": 1, "school_fees": 0.0, "college_fees": 0.0, "travel_expenses": 5500.0,
            "groceries_utilities": 14000.0, "other_monthly_expenses": 7500.0, "existing_loans": "Yes",
            "current_emi_amount": 8500.0, "credit_score": 710.0, "bank_balance": 180000.0,
            "emergency_fund": 75000.0, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 350000.0, "requested_tenure": 36,
        },
        # Test 2: Moderate Professional (with exact dependents=0)
        {
            "age": 28.0, "gender": "Male", "marital_status": "Single", "education": "Graduate",
            "monthly_salary": 45000.0, "employment_type": "Private", "company_type": "Mid-size",
            "years_of_employment": 4.0, "house_type": "Rented", "monthly_rent": 12000.0, "family_size": 1,
            "dependents": 0, "school_fees": 0.0, "college_fees": 0.0, "travel_expenses": 3000.0,
            "groceries_utilities": 9000.0, "other_monthly_expenses": 4000.0, "existing_loans": "No",
            "current_emi_amount": 0.0, "credit_score": 680.0, "bank_balance": 95000.0,
            "emergency_fund": 35000.0, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 150000.0, "requested_tenure": 18,
        },
        # Test 3: High-Risk Stretched Customer
        {
            "age": 36.0, "gender": "Male", "marital_status": "Married", "education": "High School",
            "monthly_salary": 32000.0, "employment_type": "Self-employed", "company_type": "Small",
            "years_of_employment": 8.0, "house_type": "Rented", "monthly_rent": 10000.0, "family_size": 5,
            "dependents": 3, "school_fees": 4000.0, "college_fees": 0.0, "travel_expenses": 2500.0,
            "groceries_utilities": 8500.0, "other_monthly_expenses": 2000.0, "existing_loans": "Yes",
            "current_emi_amount": 9000.0, "credit_score": 590.0, "bank_balance": 15000.0,
            "emergency_fund": 5000.0, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 200000.0, "requested_tenure": 24,
        },
    ]

    for p in test_profiles:
        df = pd.DataFrame([p])
        pred_class = clf.predict(df)[0]
        assert pred_class in ["Eligible", "High_Risk", "Not_Eligible"]

        probas = clf.predict_proba(df)[0]
        assert len(probas) == 3
        assert abs(sum(probas) - 1.0) < 1e-4
        assert all(0.0 <= prob <= 1.0 for prob in probas)

        pred_emi = float(reg.predict(df)[0])
        assert np.isfinite(pred_emi)
        assert pred_emi >= 500.0

    # Discrepancy 1 audit verification:
    # Verifies the exact model output that explained the report (6216.27) vs screenshot (6246.74)
    p2_audit_dep0 = {
        "age": 28.0, "gender": "Male", "marital_status": "Single", "education": "Graduate",
        "house_type": "Rented", "dependents": 0, "family_size": 1, "monthly_salary": 45000.0,
        "employment_type": "Private", "company_type": "Startup", "years_of_employment": 3.0,
        "monthly_rent": 12000.0, "school_fees": 0.0, "college_fees": 0.0, "groceries_utilities": 8000.0,
        "travel_expenses": 3000.0, "other_monthly_expenses": 5000.0, "existing_loans": "No",
        "current_emi_amount": 0.0, "credit_score": 680.0, "bank_balance": 80000.0,
        "emergency_fund": 30000.0, "emi_scenario": "Personal Loan EMI",
        "requested_amount": 200000.0, "requested_tenure": 36,
    }
    p2_audit_dep1 = p2_audit_dep0.copy()
    p2_audit_dep1["dependents"] = 1

    emi_dep0 = float(reg.predict(pd.DataFrame([p2_audit_dep0]))[0])
    emi_dep1 = float(reg.predict(pd.DataFrame([p2_audit_dep1]))[0])
    assert abs(emi_dep0 - 6216.27) < 0.05
    assert abs(emi_dep1 - 6246.74) < 0.05


def test_ui_presets_schema_and_integrity():
    """Verify that all default and preset profiles adhere to complete schema and valid category options."""
    all_expected_cols = set(NUMERICAL_COLUMNS) | set(CATEGORICAL_COLUMNS)

    # Test profiles defined in UI and test specifications
    profiles_to_check = [
        # Documented default
        {
            "age": 35, "gender": "Male", "marital_status": "Married", "education": "Graduate",
            "monthly_salary": 55000.0, "employment_type": "Private", "company_type": "Large Indian",
            "years_of_employment": 5.0, "house_type": "Rented", "monthly_rent": 12000.0, "family_size": 3,
            "dependents": 1, "school_fees": 2000.0, "college_fees": 0.0, "travel_expenses": 3500.0,
            "groceries_utilities": 12000.0, "other_monthly_expenses": 5000.0, "existing_loans": "No",
            "current_emi_amount": 0.0, "credit_score": 730.0, "bank_balance": 150000.0,
            "emergency_fund": 60000.0, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 200000.0, "requested_tenure": 24,
        },
        # Low risk
        {
            "age": 31, "gender": "Male", "marital_status": "Single", "education": "Graduate",
            "monthly_salary": 65000.0, "employment_type": "Private", "company_type": "MNC",
            "years_of_employment": 6.0, "house_type": "Rented", "monthly_rent": 16000.0, "family_size": 2,
            "dependents": 1, "school_fees": 0.0, "college_fees": 0.0, "travel_expenses": 5500.0,
            "groceries_utilities": 14000.0, "other_monthly_expenses": 7500.0, "existing_loans": "Yes",
            "current_emi_amount": 8500.0, "credit_score": 710.0, "bank_balance": 180000.0,
            "emergency_fund": 75000.0, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 350000.0, "requested_tenure": 36,
        },
        # Moderate
        {
            "age": 28, "gender": "Male", "marital_status": "Single", "education": "Graduate",
            "monthly_salary": 45000.0, "employment_type": "Private", "company_type": "Mid-size",
            "years_of_employment": 4.0, "house_type": "Rented", "monthly_rent": 12000.0, "family_size": 1,
            "dependents": 0, "school_fees": 0.0, "college_fees": 0.0, "travel_expenses": 3000.0,
            "groceries_utilities": 9000.0, "other_monthly_expenses": 4000.0, "existing_loans": "No",
            "current_emi_amount": 0.0, "credit_score": 680.0, "bank_balance": 95000.0,
            "emergency_fund": 35000.0, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 150000.0, "requested_tenure": 18,
        },
        # High risk
        {
            "age": 36, "gender": "Male", "marital_status": "Married", "education": "High School",
            "monthly_salary": 32000.0, "employment_type": "Self-employed", "company_type": "Small",
            "years_of_employment": 8.0, "house_type": "Rented", "monthly_rent": 10000.0, "family_size": 5,
            "dependents": 3, "school_fees": 4000.0, "college_fees": 0.0, "travel_expenses": 2500.0,
            "groceries_utilities": 8500.0, "other_monthly_expenses": 2000.0, "existing_loans": "Yes",
            "current_emi_amount": 9000.0, "credit_score": 590.0, "bank_balance": 15000.0,
            "emergency_fund": 5000.0, "emi_scenario": "Personal Loan EMI",
            "requested_amount": 200000.0, "requested_tenure": 24,
        },
    ]

    for prof in profiles_to_check:
        # Schema completeness
        assert set(prof.keys()) == all_expected_cols

        # Categorical constraints
        for cat_col in CATEGORICAL_COLUMNS:
            assert prof[cat_col] in VALID_CATEGORIES[cat_col]

        # Numerical positivity constraints
        assert prof["monthly_salary"] > 0
        assert prof["requested_amount"] > 0
        assert prof["requested_tenure"] > 0
        assert prof["age"] >= 18


