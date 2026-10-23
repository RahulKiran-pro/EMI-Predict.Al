"""
Feature Engineering Module for EMIPredict AI.

Computes key financial domain indicators, debt ratios, affordability indices,
and risk metrics from user profile inputs.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


DEFAULT_FINANCIAL_INPUTS = {
    "monthly_salary": 0.0,
    "current_emi_amount": 0.0,
    "monthly_rent": 0.0,
    "school_fees": 0.0,
    "college_fees": 0.0,
    "travel_expenses": 0.0,
    "groceries_utilities": 0.0,
    "other_monthly_expenses": 0.0,
    "bank_balance": 0.0,
    "emergency_fund": 0.0,
    "requested_amount": 0.0,
    "requested_tenure": 1.0,
}


def calculate_standard_emi(
    principal: float, annual_interest_rate_pct: float, tenure_months: int | float
) -> float:
    """
    Calculate monthly EMI using the standard reducing-balance amortization formula:
    EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)
    where r is monthly interest rate (annual / 12 / 100) and n is tenure in months.
    If interest rate is 0 or non-positive, returns straight-line amortization (P / n).
    """
    if tenure_months <= 0:
        return 0.0
    if annual_interest_rate_pct <= 0:
        return float(principal / tenure_months)
    r = (annual_interest_rate_pct / 100.0) / 12.0
    n = float(tenure_months)
    factor = (1.0 + r) ** n
    return float(principal * r * factor / (factor - 1.0))


def calculate_financial_indicators(data: dict | pd.DataFrame) -> dict | pd.DataFrame:
    """
    Compute domain-specific financial health indicators:
    - total_monthly_expenses
    - disposable_income
    - debt_to_income_ratio (DTI)
    - expense_to_income_ratio (ETI)
    - financial_burden_ratio
    - affordability_ratio
    - emergency_fund_coverage_months
    - requested_monthly_installment (zero-interest estimate)
    - projected_total_dti
    """
    is_dict = isinstance(data, dict)
    if is_dict:
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Fill defaults for calculation safety across all columns
    for col, default_val in DEFAULT_FINANCIAL_INPUTS.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_val)

    # Protect against 0 or negative tenure
    df["requested_tenure"] = df["requested_tenure"].replace(0, 1.0).clip(lower=1.0)

    salary = df["monthly_salary"]
    current_emi = df["current_emi_amount"]
    rent = df["monthly_rent"]
    school = df["school_fees"]
    college = df["college_fees"]
    travel = df["travel_expenses"]
    groceries = df["groceries_utilities"]
    other = df["other_monthly_expenses"]
    emergency = df["emergency_fund"]
    req_amt = df["requested_amount"]
    req_tenure = df["requested_tenure"]

    # 1. Total monthly living expenses (excluding EMIs)
    total_expenses = rent + school + college + travel + groceries + other
    df["total_monthly_expenses"] = total_expenses

    # 2. Net disposable income before new loan
    disposable_income = salary - total_expenses - current_emi
    df["disposable_income"] = disposable_income

    # 3. Debt to Income (DTI) ratio based on existing commitments (%)
    eps = 1e-6
    df["debt_to_income_ratio"] = ((current_emi / (salary + eps)) * 100).clip(0, 200)

    # 4. Expense to Income (ETI) ratio (%)
    df["expense_to_income_ratio"] = ((total_expenses / (salary + eps)) * 100).clip(0, 200)

    # 5. Financial burden ratio (all current obligations / income)
    df["financial_burden_ratio"] = (
        (total_expenses + current_emi) / (salary + eps)
    ).clip(0, 5)

    # 6. Affordability ratio (disposable income / salary)
    df["affordability_ratio"] = (disposable_income / (salary + eps)).clip(-1, 1)

    # 7. Emergency fund coverage (months of expenses + EMI it can cover)
    monthly_outflow = total_expenses + current_emi
    df["emergency_fund_months"] = (
        emergency / (monthly_outflow + eps)
    ).clip(0, 60)

    # 8. Rough requested monthly installment (simple principal / tenure)
    df["requested_monthly_installment"] = req_amt / req_tenure

    # 9. Projected DTI if requested loan is granted
    projected_total_emi = current_emi + df["requested_monthly_installment"]
    df["projected_total_dti"] = (
        (projected_total_emi / (salary + eps)) * 100
    ).clip(0, 300)

    if is_dict:
        return df.iloc[0].to_dict()
    return df


class FinancialRatioTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer to append derived financial ratios."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        return calculate_financial_indicators(df)
