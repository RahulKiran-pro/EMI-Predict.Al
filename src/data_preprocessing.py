"""
Data Preprocessing Module for EMIPredict AI.

Handles data cleaning, schema validation, feature segregation,
categorical normalization, and scikit-learn preprocessing pipelines.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_CLASSIFICATION = "emi_eligibility"
TARGET_REGRESSION = "max_monthly_emi"
TARGET_COLUMNS = [TARGET_CLASSIFICATION, TARGET_REGRESSION]

CATEGORICAL_COLUMNS = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
]

NUMERICAL_COLUMNS = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
]

# Valid expected categories for strict input validation
VALID_CATEGORIES = {
    "gender": ["Male", "Female"],
    "marital_status": ["Married", "Single"],
    "education": ["Graduate", "Post Graduate", "High School", "Professional"],
    "employment_type": ["Private", "Government", "Self-employed"],
    "company_type": ["Large Indian", "MNC", "Mid-size", "Startup", "Small"],
    "house_type": ["Rented", "Own", "Family"],
    "existing_loans": ["No", "Yes"],
    "emi_scenario": [
        "Home Appliances EMI",
        "Personal Loan EMI",
        "E-commerce Shopping EMI",
        "Education EMI",
        "Vehicle EMI",
    ],
}


def clean_numeric_value(val):
    """Clean a single numeric string or number that may have malformed trailing dots."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in ("nan", "nan.0", "none", "null", ""):
        return np.nan
    # Remove repeated trailing .0 suffixes (e.g., '58.0.0', '18000.0.0.0')
    s = re.sub(r"(?:\.0)+$", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalize_gender(val):
    """Normalize inconsistent gender casing and abbreviations."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if s in ("MALE", "M"):
        return "Male"
    if s in ("FEMALE", "F"):
        return "Female"
    return str(val).strip()


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Normalizes string types and strips spaces.
    - Resolves repeated .0 and malformed numeric strings in age, monthly_salary, bank_balance.
    - Normalizes gender values (e.g. MALE, M, female -> Male, Female).
    - Ensures pure numpy float64/object types for sklearn imputer compatibility.
    """
    df_clean = df.copy()

    # Clean numeric columns with known formatting artifacts
    for col in ["age", "monthly_salary", "bank_balance"]:
        if col in df_clean.columns:
            s = df_clean[col].astype(str).str.strip()
            s = s.str.replace(r"(?:\.0)+$", "", regex=True)
            s = s.mask(s.str.lower().isin(["nan", "nan.0", "none", "null", ""]), None)
            df_clean[col] = pd.to_numeric(s, errors="coerce").astype(float)

    # Clean and standardize gender
    if "gender" in df_clean.columns:
        df_clean["gender"] = df_clean["gender"].apply(normalize_gender)

    # Ensure all numerical columns are float
    for col in NUMERICAL_COLUMNS:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").astype(float)

    if TARGET_REGRESSION in df_clean.columns:
        df_clean[TARGET_REGRESSION] = pd.to_numeric(
            df_clean[TARGET_REGRESSION], errors="coerce"
        ).astype(float)

    # Clean string columns: trim whitespace, use None for missing, keep as object dtype
    for col in CATEGORICAL_COLUMNS:
        if col in df_clean.columns:
            s = df_clean[col].astype(str).str.strip()
            s = s.mask(s.str.lower().isin(["nan", "nan.0", "none", "null", ""]), None)
            df_clean[col] = s.astype(object)

    if TARGET_CLASSIFICATION in df_clean.columns:
        s = df_clean[TARGET_CLASSIFICATION].astype(str).str.strip()
        s = s.mask(s.str.lower().isin(["nan", "nan.0", "none", "null", ""]), None)
        df_clean[TARGET_CLASSIFICATION] = s.astype(object)

    return df_clean


def load_and_clean_data(csv_path: str | Path) -> pd.DataFrame:
    """Load raw dataset from CSV and apply cleaning rules."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    df = pd.read_csv(path, low_memory=False)
    return clean_raw_data(df)


def get_feature_columns():
    """Return numeric and categorical feature lists strictly excluding targets."""
    return list(NUMERICAL_COLUMNS), list(CATEGORICAL_COLUMNS)


def build_preprocessor(numeric_features=None, categorical_features=None):
    """
    Create a preprocessing pipeline for numerical and categorical input features.
    - Numerical: SimpleImputer(median) + StandardScaler
    - Categorical: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
    """
    if numeric_features is None:
        numeric_features = NUMERICAL_COLUMNS
    if categorical_features is None:
        categorical_features = CATEGORICAL_COLUMNS

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_train_test_data(
    csv_path: str | Path = "data/emi_prediction_dataset.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    sample_size: int | None = None,
):
    """
    Prepare stratified train-test splits for classification and regression tasks.
    Splits features from targets, avoiding any data leakage.
    """
    df = load_and_clean_data(csv_path)

    # Validate target columns exist
    for target in TARGET_COLUMNS:
        if target not in df.columns:
            raise KeyError(f"Required target '{target}' not found in dataset.")

    feature_cols = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    X = df[feature_cols].copy()
    y_class = df[TARGET_CLASSIFICATION].copy()
    y_reg = df[TARGET_REGRESSION].copy()

    # Drop any rows where target is missing if applicable
    valid_mask = y_class.notna() & y_reg.notna()
    X = X.loc[valid_mask]
    y_class = y_class.loc[valid_mask]
    y_reg = y_reg.loc[valid_mask]

    if sample_size and sample_size < len(X):
        # Stratified sample if requested for fast testing
        sample_indices, _ = train_test_split(
            np.arange(len(X)),
            train_size=sample_size,
            random_state=random_state,
            stratify=y_class,
        )
        X = X.iloc[sample_indices].reset_index(drop=True)
        y_class = y_class.iloc[sample_indices].reset_index(drop=True)
        y_reg = y_reg.iloc[sample_indices].reset_index(drop=True)

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = (
        train_test_split(
            X,
            y_class,
            y_reg,
            test_size=test_size,
            random_state=random_state,
            stratify=y_class,
        )
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_class_train": y_class_train,
        "y_class_test": y_class_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "numeric_features": NUMERICAL_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
    }
