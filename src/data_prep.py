"""Data loading, cleaning, feature engineering and preprocessing for the ConnectTel churn project."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET = "Churn"

# Optional services a customer can add on top of their base phone / internet plan.
ADDON_SERVICES = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
PROTECTION_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_SERVICES = ["StreamingTV", "StreamingMovies"]

ENGINEERED_FEATURES = ["TotalChargesPerTenure", "ServiceCount", "TenureGroup", "HasProtection", "HasStreaming",
                       "IsAutoPay"]

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49-72m"]


def load_raw(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def clean(df):
    """Fix dtypes, impute TotalCharges for brand-new customers and encode the target as 0/1."""
    df = df.copy()
    # TotalCharges is blank (" ") only for customers with tenure 0: they have not been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df.drop(columns=["customerID"])


def engineer_features(df):
    """Add the engineered features described in the project brief (plus three extras)."""
    df = df.copy()
    # Average amount billed per month of tenure; new customers fall back to their monthly charge.
    df["TotalChargesPerTenure"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"]
    )
    df["ServiceCount"] = df[ADDON_SERVICES].eq("Yes").sum(axis=1)
    df["TenureGroup"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    df["HasProtection"] = df[PROTECTION_SERVICES].eq("Yes").any(axis=1).map({True: "Yes", False: "No"})
    df["HasStreaming"] = df[STREAMING_SERVICES].eq("Yes").any(axis=1).map({True: "Yes", False: "No"})
    df["IsAutoPay"] = df["PaymentMethod"].str.contains("automatic").map({True: "Yes", False: "No"})
    return df


def what_if(X, **changes):
    """Copy of feature frame X with raw columns overridden (e.g. Contract="One year") and the engineered
    features recomputed, so derived columns stay consistent with the change. A column given as a dict
    maps old values to new ones (e.g. PaymentMethod={"Electronic check": "Credit card (automatic)"})."""
    X = X.drop(columns=ENGINEERED_FEATURES).copy()
    for col, value in changes.items():
        X[col] = X[col].replace(value) if isinstance(value, dict) else value
    return engineer_features(X)


def load_model_data(path=RAW_DATA_PATH):
    """Raw CSV -> cleaned, feature-engineered frame ready for modelling."""
    return engineer_features(clean(load_raw(path)))


def split_features(df):
    """Return (X, y, numeric_cols, categorical_cols)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """One-hot encode categoricals and standard-scale numerics.

    Meant to live inside a sklearn Pipeline so it is re-fitted on each CV training fold.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        verbose_feature_names_out=False,
    )
