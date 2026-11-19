# churn/features.py
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config


def build_preprocessor(scale=True):
    """Clean numeric and categorical columns. Scaling is only needed for linear models."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), config.NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_COLS),
    ])