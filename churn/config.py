# churn/config.py
from pathlib import Path

# Folders and files, built from this file's location so they work from anywhere
ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Reproducibility and the split
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Columns
TARGET = "Churn"
ID_COL = "customerID"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# Business assumptions used to choose the decision threshold.
# Replace these with real figures from the business if you have them.
COST_OF_OFFER = 50            # dollars spent on each customer we contact (discount, call time)
COST_OF_LOST_CUSTOMER = 500   # future revenue lost when a customer leaves
OFFER_SUCCESS_RATE = 0.5      # share of contacted churners the offer actually keeps