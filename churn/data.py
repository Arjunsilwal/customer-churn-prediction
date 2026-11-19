# churn/data.py
import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config


def load_raw(path=config.RAW_DATA):
    """Read the CSV exactly as it is on disk."""
    return pd.read_csv(path)


def clean(df):
    """Fix the problems found in the raw file and return a model-ready table."""
    df = df.copy()

    # TotalCharges is stored as text because 11 rows contain a blank space.
    # Those customers have tenure 0: they have not been billed yet, so the true value is 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # The label must be a number: Yes becomes 1, No becomes 0.
    df[config.TARGET] = df[config.TARGET].map({"Yes": 1, "No": 0})

    # The ID is unique per customer and carries no pattern to learn.
    df = df.drop(columns=[config.ID_COL])
    return df


def load_data():
    return clean(load_raw())


def split(df):
    """The one and only train/test split. Every script calls this, so the test set never changes."""
    X = df.drop(columns=config.TARGET)
    y = df[config.TARGET]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


if __name__ == "__main__":
    raw = load_raw()
    print("Raw shape:", raw.shape)
    print("TotalCharges type in the raw file:", raw["TotalCharges"].dtype)
    blanks = raw["TotalCharges"].str.strip() == ""
    print("Blank TotalCharges:", blanks.sum(), "| their tenure values:", raw.loc[blanks, "tenure"].unique())

    df = load_data()
    print("Clean shape:", df.shape)
    print("Missing values after cleaning:", int(df.isna().sum().sum()))
    print("Churn rate:", round(df[config.TARGET].mean(), 3))

    X_train, X_test, y_train, y_test = split(df)
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Churn rate train/test:", round(y_train.mean(), 3), round(y_test.mean(), 3))