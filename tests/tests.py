# tests/test_data.py
import pandas as pd

from churn import config
from churn.data import clean


def make_raw():
    return pd.DataFrame({
        "customerID": ["0001-A", "0002-B"],
        "tenure": [0, 24],
        "TotalCharges": [" ", "1500.5"],
        "Churn": ["No", "Yes"],
    })


def test_clean_fixes_total_charges():
    df = clean(make_raw())
    assert df["TotalCharges"].tolist() == [0.0, 1500.5]
    assert df["TotalCharges"].dtype == float


def test_clean_maps_label_to_numbers():
    df = clean(make_raw())
    assert df[config.TARGET].tolist() == [0, 1]


def test_clean_drops_id_and_keeps_input_unchanged():
    raw = make_raw()
    df = clean(raw)
    assert config.ID_COL not in df.columns
    assert config.ID_COL in raw.columns   # clean() must not modify its input