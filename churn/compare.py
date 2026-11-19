# churn/compare.py
import time

import pandas as pd
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from churn import config
from churn.data import load_data, split
from churn.models import make_pipeline

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),  # the dummy never predicts churn
    "recall": "recall",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def main():
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)
    X_train, _, y_train, _ = split(load_data())
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    rows = []
    for name in ["dummy", "logreg", "random_forest", "hist_gb"]:
        start = time.perf_counter()
        scores = cross_validate(make_pipeline(name), X_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            row[metric] = scores[f"test_{metric}"].mean()
        row["roc_auc_std"] = scores["test_roc_auc"].std()
        row["seconds"] = time.perf_counter() - start
        rows.append(row)

    results = pd.DataFrame(rows).set_index("model").round(3)
    print(results.sort_values("roc_auc", ascending=False))

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.REPORTS / "model_comparison.csv")


if __name__ == "__main__":
    main()