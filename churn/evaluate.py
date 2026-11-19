# churn/evaluate.py
import json

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from churn import config
from churn.data import load_data, split
from churn.threshold import business_cost


def main():
    _, X_test, _, y_test = split(load_data())
    bundle = joblib.load(config.MODEL_PATH)
    pipe, threshold = bundle["pipeline"], bundle["threshold"]

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "model": bundle["model_name"],
        "threshold": threshold,
        "test_customers": int(len(y_test)),
        "test_churners": int(y_test.sum()),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "baseline_accuracy": 1 - y_test.mean(),
        "recall_at_default_0.5": recall_score(y_test, (proba >= 0.5).astype(int)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "cost_with_model": business_cost(y_test, pred),
        "cost_at_default_0.5": business_cost(y_test, (proba >= 0.5).astype(int)),
        "cost_without_model": business_cost(y_test, np.zeros_like(pred)),   # contact nobody
    }
    metrics = {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in metrics.items()}

    for key, value in metrics.items():
        print(f"{key}: {value}")

    (config.REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()