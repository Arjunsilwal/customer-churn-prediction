# churn/threshold.py
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churn import config
from churn.data import load_data, split
from churn.models import make_pipeline


def business_cost(y_true, y_pred):
    """Dollars spent and lost for a set of yes/no decisions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    offers = (tp + fp) * config.COST_OF_OFFER                       # everyone flagged gets an offer
    missed = fn * config.COST_OF_LOST_CUSTOMER                      # churners we never contacted
    not_saved = tp * (1 - config.OFFER_SUCCESS_RATE) * config.COST_OF_LOST_CUSTOMER  # contacted but left anyway
    return float(offers + missed + not_saved)


def main():
    X_train, _, y_train, _ = split(load_data())
    best = json.loads((config.REPORTS / "best_model.json").read_text())
    pipe = make_pipeline(best["model"], best["params"])

    # Out-of-fold probabilities: each customer is scored by a model that never saw them
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": t,
            "flagged": tp + fp,
            "caught": tp,
            "missed": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn),
            "cost": business_cost(y_train, pred),
        })
    table = pd.DataFrame(rows).set_index("threshold").round(3)
    print(table)

    best_t = float(table["cost"].idxmin())
    no_model = business_cost(y_train, np.zeros(len(y_train), dtype=int))
    print(f"\nCost with no model (contact nobody): ${no_model:,.0f}")
    print(f"Cost at the default threshold 0.5:   ${table.loc[0.5, 'cost']:,.0f}")
    print(f"Lowest cost, at threshold {best_t}:   ${table.loc[best_t, 'cost']:,.0f}")

    (config.REPORTS / "threshold.json").write_text(json.dumps({"threshold": best_t}, indent=2))


if __name__ == "__main__":
    main()