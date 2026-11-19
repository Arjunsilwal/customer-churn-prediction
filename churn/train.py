# churn/train.py
import json
from datetime import datetime, timezone

import joblib

from churn import config
from churn.data import load_data, split
from churn.models import make_pipeline


def main():
    X_train, _, y_train, _ = split(load_data())
    best = json.loads((config.REPORTS / "best_model.json").read_text())
    threshold = json.loads((config.REPORTS / "threshold.json").read_text())["threshold"]

    # Refit the chosen pipeline on the entire training set
    pipe = make_pipeline(best["model"], best["params"]).fit(X_train, y_train)

    # Save the model together with everything needed to use it correctly
    bundle = {
        "pipeline": pipe,
        "threshold": threshold,
        "model_name": best["model"],
        "params": best["params"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": config.NUMERIC_COLS + config.CATEGORICAL_COLS,
    }
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)
    print(f"Saved {best['model']} with threshold {threshold} to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()