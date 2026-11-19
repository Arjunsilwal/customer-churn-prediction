# churn/tune.py
import json

from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from churn import config
from churn.data import load_data, split
from churn.models import make_pipeline


def main():
    X_train, _, y_train, _ = split(load_data())
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    # Logistic regression has one important setting, so a small grid covers it
    logreg_search = GridSearchCV(
        make_pipeline("logreg"),
        {"model__C": [0.01, 0.1, 1, 10, 100, 1000]},
        cv=cv, scoring="roc_auc", n_jobs=-1,
    ).fit(X_train, y_train)

    # Gradient boosting has many settings, so sample 30 random combinations
    hgb_search = RandomizedSearchCV(
        make_pipeline("hist_gb"),
        {
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_iter": randint(100, 600),
            "model__max_leaf_nodes": randint(8, 48),
            "model__min_samples_leaf": randint(10, 100),
            "model__l2_regularization": uniform(0, 1),
        },
        n_iter=30, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=config.RANDOM_STATE,
    ).fit(X_train, y_train)

    results = {
        "logreg": (logreg_search.best_score_, logreg_search.best_params_),
        "hist_gb": (hgb_search.best_score_, hgb_search.best_params_),
    }
    for name, (score, params) in results.items():
        readable = {k: round(float(v), 4) for k, v in params.items()}
        print(f"{name}: cross-validated ROC-AUC {score:.4f} with {readable}")

    best_name = max(results, key=lambda n: results[n][0])
    best_score, best_params = results[best_name]
    best = {
        "model": best_name,
        "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()},
        "cv_roc_auc": round(float(best_score), 4),
    }
    print("Selected:", best)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "best_model.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()