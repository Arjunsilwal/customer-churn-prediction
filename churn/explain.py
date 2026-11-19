# churn/explain.py
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from churn import config
from churn.data import load_data, split


def main():
    _, X_test, _, y_test = split(load_data())
    pipe = joblib.load(config.MODEL_PATH)["pipeline"]

    # Shuffle one column at a time and measure how much test ROC-AUC drops
    result = permutation_importance(
        pipe, X_test, y_test, scoring="roc_auc", n_repeats=10,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    importance = pd.DataFrame({
        "importance": result.importances_mean,
        "std": result.importances_std,
    }, index=X_test.columns).sort_values("importance", ascending=False).round(4)
    print(importance)

    importance.to_csv(config.REPORTS / "feature_importance.csv")
    top = importance.head(10).iloc[::-1]
    top["importance"].plot(kind="barh", xerr=top["std"], title="Permutation importance (drop in test ROC-AUC)")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "feature_importance.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()