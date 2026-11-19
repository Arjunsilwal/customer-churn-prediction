# churn/eda.py
import matplotlib.pyplot as plt
import pandas as pd

from churn import config
from churn.data import load_data, split


def churn_rate_by(df, column):
    table = df.groupby(column)[config.TARGET].agg(customers="count", churn_rate="mean")
    return table.sort_values("churn_rate", ascending=False).round(3)


def main():
    df = load_data()
    # Explore the training part only, so nothing about the test customers shapes our choices
    X_train, _, y_train, _ = split(df)
    train = X_train.assign(**{config.TARGET: y_train})

    print("Overall churn rate:", round(train[config.TARGET].mean(), 3), "\n")
    for column in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen"]:
        print(churn_rate_by(train, column), "\n")

    train["tenure_group"] = pd.cut(train["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                                   labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    print(churn_rate_by(train, "tenure_group").sort_index(), "\n")

    print(train.groupby(config.TARGET)[config.NUMERIC_COLS].median().round(1))

    config.FIGURES.mkdir(parents=True, exist_ok=True)

    rates = churn_rate_by(train, "Contract")["churn_rate"]
    ax = rates.plot(kind="bar", rot=0, title="Churn rate by contract type")
    ax.set_ylabel("Churn rate")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "churn_by_contract.png", dpi=120)
    plt.close()

    for label, name in [(0, "Stayed"), (1, "Churned")]:
        train.loc[train[config.TARGET] == label, "tenure"].plot(kind="hist", bins=24, alpha=0.6, label=name)
    plt.title("Tenure of customers who stayed vs churned")
    plt.xlabel("Tenure (months)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES / "tenure_by_churn.png", dpi=120)
    plt.close()
    print("\nCharts saved to", config.FIGURES)


if __name__ == "__main__":
    main()