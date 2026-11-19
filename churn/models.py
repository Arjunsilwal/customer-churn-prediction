# churn/models.py
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn import config
from churn.features import build_preprocessor

# Only the linear model needs scaled inputs
NEEDS_SCALING = {"dummy": False, "logreg": True, "random_forest": False, "hist_gb": False}


def make_estimator(name):
    estimators = {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=config.RANDOM_STATE
        ),
        "hist_gb": HistGradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }
    return estimators[name]


def make_pipeline(name, params=None):
    """Preprocessing plus model, optionally with tuned settings such as {"model__C": 0.1}."""
    pipe = Pipeline([
        ("prep", build_preprocessor(scale=NEEDS_SCALING[name])),
        ("model", make_estimator(name)),
    ])
    if params:
        pipe.set_params(**params)
    return pipe