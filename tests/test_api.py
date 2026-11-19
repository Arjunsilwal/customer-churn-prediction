# tests/test_api.py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import Customer, app, get_bundle

VALID_CUSTOMER = Customer.model_config["json_schema_extra"]["examples"][0]


class FakePipeline:
    """Stands in for the real model and always returns the probability we choose."""
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, rows):
        return np.array([[1 - self.probability, self.probability] for _ in range(len(rows))])


def fake_bundle(probability):
    return {"pipeline": FakePipeline(probability), "threshold": 0.2, "model_name": "fake",
            "trained_at": "test", "params": {}}


@pytest.fixture
def client():
    yield TestClient(app)   # no "with": the real model is not loaded
    app.dependency_overrides.clear()


def use_probability(probability):
    app.dependency_overrides[get_bundle] = lambda: fake_bundle(probability)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_every_response_has_a_request_id(client):
    response = client.get("/health")
    assert len(response.headers["X-Request-ID"]) > 0


def test_request_id_from_caller_is_kept(client):
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"


def test_home_page_shows_the_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<form" in response.text
    assert "Month-to-month" in response.text      # dropdown options come from the Pydantic model


def test_predictions_are_logged(client, caplog):
    use_probability(0.35)
    with caplog.at_level("INFO", logger="churn_api"):
        client.post("/predict", json=VALID_CUSTOMER)
    assert "will_churn=True" in caplog.text


def test_high_probability_is_flagged(client):
    use_probability(0.35)
    body = client.post("/predict", json=VALID_CUSTOMER).json()
    assert body["churn_probability"] == pytest.approx(0.35)
    assert body["will_churn"] is True


def test_low_probability_is_not_flagged(client):
    use_probability(0.10)
    assert client.post("/predict", json=VALID_CUSTOMER).json()["will_churn"] is False


@pytest.mark.parametrize("field, bad_value", [
    ("Contract", "Weekly"),
    ("tenure", -3),
    ("MonthlyCharges", "a lot"),
    ("InternetService", "Satellite"),
])
def test_invalid_input_returns_422(client, field, bad_value):
    use_probability(0.5)
    response = client.post("/predict", json={**VALID_CUSTOMER, field: bad_value})
    assert response.status_code == 422


def test_missing_field_returns_422(client):
    use_probability(0.5)
    customer = {k: v for k, v in VALID_CUSTOMER.items() if k != "Contract"}
    assert client.post("/predict", json=customer).status_code == 422


def test_real_model_end_to_end():
    """Integration test: loads the saved model through the real startup code."""
    with TestClient(app) as c:
        response = c.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0