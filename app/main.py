# app/main.py
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, get_args

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from churn import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("churn_api")

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

YesNo = Literal["Yes", "No"]
InternetAddOn = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer, with exactly the columns the model was trained on."""
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
    }]})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddOn
    OnlineBackup: InternetAddOn
    DeviceProtection: InternetAddOn
    TechSupport: InternetAddOn
    StreamingTV: InternetAddOn
    StreamingMovies: InternetAddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold: float
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the saved model once at startup, not on every request
    if config.MODEL_PATH.exists():
        app.state.bundle = joblib.load(config.MODEL_PATH)
        logger.info("Loaded model %s (threshold %s)", app.state.bundle["model_name"], app.state.bundle["threshold"])
    else:
        app.state.bundle = None
        logger.warning("No model found at %s", config.MODEL_PATH)
    yield
    app.state.bundle = None


app = FastAPI(title="Customer Churn API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with a unique ID, its status code and how long it took."""
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
                request_id, request.method, request.url.path, response.status_code, duration_ms)
    return response


def get_bundle(request: Request):
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python -m churn.train first.")
    return bundle


Bundle = Annotated[dict, Depends(get_bundle)]


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": getattr(request.app.state, "bundle", None) is not None}


@app.get("/model")
def model_info(bundle: Bundle):
    return {k: bundle[k] for k in ["model_name", "threshold", "trained_at", "params"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, bundle: Bundle):
    row = pd.DataFrame([customer.model_dump()])
    probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
    logger.info("prediction probability=%.4f will_churn=%s", probability, probability >= bundle["threshold"])
    return Prediction(
        churn_probability=round(probability, 4),
        will_churn=probability >= bundle["threshold"],
        threshold=bundle["threshold"],
        model=bundle["model_name"],
    )


def form_fields():
    """Describe every Customer field for the web form, taken from the Pydantic model itself."""
    fields = []
    for name, info in Customer.model_fields.items():
        choices = list(get_args(info.annotation))
        fields.append({
            "name": name,
            "choices": choices,                       # empty for number fields
            "step": "1" if info.annotation is int else "0.01",
        })
    return fields


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request):
    example = Customer.model_config["json_schema_extra"]["examples"][0]
    return templates.TemplateResponse(
        request, "index.html", {"fields": form_fields(), "example": example}
    )