from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from src.schemas import CustomerInput, PredictionResponse


# Locate the saved pipeline relative to this project's folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline_tuned.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the saved pipeline once when the API starts."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved pipeline not found: {MODEL_PATH}. "
            "Run the notebook's model-saving cell first."
        )

    pipeline = joblib.load(MODEL_PATH)

    # Find the probability column for churn: 1 means Yes.
    classes = list(pipeline.classes_)

    if set(classes) != {0, 1}:
        raise ValueError("The saved model must use labels 0 and 1.")

    app.state.pipeline = pipeline
    app.state.churn_class_index = classes.index(1)

    yield

    # Release the model when the API shuts down.
    del app.state.pipeline


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using the saved decision-tree pipeline.",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput, request: Request):
    """Validate customer inputs and return a churn prediction."""

    # Convert validated customer details into a one-row DataFrame.
    customer_df = pd.DataFrame([customer.model_dump()])

    pipeline = request.app.state.pipeline
    churn_index = request.app.state.churn_class_index

    # The saved pipeline performs preprocessing automatically.
    predicted_class = int(pipeline.predict(customer_df)[0])

    churn_probability = float(
        pipeline.predict_proba(customer_df)[0, churn_index]
    )

    return PredictionResponse(
        prediction="Yes" if predicted_class == 1 else "No",
        churn_probability=churn_probability
    )