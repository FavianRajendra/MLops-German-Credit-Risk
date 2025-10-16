# Goal: Serve the two-stage, semi-supervised prediction pipeline via a production-ready FastAPI endpoint.
# This endpoint accepts raw customer data and returns the assigned Risk Segment (Cluster ID).

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os

# --- 1. CONFIGURATION ---
PIPELINE_FILE = 'two_stage_prediction_pipeline.joblib'

# Define the risk segment explanations
RISK_SEGMENTS = {
    0: "Cluster 0: Low Risk/High Stability Segment",
    1: "Cluster 1: Medium Risk/Moderate Credit Segment",
    2: "Cluster 2: High Risk/High Debt Segment"
    # Note: These meanings are inferred, but in a real project, they would be defined via EDA.
}

# --- 2. MODEL LOADING ---

try:
    # Load the entire trained pipeline (preprocessing + classifier)
    pipeline = joblib.load(PIPELINE_FILE)
    print(f"Model pipeline '{PIPELINE_FILE}' loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Pipeline file '{PIPELINE_FILE}' not found. Run model_train.py first!")
    pipeline = None
except Exception as e:
    print(f"ERROR loading pipeline: {e}")
    pipeline = None


# --- 3. API SCHEMA DEFINITION (Input) ---

# Define the exact input features required by the pipeline.
# These must match the raw features used in model_train.py.
class CustomerFeatureRequest(BaseModel):
    # Numerical Features
    Age: int = Field(..., description="Customer's age.", example=35, ge=18)
    Duration: int = Field(..., description="Duration of the loan in months.", example=24, ge=1)

    # Categorical/Semi-Numerical Features
    Credit_amount: int = Field(..., description="Credit amount requested (DM/USD).", example=12000, ge=100)
    Job: int = Field(..., description="Job classification (0-3 scale).", example=2, ge=0, le=3)

    # Categorical String Features
    Sex: str = Field(..., description="Gender (male or female).", example="male")
    Housing: str = Field(..., description="Housing status (own, rent, or free).", example="rent")
    Saving_accounts: str = Field(..., description="Saving account status (little, moderate, rich).", example="moderate")
    Checking_account: str = Field(..., description="Checking account status.", example="little")
    Purpose: str = Field(..., description="Purpose of the loan (e.g., car, education).", example="radio/TV")

    # Ensures input validation and helps the API Docs immensely
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 29,
                    "Credit_amount": 4500,
                    "Duration": 18,
                    "Job": 2,
                    "Sex": "female",
                    "Housing": "own",
                    "Saving_accounts": "moderate",
                    "Checking_account": "little",
                    "Purpose": "furniture/equipment"
                }
            ]
        }
    }


# --- 4. API SCHEMA DEFINITION (Output) ---

class PredictionResponse(BaseModel):
    risk_segment_id: int = Field(...,
                                 description="The predicted risk segment (Cluster ID) assigned to the customer (0, 1, or 2).")
    risk_segment_name: str = Field(..., description="A human-readable name for the assigned risk segment.")
    model_info: str = Field(...,
                            description="Confirmation that the prediction was run via the two-stage MLOps pipeline.")


# --- 5. FASTAPI APPLICATION ---

app = FastAPI(
    title="MLOps Credit Risk Segmentation API",
    description="Serves the Two-Stage Semi-Supervised Credit Risk Pipeline."
)


@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {"message": "MLOps Prediction API is operational."}


@app.post("/predict_segment/", response_model=PredictionResponse)
def predict_risk_segment(request: CustomerFeatureRequest):
    """
    Receives customer features and returns the predicted risk segment (Cluster ID).
    """
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="ML Pipeline not loaded. Check server logs or run model_train.py."
        )

    try:
        # 1. Convert Pydantic request model (request) into a DataFrame
        # The column names must exactly match the internal model names (lowercase, no spaces).

        # We collect input data ensuring correct case and structure for the pipeline.
        data = {
            'age': [request.Age],
            'credit amount': [request.Credit_amount],
            # Note: space is replaced by underscore in training script but we handle it here
            'duration': [request.Duration],
            'job': [request.Job],
            'sex': [request.Sex],
            'housing': [request.Housing],
            'saving accounts': [request.Saving_accounts],
            'checking account': [request.Checking_account],
            'purpose': [request.Purpose]
        }

        # Create DataFrame from the input data
        # NOTE: We use the lowercase feature names expected by the training script.
        df_input = pd.DataFrame(data)

        # 2. Run the prediction (the pipeline handles all preprocessing)
        # The output is a single segment ID (0, 1, or 2)
        segment_id = pipeline.predict(df_input)[0]

        # 3. Construct response
        segment_name = RISK_SEGMENTS.get(segment_id, "Unknown Segment")

        return PredictionResponse(
            risk_segment_id=int(segment_id),
            risk_segment_name=segment_name,
            model_info="Prediction run via Two-Stage Semi-Supervised Pipeline (Decision Tree on K-Means Clusters)."
        )

    except Exception as e:
        print(f"Prediction Error: {e}")
        # In a production environment, avoid showing internal error details
        raise HTTPException(status_code=500, detail="Internal prediction error. Check input format.")
