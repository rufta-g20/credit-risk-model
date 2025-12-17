import pandas as pd
import numpy as np
import logging
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, status
from mlflow.tracking import MlflowClient
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

# --- CONFIGURATION ---
MODEL_NAME = "DecisionTree_Benchmark"
STAGE = "Staging"
LOG_FORMAT = "%(levelname)s:%(asctime)s:%(module)s:%(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- INITIALIZATION ---
app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Real-time Credit Scoring using RFM and Temporal features.",
    version="1.1.0"
)

# Global variable to hold the loaded MLflow model
model = None

# --- LIFESPAN EVENTS ---
@app.on_event("startup")
def load_model():
    """
    Load the model from the MLflow Model Registry upon server startup.
    Ensures the model is ready before the first request arrives.
    """
    global model
    logger.info(f"Connecting to MLflow Registry to load model: {MODEL_NAME} ({STAGE})")
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])

        if not versions:
            error_msg = f"No model version found for '{MODEL_NAME}' in stage '{STAGE}'."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        version = versions[0].version
        model_uri = f"models:/{MODEL_NAME}/{version}"

        # Load as a PyFunc model (wrapper for any ML framework)
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Successfully loaded {MODEL_NAME} v{version} from registry.")

    except Exception as e:
        logger.exception("CRITICAL: Failed to load model during startup.")
        raise RuntimeError("API failed to initialize model.") from e

# --- UTILITY ENDPOINTS ---
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Health check endpoint for monitoring (e.g., Docker/Kubernetes probes).
    Feedback Requirement: Task 6.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "stage": STAGE
    }


# --- PREDICTION ENDPOINT ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(customer_data: CustomerFeatures):
    logger.info("Prediction request received.")

    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    
    try:
        # 1. Convert Pydantic request directly to a DataFrame
        # The column names match what the 'final_pipeline' expects
        raw_input = pd.DataFrame([customer_data.dict()])

        # 2. Inference
        # The loaded MLflow model IS the pipeline, so it does the WoE/OHE internally
        if hasattr(model, "predict_proba"):
            risk_prob = float(model.predict_proba(raw_input)[0][1])
        else:
            # Fallback for models without predict_proba
            risk_prob = float(model.predict(raw_input)[0])

        risk_label = "High Risk" if risk_prob >= 0.5 else "Low Risk"

        return PredictionResponse(
            risk_probability=round(risk_prob, 4), 
            risk_label=risk_label
        )
    
    except Exception as e:
        logger.exception(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Check if input features match training columns.")