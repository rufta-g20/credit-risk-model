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
    logger.info(f"Prediction request received for Customer.")

    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    try:
        # 1. Convert Pydantic to DataFrame
        # IMPORTANT: Use the RAW feature names the pipeline expects
        input_df = pd.DataFrame([customer_data.dict()])

        # 2. TRANSFORM the raw data
        # If you saved your pipeline separately, load it here. 
        # If your MLflow model is a Pipeline object, it handles this.
        # Given your error, the model is likely just the Tree, so we transform manually:
        
        # Note: In a production setup, you'd load 'full_pipeline.pkl' here
        # For now, we use the model's internal transformation if it's a pipeline
        prediction_prob = model.predict(input_df) 

        # 3. Handle the logic based on model output
        risk_prob = float(prediction_prob[0])
        threshold = 0.5
        risk_label = "High Risk" if risk_prob >= threshold else "Low Risk"

        return PredictionResponse(risk_probability=risk_prob, risk_label=risk_label)

    except Exception as e:
        logger.exception(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    