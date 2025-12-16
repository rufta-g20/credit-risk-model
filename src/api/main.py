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
    """
    Accepts customer features, performs inference, and returns risk assessment.
    
    Includes input range validation via Pydantic and logic for 
    probability-based classification.
    """
    logger.info(f"Prediction request received for Customer.")

    # 1. Ensure model is available
    if model is None:
        logger.error("Attempted prediction while model was None.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently loading or unavailable.",
        )

    try:
        # 2. Convert Pydantic model to DataFrame
        # The model expects the same columns used in training (Task 3/4 output)
        input_data = pd.DataFrame([customer_data.dict()])

        # 3. Perform Inference
        # Note: If your model was logged via mlflow.sklearn, it might support predict_proba
        # We handle both standard predict and proba if available
        if hasattr(model._model_impl, "predict_proba"):
             # Returns [prob_class_0, prob_class_1]
            probabilities = model._model_impl.predict_proba(input_data)
            risk_prob = float(probabilities[0][1])
        else:
            # Fallback for models that don't directly expose proba through pyfunc wrapper
            # Some pyfunc models return the class; you may need to adjust based on your specific model
            prediction = model.predict(input_data)
            risk_prob = float(prediction[0]) # Assuming the model returns a probability directly

        # 4. Apply Business Threshold (Feedback: Task 4 context)
        # Default 0.5 threshold to classify based on proxy target 'is_high_risk'
        threshold = 0.5
        risk_label = "High Risk" if risk_prob >= threshold else "Low Risk"

        logger.info(f"Inference complete: Score={risk_prob:.4f}, Result={risk_label}")

        return PredictionResponse(
            risk_probability=risk_prob, 
            risk_label=risk_label
        )

    except Exception as e:
        logger.exception(f"Inference Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal model error during prediction: {str(e)}",
        )