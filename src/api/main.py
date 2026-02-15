import pandas as pd
import numpy as np
import logging
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, status
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

# --- STRUCTURED LOGGING SETUP ---
LOG_FORMAT = '{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": "%(message)s"}'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
MODEL_NAME = "Logistic_Scorecard" 
STAGE = "None" 

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Basel II compliant credit scoring engine.",
    version="2.0.0"
)

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Loading the latest version from MLflow
        model_uri = f"models:/{MODEL_NAME}/latest"
        logger.info(f"Attempting to load model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model successfully loaded into memory.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # We don't raise RuntimeError here so the container starts, 
        # but /health will show the failure.

@app.get("/health")
def health():
    return {
        "status": "ready" if model else "loading/error",
        "model_registry_name": MODEL_NAME,
        "api_version": "2.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerFeatures):
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        logger.info(f"Processing prediction for customer with F_Count: {data.F_Count}")
        
        # 1. Use model_dump() for Pydantic V2
        input_df = pd.DataFrame([data.model_dump()])
    
        # 2. In MLflow pyfunc, predict() often returns the class. 
        # To get probabilities reliably across different MLflow versions:
        try:
            # Some flavors return probabilities directly if configured, 
            # but usually, we call the underlying sklearn model:
            prediction_prob = model._model_impl.python_model.predict_proba(input_df)[0][1]
        except AttributeError:
            # Fallback if the internal structure is different
            prediction = model.predict(input_df)
            prediction_prob = float(prediction[0])

        risk_label = "High Risk" if prediction_prob > 0.5 else "Low Risk"
        
        return PredictionResponse(
            risk_probability=round(float(prediction_prob), 4),
            risk_label=risk_label,
            justification={} 
        )

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference Error: {str(e)}"
        )