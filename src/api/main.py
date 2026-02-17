import pandas as pd
import numpy as np
import logging
import mlflow.sklearn  
from fastapi import FastAPI, HTTPException, status
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
from src.predict import get_prediction_justification

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
        logger.info(f"Loading native Sklearn model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Native Sklearn Pipeline successfully loaded.")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
       
@app.get("/health")
def health():
    return {
        "status": "ready" if model else "loading/error",
        "model_registry_name": MODEL_NAME,
        "api_version": "2.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # 1. Prepare Input
        input_df = pd.DataFrame([data.model_dump()])
        float_cols = ["M_Debit_Total", "M_Debit_Mean", "M_Debit_Std", "F_Count", "R_Min_Days", "Hour_Mode"]
        for col in float_cols:
            input_df[col] = input_df[col].astype(float)
    
        # 2. Direct Prediction (Since 'model' is now the actual Pipeline)
        # pipeline.predict_proba returns [ [prob_0, prob_1] ]
        prob_value = model.predict_proba(input_df)[0][1]
        risk_label = "High Risk" if prob_value > 0.5 else "Low Risk"
        
        # 3. Justification (Passing the native pipeline)
        justification = get_prediction_justification(model, input_df)
    
        return PredictionResponse(
            risk_probability=round(float(prob_value), 4),
            risk_label=risk_label,
            justification=justification
        )
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))