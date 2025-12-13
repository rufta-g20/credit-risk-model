import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

# --- CONFIGURATION ---
MODEL_NAME = "DecisionTree_Benchmark" # The best model identified in Task 5
STAGE = "Staging" # Load the model from the Staging registry stage

# --- INITIALIZATION ---
app = FastAPI(title="Credit Risk Model API", version="1.0.0")
model = None

# --- LIFESPAN EVENTS (Startup) ---
@app.on_event("startup")
def load_model():
    """Load the model from the MLflow Model Registry upon server startup."""
    global model
    try:
        # 1. Get the URI for the model in the specified stage
        client = MlflowClient()
        version = client.get_latest_versions(MODEL_NAME, stages=[STAGE])[0].version
        model_uri = f"models:/{MODEL_NAME}/{version}"
        
        # 2. Load the model artifact
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"INFO: Successfully loaded model '{MODEL_NAME}' version {version} from {STAGE} stage.")
        
    except Exception as e:
        print(f"ERROR: Could not load model '{MODEL_NAME}' from MLflow. Ensure 'mlruns/' exists and the model is registered to the '{STAGE}' stage.")
        print(e)
        # You might want to raise the exception to prevent the server from starting

# --- API ENDPOINT ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(customer_data: CustomerFeatures):
    """Accepts customer features and returns the risk probability."""
    
    # 1. Convert the Pydantic input model to a pandas DataFrame (needed for the pipeline)
    input_df = pd.DataFrame([customer_data.dict()])
    
    # 2. Use the MLflow loaded model to get predictions
    # Note: The MLflow saved model is the full pipeline, so it handles WoE transformation internally.
    probabilities = model.predict(input_df) 
    
    # The Decision Tree model (as saved by scikit-learn) returns a probability array (e.g., [[prob_0, prob_1]])
    # We want the probability of being High Risk (Class 1)
    risk_prob = probabilities[:, 1][0]
    
    # 3. Determine the risk label (using a simple threshold, e.g., 0.5)
    risk_label = "High Risk" if risk_prob >= 0.5 else "Low Risk"

    # 4. Return the structured response
    return PredictionResponse(
        risk_probability=risk_prob,
        risk_label=risk_label
    )