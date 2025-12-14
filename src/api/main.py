import pandas as pd
from fastapi import FastAPI, HTTPException, status
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import logging  # <-- NEW

# --- CONFIGURATION ---
MODEL_NAME = "DecisionTree_Benchmark"
STAGE = "Staging"
LOG_FORMAT = (
    "%(levelname)s:%(asctime)s:%(module)s:%(message)s"  # Define a structured log format
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)  # Get logger instance

# --- INITIALIZATION ---
app = FastAPI(title="Credit Risk Model API", version="1.0.0")
model = None


# --- LIFESPAN EVENTS (Startup) ---
@app.on_event("startup")
def load_model():
    """Load the model from the MLflow Model Registry upon server startup."""
    global model
    logger.info("Attempting to load model from MLflow registry...")
    try:
        # 1. Get the URI for the model in the specified stage
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[STAGE])

        if not versions:
            error_msg = (
                f"Model '{MODEL_NAME}' in stage '{STAGE}' not found in registry."
            )
            logger.error(error_msg)
            # Raise an exception to prevent the server from starting with a missing model
            raise RuntimeError(error_msg)

        version = versions[0].version
        model_uri = f"models:/{MODEL_NAME}/{version}"

        # 2. Load the model artifact
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(
            f"Successfully loaded model '{MODEL_NAME}' version {version} from {STAGE} stage."
        )

    except Exception as e:
        logger.exception(f"FATAL ERROR during model loading: {e}")
        # Re-raise or exit to indicate a critical startup failure
        raise RuntimeError(
            "API failed to initialize due to critical model loading error."
        ) from e


# --- API ENDPOINT ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(customer_data: CustomerFeatures):
    """Accepts customer features and returns the risk probability."""

    logger.info(
        f"Received prediction request for customer data: {customer_data.dict()}"
    )

    if model is None:
        logger.error(
            "Prediction request failed: Model is not loaded (or initialization failed)."
        )
        # Return a 503 Service Unavailable if model is unexpectedly missing
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable for prediction.",
        )

    try:
        # 1. Convert the Pydantic input model to a pandas DataFrame
        input_df = pd.DataFrame([customer_data.dict()])

        # 2. Use the MLflow loaded model to get predictions
        probabilities = model.predict(input_df)

        # We want the probability of being High Risk (Class 1)
        risk_prob = probabilities[:, 1][0]

        # 3. Determine the risk label
        threshold = 0.5
        risk_label = "High Risk" if risk_prob >= threshold else "Low Risk"

        logger.info(
            f"Prediction successful. Probability: {risk_prob:.4f}, Label: {risk_label}"
        )

        # 4. Return the structured response
        return PredictionResponse(risk_probability=risk_prob, risk_label=risk_label)

    except Exception as e:
        # Catch unexpected errors during prediction (e.g., data format issue during transformation)
        logger.exception(f"Unexpected error during prediction process: {e}")
        # Return a 500 Internal Server Error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during model prediction: {e}",
        )
