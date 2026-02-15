from pydantic import BaseModel, Field
from typing import Optional, Dict

class CustomerFeatures(BaseModel):
    """Features representing a single customer's aggregated data."""
    R_Min_Days: float = Field(..., ge=0, description="Recency in days")
    F_Count: float = Field(..., ge=1, description="Frequency of transactions")
    M_Debit_Total: float = Field(..., ge=0, description="Total monetary value of debits")
    M_Debit_Mean: float = Field(..., ge=0)
    M_Debit_Std: float = Field(..., ge=0)
    Hour_Mode: int = Field(..., ge=0, le=23)
    Channel_Mode: str = Field(..., description="E.g., ChannelId_3")
    Product_Mode: str = Field(..., description="E.g., airtime")

    model_config = {
        "json_schema_extra": {
            "example": {
                "R_Min_Days": 5.0,
                "F_Count": 10.0,
                "M_Debit_Total": 5500.0,
                "M_Debit_Mean": 550.0,
                "M_Debit_Std": 150.0,
                "Hour_Mode": 14,
                "Channel_Mode": "ChannelId_3",
                "Product_Mode": "airtime",
            }
        }
    }

class PredictionResponse(BaseModel):
    """Production-grade response with placeholder for SHAP justifications."""
    risk_probability: float
    risk_label: str
    # SHAP feature importance mapping (to be populated in Friday's task)
    justification: Optional[Dict[str, float]] = Field(
        None, description="Feature importance scores justifying the prediction."
    )
    status: str = "success"