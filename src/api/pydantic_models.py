from pydantic import BaseModel, Field
from typing import Optional, List
from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    """Features representing a single customer's data before encoding."""
    R_Min_Days: float
    F_Count: float
    M_Debit_Total: float
    M_Debit_Mean: float
    M_Debit_Std: float
    # New Temporal Features (derived from the mode of customer transactions)
    Hour_Mode: int = Field(..., ge=0, le=23, description="Most frequent hour of transaction")
    Channel_Mode: str = Field(..., description="Most frequent ChannelId used.")
    Product_Mode: str = Field(..., description="Most frequent ProductCategory used.")

    class Config:
        schema_extra = {
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

# Define the response structure for the /predict endpoint
class PredictionResponse(BaseModel):
    """The risk probability returned by the model."""

    risk_probability: float = Field(
        ...,
        description="The probability (0 to 1) that the customer is high risk (Class 1).",
    )
    risk_label: str = Field(
        ..., description="The classification label (High Risk or Low Risk)."
    )
