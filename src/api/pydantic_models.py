from pydantic import BaseModel, Field
from typing import Optional, List


# Define the input structure for the /predict endpoint (features after RFMAggregation)
class CustomerFeatures(BaseModel):
    """Features representing a single customer's RFM and categorical data."""

    R_Min_Days: float = Field(
        ..., description="Recency (Minimum Days since last transaction)."
    )
    F_Count: float = Field(..., description="Frequency (Total transaction count).")
    M_Debit_Total: float = Field(
        ..., description="Monetary (Total debit/spending amount)."
    )
    M_Debit_Mean: float = Field(
        ..., description="Monetary (Mean debit/spending amount)."
    )
    M_Debit_Std: float = Field(
        ..., description="Monetary (Standard Deviation of debit/spending amount)."
    )
    Channel_Mode: str = Field(..., description="Most frequent ChannelId used.")
    Product_Mode: str = Field(..., description="Most frequent ProductCategory used.")

    class Config:
        # Example data for FastAPI documentation
        schema_extra = {
            "example": {
                "R_Min_Days": 5.0,
                "F_Count": 10.0,
                "M_Debit_Total": 5500.0,
                "M_Debit_Mean": 550.0,
                "M_Debit_Std": 150.0,
                "Channel_Mode": "APP",
                "Product_Mode": "P4",
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
