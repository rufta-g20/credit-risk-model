import pandas as pd
import numpy as np
import pytest
from src.data_processing import (
    CustomerFeatureExtractor, 
    ProxyTargetGenerator, 
    create_inference_pipeline, 
    ModelConfig
)

@pytest.fixture
def dummy_data():
    """Creates a larger, more diverse transaction DataFrame for robust testing."""
    np.random.seed(42)
    num_transactions = 50 

    data = {
        "TransactionId": [f"T{i}" for i in range(num_transactions)],
        "BatchId": ["B1"] * num_transactions,
        "CustomerId": [f"CUST{i}" for i in np.random.randint(1, 15, num_transactions)],
        "CurrencyCode": ["UGX"] * num_transactions,
        "CountryCode": [256] * num_transactions,
        "ProviderId": ["P1"] * num_transactions,
        "ProductId": ["PD1"] * num_transactions,
        "ProductCategory": np.random.choice(["airtime", "financial_services"], num_transactions),
        "ChannelId": np.random.choice(["ChannelId_2", "ChannelId_3"], num_transactions),
        "Amount": np.random.choice([1000, -500, 5000, -1000, 200], num_transactions),
        "Value": [1000] * num_transactions,
        "TransactionStartTime": pd.to_datetime("2018-11-15 12:00:00") 
                                + pd.to_timedelta(np.random.randint(0, 90, num_transactions), unit="D"),
        "PricingStrategy": [2] * num_transactions,
        "FraudResult": [0] * num_transactions,
    }
    return pd.DataFrame(data)

def test_extractor_output_shape(dummy_data):
    """Test: Check that the extractor outputs one row per unique customer."""
    extractor = CustomerFeatureExtractor()
    expected_rows = dummy_data["CustomerId"].nunique()
    rfm_df = extractor.transform(dummy_data)
    
    assert rfm_df.shape[0] == expected_rows

def test_temporal_feature_extraction(dummy_data):
    """Test: Verify that temporal features (Hour_Mode) are derived during extraction."""
    extractor = CustomerFeatureExtractor()
    rfm_df = extractor.transform(dummy_data)
    
    assert "Hour_Mode" in rfm_df.columns
    assert rfm_df["Hour_Mode"].min() >= 0
    assert rfm_df["Hour_Mode"].max() <= 23

def test_target_generation_logic(dummy_data):
    """Test: Verify that ProxyTargetGenerator creates binary 'is_high_risk' labels."""
    # 1. Extract features
    extractor = CustomerFeatureExtractor()
    rfm_df = extractor.transform(dummy_data)
    
    # 2. Generate labels
    labeller = ProxyTargetGenerator()
    labeled_df = labeller.generate_labels(rfm_df)
    
    assert "is_high_risk" in labeled_df.columns
    assert set(labeled_df["is_high_risk"].unique()).issubset({0, 1})

def test_encoding_columns_presence(dummy_data):
    """Test: Ensure both WoE and One-Hot Encoded columns exist after the inference pipeline."""
    # Setup the full flow as it would happen in training
    extractor = CustomerFeatureExtractor()
    rfm_df = extractor.transform(dummy_data)
    
    labeller = ProxyTargetGenerator()
    labeled_df = labeller.generate_labels(rfm_df)
    
    X = labeled_df.drop(columns=["is_high_risk"])
    y = labeled_df["is_high_risk"]
    
    pipeline = create_inference_pipeline()
    processed_df = pipeline.fit_transform(X, y)
    
    # Check for WoE columns (appended with _WOE)
    woe_cols = [col for col in processed_df.columns if "_WOE" in col]
    assert len(woe_cols) > 0
    
    # Check for One-Hot Encoded columns (prefixed with Channel_Mode_)
    ohe_cols = [col for col in processed_df.columns if "Channel_Mode_" in col]
    assert len(ohe_cols) > 0

def test_extractor_robustness():
    """Test: Ensures the extractor handles minimal data without crashing."""
    mock_data = pd.DataFrame({
        "TransactionId": [1, 2],
        "CustomerId": ["C1", "C1"], # Multiple transactions for same customer
        "TransactionStartTime": ["2019-02-13 12:00:00", "2019-02-13 13:00:00"],
        "Amount": [100.0, -50.0],
        "ChannelId": ["A", "A"],
        "ProductCategory": ["P1", "P1"]
    })
    
    extractor = CustomerFeatureExtractor()
    rfm_df = extractor.transform(mock_data)
    
    assert rfm_df.shape[0] == 1  # Aggregated to one customer
    assert rfm_df["M_Debit_Total"].iloc[0] == 100.0 # Only positive amounts are debits in our logic
    assert rfm_df["F_Count"].iloc[0] == 2