import pandas as pd
import numpy as np
import pytest
from src.data_processing import create_full_pipeline, RFMAggregator

@pytest.fixture
def dummy_data():
    """Creates a larger, more diverse transaction DataFrame for robust testing."""
    np.random.seed(42)
    num_transactions = 50  # Increased for better cluster stability

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

def test_pipeline_output_shape(dummy_data):
    """Test: Check that the pipeline outputs one row per unique customer."""
    pipeline = create_full_pipeline()
    expected_rows = dummy_data["CustomerId"].nunique()
    processed_df = pipeline.fit_transform(dummy_data)
    
    assert processed_df.shape[0] == expected_rows

def test_temporal_feature_extraction(dummy_data):
    """Test: Verify that temporal features (Hour_Mode) are derived during transformation."""
    aggregator = RFMAggregator()
    rfm_df = aggregator.transform(dummy_data)
    
    assert "Hour_Mode" in rfm_df.columns
    assert rfm_df["Hour_Mode"].min() >= 0
    assert rfm_df["Hour_Mode"].max() <= 23

def test_encoding_columns_presence(dummy_data):
    """Test: Ensure both WoE and One-Hot Encoded columns exist after pipeline."""
    pipeline = create_full_pipeline()
    processed_df = pipeline.fit_transform(dummy_data)
    
    # Check for WoE columns (appended with _WOE)
    woe_cols = [col for col in processed_df.columns if "_WOE" in col]
    assert len(woe_cols) > 0
    
    # Check for One-Hot Encoded columns (prefixed with Channel_Mode_)
    ohe_cols = [col for col in processed_df.columns if "Channel_Mode_" in col]
    assert len(ohe_cols) > 0

def test_target_column_logic(dummy_data):
    """Test: Verify that is_high_risk is binary and present."""
    pipeline = create_full_pipeline()
    processed_df = pipeline.fit_transform(dummy_data)
    
    assert "is_high_risk" in processed_df.columns
    assert set(processed_df["is_high_risk"].unique()).issubset({0, 1})

def test_rfm_aggregator_robustness():
    """Test: Ensures the aggregator handles minimal unique customers for 3-cluster KMeans."""
    mock_data = pd.DataFrame({
        "TransactionId": [1, 2, 3],
        "CustomerId": ["C1", "C2", "C3"], # Exactly 3 to satisfy N_CLUSTERS
        "TransactionStartTime": ["2019-02-13 12:00:00"] * 3,
        "Amount": [100.0, 200.0, 300.0],
        "ChannelId": ["A", "B", "A"],
        "ProductCategory": ["P1", "P1", "P2"]
    })
    
    aggregator = RFMAggregator()
    # Should not raise ValueError even with small sample size
    rfm_df = aggregator.transform(mock_data)
    assert rfm_df.shape[0] == 3
    assert rfm_df["M_Debit_Std"].iloc[0] == 0.0