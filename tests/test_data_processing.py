import pandas as pd
import numpy as np
import pytest
from src.data_processing import create_full_pipeline
from src.data_processing import RFMAggregator

# Define a fixture with a larger, more diverse dataset (10 customers, 30 transactions)
@pytest.fixture
def dummy_data():
    """Creates a larger, more diverse transaction DataFrame for robust testing."""
    np.random.seed(42)
    num_transactions = 30
    
    data = {
        'TransactionId': [f'T{i}' for i in range(num_transactions)],
        'BatchId': ['B1'] * num_transactions,
        'AccountId': [f'A{i}' for i in np.random.randint(1, 11, num_transactions)],
        'SubscriptionId': [f'S{i}' for i in np.random.randint(1, 11, num_transactions)],
        'CustomerId': [f'CUST{i}' for i in np.random.randint(1, 11, num_transactions)], # 10 unique customers
        'CurrencyCode': ['UGX'] * num_transactions,
        'CountryCode': [256] * num_transactions,
        'ProviderId': np.random.choice([f'P{i}' for i in range(1, 5)], num_transactions),
        'ProductId': np.random.choice([f'PD{i}' for i in range(1, 4)], num_transactions),
        'ProductCategory': np.random.choice(['CAT_A', 'CAT_B', 'CAT_C'], num_transactions),
        'ChannelId': np.random.choice(['Web', 'Mobile'], num_transactions),
        # Ensure diverse amounts (positive and negative)
        'Amount': np.random.choice([1000, -500, 2000, 100, -100, 50, 500, 200, 300, 1500, 5000, -1000], num_transactions),
        'Value': np.abs(np.random.choice([1000, -500, 2000, 100, -100, 50, 500, 200, 300, 1500, 5000, -1000], num_transactions)),
        # Simulate transaction times spanning the 3-month window
        'TransactionStartTime': pd.to_datetime('2018-11-15 00:00:00') + pd.to_timedelta(np.random.randint(0, 90, num_transactions), unit='D'),
        'PricingStrategy': np.random.randint(0, 4, num_transactions),
        'FraudResult': np.random.choice([0, 1], num_transactions, p=[0.9, 0.1]),
    }
    return pd.DataFrame(data)

def test_pipeline_output_shape(dummy_data):
    """Test 1: Check that the pipeline outputs one row per unique customer."""
    
    # Arrange
    pipeline = create_full_pipeline()
    expected_rows = dummy_data['CustomerId'].nunique()
    
    # Act
    processed_df = pipeline.fit_transform(dummy_data)
    
    # Assert
    assert processed_df.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {processed_df.shape[0]} rows."

def test_target_column_existence(dummy_data):
    """Test 2: Check that the proxy target column 'is_high_risk' is created."""
    
    # Arrange
    pipeline = create_full_pipeline()
    required_column = 'is_high_risk'
    
    # Act
    processed_df = pipeline.fit_transform(dummy_data)
    
    # Assert
    assert required_column in processed_df.columns, f"Expected column '{required_column}' not found in processed data."

def test_rfm_aggregator_handles_single_transactions():
    """
    Test for code robustness: Ensures customers with only one transaction 
    correctly get M_Debit_Std (Standard Deviation) imputed to 0, which 
    is a key step in data preparation.
    """
    # Arrange: Mock data where Customer CUST102 has only one transaction.
    mock_data = pd.DataFrame({
        'TransactionId': [1, 2, 3],
        'CustomerId': ['CUST101', 'CUST102', 'CUST101'], # CUST102 is the single-transaction customer
        'TransactionStartTime': ['2019-02-13 12:00:00', '2019-02-13 12:00:00', '2019-02-12 12:00:00'],
        'Amount': [500.0, 1000.0, 100.0], # CUST102 Debit amount is 1000.0
        'ChannelId': ['APP', 'WEB', 'APP'],
        'ProductCategory': ['P1', 'P2', 'P1']
    })
    
    # Instantiate the custom transformer
    aggregator = RFMAggregator()
    
    # Act: Transform the raw data to RFM features
    rfm_df = aggregator.transform(mock_data)
    
    # Assert
    # 1. Check the overall shape (2 unique customers)
    assert rfm_df.shape[0] == 2
    
    # 2. Check that the standard deviation for the single-transaction customer (CUST102) is 0
    assert rfm_df.loc['CUST102', 'M_Debit_Std'] == 0.0