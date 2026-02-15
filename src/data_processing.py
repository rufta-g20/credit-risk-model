import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Union
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from xverse.transformer import WOE
import pandas.core.algorithms as algos
import warnings

# Fix for xverse compatibility with pandas 1.5.x+
if not hasattr(algos, 'quantile'):
    algos.quantile = np.quantile

warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass(frozen=True)
class ModelConfig:
    """Centralized configuration for the Credit Risk Model."""
    CUTOFF_DATE: datetime = datetime(2019, 2, 14)
    N_CLUSTERS: int = 3
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    UPPER_QUANTILE: float = 0.95

# =========================================================================
# 1. Feature Extraction (Used by both Training AND API)
# =========================================================================

class CustomerFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Modular transformer to extract RFM and Temporal features from raw transactions.
    Separated from target generation for API compatibility.
    """
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Temporal Cleaning
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"]).dt.tz_localize(None)
        df["TransactionHour"] = df["TransactionStartTime"].dt.hour
        
        # RFM Prep
        df["Recency_Days"] = (self.config.CUTOFF_DATE - df["TransactionStartTime"]).dt.days
        df["Debit_Amount"] = df["Amount"].apply(lambda x: abs(x) if x > 0 else 0)

        # Aggregation
        rfm_df = df.groupby("CustomerId").agg(
            M_Debit_Total=("Debit_Amount", "sum"),
            M_Debit_Mean=("Debit_Amount", "mean"),
            M_Debit_Std=("Debit_Amount", "std"),
            F_Count=("TransactionId", "count"),
            R_Min_Days=("Recency_Days", "min"),
            Hour_Mode=("TransactionHour", lambda x: x.mode()[0] if not x.mode().empty else 0),
            Channel_Mode=("ChannelId", lambda x: x.mode()[0] if not x.mode().empty else "Missing"),
            Product_Mode=("ProductCategory", lambda x: x.mode()[0] if not x.mode().empty else "Missing"),
        )

        rfm_df["M_Debit_Std"] = rfm_df["M_Debit_Std"].fillna(0)

        # Production Hardening: Clipping outliers
        for col in ["R_Min_Days", "F_Count", "M_Debit_Total"]:
            limit = rfm_df[col].quantile(self.config.UPPER_QUANTILE)
            rfm_df[col] = rfm_df[col].clip(upper=limit)

        return rfm_df

# =========================================================================
# 2. Target Generator (Used ONLY during Training/Labelling)
# =========================================================================

class ProxyTargetGenerator:
    """Logic to generate 'is_high_risk' labels using K-Means."""
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.kmeans = KMeans(
            n_clusters=self.config.N_CLUSTERS, 
            random_state=self.config.RANDOM_STATE, 
            n_init=10
        )

    def generate_labels(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        X_cluster = rfm_df[["R_Min_Days", "F_Count", "M_Debit_Total"]]
        X_scaled = (X_cluster - X_cluster.mean()) / (X_cluster.std() + 1e-6)
        
        clusters = self.kmeans.fit_predict(X_scaled)
        rfm_df["Risk_Cluster"] = clusters

        # Logic: High Recency + Low Frequency = High Risk
        analysis = (
            rfm_df.groupby("Risk_Cluster")
            .agg(Avg_R=("R_Min_Days", "mean"), Avg_F=("F_Count", "mean"))
            .sort_values(by=["Avg_R", "Avg_F"], ascending=[False, True])
        )

        high_risk_id = analysis.index[0]
        rfm_df["is_high_risk"] = (rfm_df["Risk_Cluster"] == high_risk_id).astype(int)
        return rfm_df.drop(columns=["Risk_Cluster"])

# =========================================================================
# 3. Encoding Logic
# =========================================================================

class FeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.woe_fit = WOE()
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.ohe_cols = ["Channel_Mode"]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.ohe.fit(X[self.ohe_cols])
        self.woe_cols = [c for c in X.columns if c not in self.ohe_cols and c != "is_high_risk"]
        
        # In scikit-learn pipelines, y is passed separately
        target = y if y is not None else X["is_high_risk"]
        self.woe_fit.fit(X[self.woe_cols], target)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        # OHE
        ohe_data = self.ohe.transform(X_copy[self.ohe_cols])
        ohe_df = pd.DataFrame(ohe_data, index=X.index, columns=self.ohe.get_feature_names_out())
        
        # WoE
        X_woe = self.woe_fit.transform(X_copy[self.woe_cols])
        X_woe.columns = [f"{c}_WOE" for c in X_woe.columns]
        X_woe = X_woe.replace('NA', np.nan).apply(pd.to_numeric, errors='coerce').fillna(0)

        return pd.concat([X_woe, ohe_df], axis=1)

def create_inference_pipeline() -> Pipeline:
    """The pipeline used for predicting on new data."""
    return Pipeline([
        ('encoder', FeatureEncoder())
    ])