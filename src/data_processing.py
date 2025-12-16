import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Added OneHotEncoder
from sklearn.cluster import KMeans
from xverse.transformer import WOE
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

"""
BUSINESS LOGIC FOR PROXY TARGET:
We designate the 'Least Engaged' cluster as High Risk (1). 
In credit scoring, customers with high Recency (long time since last seen) 
and low Frequency/Monetary value represent higher uncertainty and risk 
as they lack a consistent repayment or transaction history.
"""

# --- GLOBAL CONSTANTS ---
CUTOFF_DATE = datetime(2019, 2, 14)
N_CLUSTERS = 3 
RANDOM_STATE = 42 

# =========================================================================
# 1. Custom Transformer: RFM Aggregation and Target Creation (Proxy)
# =========================================================================

class RFMAggregator(BaseEstimator, TransformerMixin):
    """
    Groups data by CustomerId, calculates RFM metrics, extracts temporal features,
    and creates the Proxy Target ('is_high_risk') using K-Means clustering.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # --- 1. Data Cleaning & Temporal Feature Extraction (Feedback: Task 3) ---
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
        df["TransactionStartTime"] = df["TransactionStartTime"].dt.tz_localize(None)

        # Explicitly derive temporal features as requested in feedback
        df["TransactionHour"] = df["TransactionStartTime"].dt.hour
        df["TransactionDay"] = df["TransactionStartTime"].dt.day
        df["TransactionMonth"] = df["TransactionStartTime"].dt.month
        df["TransactionYear"] = df["TransactionStartTime"].dt.year

        df.drop(columns=["CountryCode", "CurrencyCode", "Value"], inplace=True, errors="ignore")

        # Calculate time difference for Recency
        df["Recency_Days"] = (CUTOFF_DATE - df["TransactionStartTime"]).dt.days

        # Separate Debit and Credit transactions
        df["Debit_Amount"] = df["Amount"].apply(lambda x: abs(x) if x > 0 else 0)
        df["Credit_Amount"] = df["Amount"].apply(lambda x: abs(x) if x < 0 else 0)

        # --- 2. RFM Aggregation ---
        rfm_df = df.groupby("CustomerId").agg(
            M_Debit_Total=("Debit_Amount", "sum"),
            M_Debit_Mean=("Debit_Amount", "mean"),
            M_Debit_Std=("Debit_Amount", "std"),
            F_Count=("TransactionId", "count"),
            R_Min_Days=("Recency_Days", "min"),
            # Capture temporal modes to keep them in the customer-level dataset
            Hour_Mode=("TransactionHour", lambda x: x.mode()[0] if not x.mode().empty else 0),
            Channel_Mode=("ChannelId", lambda x: x.mode()[0] if not x.mode().empty else "Missing"),
            Product_Mode=("ProductCategory", lambda x: x.mode()[0] if not x.mode().empty else "Missing"),
        )

        rfm_df["M_Debit_Std"].fillna(0, inplace=True)

        # --- 3. K-Means Clustering & Target Assignment (Feedback: Task 4 Robustness) ---
        X_cluster = rfm_df[["R_Min_Days", "F_Count", "M_Debit_Total"]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        rfm_df["Risk_Cluster"] = kmeans.fit_predict(X_scaled)

        # Robustness check: Ensure clusters aren't empty
        if rfm_df["Risk_Cluster"].nunique() < N_CLUSTERS:
            warnings.warn(f"Clustering generated only {rfm_df['Risk_Cluster'].nunique()} clusters.")

        # Business Logic: In Credit Risk, "Least Engaged" customers (High Recency, Low Frequency)
        # often represent a high risk because there is less behavioral data to prove reliability,
        # or they may be dormant users returning for a large, potentially fraudulent transaction.
        cluster_analysis = (
            rfm_df.groupby("Risk_Cluster")
            .agg(Avg_Recency=("R_Min_Days", "mean"), Avg_Frequency=("F_Count", "mean"))
            .sort_values(by=["Avg_Recency", "Avg_Frequency"], ascending=[False, True])
        )

        high_risk_cluster_id = cluster_analysis.index[0]
        rfm_df["is_high_risk"] = np.where(rfm_df["Risk_Cluster"] == high_risk_cluster_id, 1, 0)
        rfm_df.drop(columns=["Risk_Cluster"], inplace=True)

        return rfm_df

# =========================================================================
# 2. Custom Transformer: Mixed Encoding (WoE + One-Hot) (Feedback: Task 3)
# =========================================================================

class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Applies WoE to numerical, temporal, and high-cardinality features 
    while using One-Hot Encoding for low-cardinality categorical features.
    """
    def __init__(self):
        self.woe_fit = WOE()
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # Channel_Mode is OHE; Product_Mode and Hour_Mode will be WoE
        self.ohe_cols = ["Channel_Mode"] 

    def fit(self, X, y=None):
        # 1. Fit OneHotEncoder for categorical features
        self.ohe.fit(X[self.ohe_cols])
        
        # 2. Identify columns for WoE (Everything except OHE cols and the target)
        # This explicitly includes Hour_Mode, R_Min_Days, F_Count, etc.
        self.woe_cols = [c for c in X.columns if c not in self.ohe_cols and c != "is_high_risk"]
        
        # 3. Fit WoE
        self.woe_fit.fit(X[self.woe_cols], X["is_high_risk"])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        # 1. Apply OHE transformation
        ohe_features = self.ohe.transform(X_transformed[self.ohe_cols])
        ohe_df = pd.DataFrame(
            ohe_features, 
            index=X_transformed.index, 
            columns=self.ohe.get_feature_names_out(self.ohe_cols)
        )
        
        # 2. Apply WoE transformation
        # This turns raw values (like Hour 14) into a risk-weighted score
        X_woe = self.woe_fit.transform(X_transformed[self.woe_cols])
        X_woe.columns = [f"{col}_WOE" for col in X_woe.columns]
        
        # 3. Combine: [Target] + [WoE Features] + [One-Hot Features]
        # We keep 'is_high_risk' for training, but it will be dropped in the final X_train
        final_df = pd.concat([X_transformed[["is_high_risk"]], X_woe, ohe_df], axis=1)
        
        return final_df