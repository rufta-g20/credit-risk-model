import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # Explicitly import KMeans for clustering
from xverse.transformer import WOE
import warnings

# Suppress warnings for cleaner output during execution
warnings.filterwarnings("ignore", category=FutureWarning)

# --- GLOBAL CONSTANTS ---
CUTOFF_DATE = datetime(2019, 2, 14)
N_CLUSTERS = 3  # <-- UPDATED: Changed from 2 to 3 clusters for segmentation
RANDOM_STATE = 42  # Ensures reproducible clustering (Instruction 2)

# =========================================================================
# 1. Custom Transformer: RFM Aggregation and Target Creation (Proxy)
# =========================================================================


class RFMAggregator(BaseEstimator, TransformerMixin):
    """
    Groups data by CustomerId, calculates RFM metrics, and creates the Proxy Target
    ('is_high_risk') using K-Means clustering (Task 4).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # --- Data Cleaning & Feature Extraction ---
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
        df.drop(
            columns=["CountryCode", "CurrencyCode", "Value"],
            inplace=True,
            errors="ignore",
        )
        df["TransactionStartTime"] = df["TransactionStartTime"].dt.tz_localize(None)

        # Calculate time difference for Recency
        df["Recency_Days"] = (CUTOFF_DATE - df["TransactionStartTime"]).dt.days

        # Separate Debit (spending) and Credit (receiving) transactions
        df["Debit_Amount"] = df["Amount"].apply(lambda x: abs(x) if x > 0 else 0)
        df["Credit_Amount"] = df["Amount"].apply(lambda x: abs(x) if x < 0 else 0)

        # --- RFM Aggregation (Instruction 1) ---
        rfm_df = df.groupby("CustomerId").agg(
            # Monetary (Debit/Spending)
            M_Debit_Total=("Debit_Amount", "sum"),
            M_Debit_Mean=("Debit_Amount", "mean"),
            M_Debit_Std=("Debit_Amount", "std"),
            # Frequency
            F_Count=("TransactionId", "count"),
            # Recency (Min Days = Most Recent Transaction)
            R_Min_Days=("Recency_Days", "min"),
            # Other features
            Channel_Mode=(
                "ChannelId",
                lambda x: x.mode()[0] if not x.mode().empty else "Missing",
            ),
            Product_Mode=(
                "ProductCategory",
                lambda x: x.mode()[0] if not x.mode().empty else "Missing",
            ),
        )

        # Handle aggregation NaNs (single transactions have no std dev)
        rfm_df["M_Debit_Std"].fillna(0, inplace=True)

        # --- K-Means Clustering and Target Assignment (Instructions 2 & 3) ---

        # 1. Pre-process RFM features (Scale)
        X_cluster = rfm_df[["R_Min_Days", "F_Count", "M_Debit_Total"]].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        # 2. Cluster Customers into 3 groups (Instruction 2)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        rfm_df["Risk_Cluster"] = kmeans.fit_predict(X_scaled)

        # 3. Analyze and Define High-Risk Label (Instruction 3)
        # High Risk = Least Engaged: High Recency (R_Min_Days) AND Low Frequency (F_Count)
        cluster_analysis = (
            rfm_df.groupby("Risk_Cluster")
            .agg(
                Avg_Recency=("R_Min_Days", "mean"),
                Avg_Frequency=("F_Count", "mean"),
                Avg_Monetary=("M_Debit_Total", "mean"),
            )
            .sort_values(by=["Avg_Recency", "Avg_Frequency"], ascending=[False, True])
        )  # Sort by highest Recency (least engaged)

        # The cluster with the highest average Recency is designated as the least engaged/High-Risk group (1).
        high_risk_cluster_id = cluster_analysis.index[0]

        # 4. Create the final target variable (proxy)
        rfm_df["is_high_risk"] = np.where(
            rfm_df["Risk_Cluster"] == high_risk_cluster_id, 1, 0
        )  # Final column name is 'is_high_risk'

        # Drop the intermediate cluster ID
        rfm_df.drop(columns=["Risk_Cluster"], inplace=True)

        return rfm_df


# =========================================================================
# 2. Custom Transformer: Weight of Evidence (WoE)
# =========================================================================


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Applies WoE transformation to specified features using the proxy target.
    """

    def __init__(self, features_to_woe=None):
        self.features_to_woe = features_to_woe
        self.woe_fit = WOE()

    def fit(self, X, y=None):
        # Identify features dynamically, excluding the target column
        self.features_to_woe = [col for col in X.columns if col not in ["is_high_risk"]]

        # xverse WOE requires the target (y) to be passed separately
        # X['is_high_risk'] is the target created in the previous step
        self.woe_fit.fit(X[self.features_to_woe], X["is_high_risk"])
        return self

    def transform(self, X):
        X_feats = X.drop(
            columns=["is_high_risk"], errors="ignore"
        )  # Drop target before transformation

        X_woe = self.woe_fit.transform(X_feats[self.features_to_woe])
        X_woe.columns = [f"{col}_WOE" for col in X_woe.columns]

        X_transformed = X.merge(X_woe, left_index=True, right_index=True, how="left")

        # Drop the original features now that we have the WoE version
        X_transformed.drop(columns=self.features_to_woe, inplace=True, errors="ignore")

        return X_transformed


# =========================================================================
# 3. Full Feature Engineering Pipeline
# =========================================================================


def create_full_pipeline():
    """
    Creates and returns the complete scikit-learn pipeline for data processing.
    """

    # STEP A: RFM Aggregation and Proxy Target Creation (Task 4)
    rfm_step = ("rfm_aggregator", RFMAggregator())

    # STEP B: WoE Transformation (Task 3)
    woe_step = ("woe_transformer", WOETransformer())

    # The full pipeline first aggregates the raw data and then transforms the features using WoE.
    full_pipeline = Pipeline(steps=[rfm_step, woe_step])

    return full_pipeline


# =========================================================================
# 4. Usage Example (for testing/debugging and target analysis)
# =========================================================================

if __name__ == "__main__":
    print("Running proxy target and feature engineering pipeline test...")

    try:
        raw_df = pd.read_csv("../data/raw/data.csv")
    except FileNotFoundError:
        print("Error: data.csv not found. Cannot run test.")
        exit()

    pipeline = create_full_pipeline()

    # Fit and Transform the data
    transformed_data = pipeline.fit_transform(raw_df)

    # Separate features (X) and the proxy target (y)
    X_processed = transformed_data.drop(columns=["is_high_risk"])
    y_target = transformed_data["is_high_risk"]

    print("\n--- Processed Feature Set (X) ---")
    print(X_processed.head())
    print(f"\nTotal Customers (Rows): {X_processed.shape[0]}")

    # Check the newly created target balance (Deliverable proof for Task 4)
    print("\n--- Proxy Target (is_high_risk) Distribution ---")
    print(y_target.value_counts(normalize=True))

    print(f"\nTarget Variable Name: is_high_risk")
    print(f"High-Risk (1) Proportion: {y_target.mean():.2%}")
    print(
        "Task 4: Proxy Target Engineering is complete. Data is ready for model training."
    )
