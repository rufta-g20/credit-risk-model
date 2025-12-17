import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from src.data_processing import create_full_pipeline, RFMAggregator
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- GLOBAL CONFIGURATION ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_TRACKING_URI = "./mlruns"

# --- HELPER FUNCTION: Metric Logging ---
def log_metrics(y_true, y_pred, y_prob, model_name):
    """Calculates and logs standard classification metrics to MLflow."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob[:, 1]),
    }
    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    print(f"\n--- {model_name} Test Metrics ---")
    print(classification_report(y_true, y_pred))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    return metrics["roc_auc"]

# --- MODEL TRAINING AND TRACKING ---
def train_and_log_model(X_train_raw, X_test_raw, y_train, y_test, model_obj, params, model_name):
    """
    Trains model using a production pipeline (Encoder + Classifier), 
    optimizes hyperparameters, and logs artifacts/model to MLflow.
    """
    with mlflow.start_run(run_name=f"{model_name}_Run") as run:
        print(f"\n--- Starting MLflow Run for {model_name} ---")

        # 1. Initialize Hyperparameter Search
        # We search on the classifier alone for speed
        grid_search = GridSearchCV(
            estimator=model_obj, 
            param_grid=params, 
            scoring="roc_auc", 
            cv=5, 
            n_jobs=-1
        )

        # 2. Prepare Data for Grid Search
        # We must encode the data first because GridSearchCV doesn't see the full pipeline here
        encoder_temp = create_full_pipeline(include_aggregator=False)
        
        # We pass y_train so FeatureEncoder can calculate Weight of Evidence (WoE)
        X_train_encoded = encoder_temp.fit_transform(X_train_raw, y_train)
        
        # Drop the target column from features before fitting the classifier
        if "is_high_risk" in X_train_encoded.columns:
            X_train_encoded = X_train_encoded.drop(columns=["is_high_risk"])
        
        print(f"Fitting GridSearch for {model_name}...")
        grid_search.fit(X_train_encoded, y_train)

        # 3. Create the Final Production Pipeline
        # This combines your FeatureEncoder and the BEST tuned classifier into one object
        final_pipeline = create_full_pipeline(include_aggregator=False)
        final_pipeline.steps.append(('classifier', grid_search.best_estimator_))
        
        # 4. Fit the entire pipeline on RAW data
        # This ensures the pipeline learns how to handle raw features end-to-end
        final_pipeline.fit(X_train_raw, y_train)

        # 5. Evaluation on RAW test data
        # The pipeline automatically handles encoding for the test set
        y_prob = final_pipeline.predict_proba(X_test_raw)
        y_pred = final_pipeline.predict(X_test_raw)
        test_auc = log_metrics(y_test, y_pred, y_prob, model_name)

        # 6. Log Artifacts (Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
        disp.plot(cmap="Blues", ax=ax)
        plt.title(f"Confusion Matrix: {model_name}")
        mlflow.log_figure(fig, f"confusion_matrix_{model_name}.png")
        plt.close()

        # 7. Log Parameters and the Entire Pipeline to Registry
        mlflow.log_params(grid_search.best_params_)
        
        # We log the WHOLE pipeline so the API can use it without manual preprocessing
        mlflow.sklearn.log_model(
            sk_model=final_pipeline, 
            artifact_path="model", 
            registered_model_name=model_name
        )
        
        print(f"✅ Successfully logged {model_name} to MLflow.")
        return test_auc, run.info.run_id

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Credit Risk Scorecard Development")

    # 1. Load Data
    try:
        raw_df = pd.read_csv("./data/raw/data.csv")
    except FileNotFoundError:
        print("Error: data.csv not found.")
        exit()

    print("--- 1. Aggregating Data (Transaction -> Customer Level) ---")
    
    # 2. Aggregate raw transactions into customer profiles
    aggregator = RFMAggregator()
    customer_level_data = aggregator.transform(raw_df)
    
    # 3. Align X and y lengths
    y = customer_level_data["is_high_risk"]
    X_raw = customer_level_data.drop(columns=["is_high_risk"])

    print(f"Sample consistency check: X={len(X_raw)}, y={len(y)}")

    # 4. Split the aggregated customer data
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 5. Model Configurations
    models_to_run = {
        "LogisticRegression_Scorecard": {
            # ADDED: class_weight="balanced"
            "model": LogisticRegression(solver="liblinear", random_state=RANDOM_STATE, class_weight="balanced"),
            "params": {"penalty": ["l1", "l2"], "C": [0.1, 1.0, 10.0]},
        },
        "DecisionTree_Benchmark": {
            # ADDED: class_weight="balanced"
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            "params": {"max_depth": [5, 7, 9], "min_samples_leaf": [10, 20]},
        },
    }

    best_auc = 0
    best_model_name = ""

    # 6. Execute Experiments
    for name, config in models_to_run.items():
        auc, run_id = train_and_log_model(
            X_train, X_test, y_train, y_test, config["model"], config["params"], name
        )

        if auc > best_auc:
            best_auc = auc
            best_model_name = name

    # 7. Transition Best Model to Staging
    if best_model_name:
        client = mlflow.tracking.MlflowClient()
        try:
            latest_version = client.get_latest_versions(best_model_name, stages=["None"])[0]
            client.transition_model_version_stage(
                name=best_model_name, 
                version=latest_version.version, 
                stage="Staging"
            )
            print(f"\n✅ SUCCESS: {best_model_name} V{latest_version.version} moved to STAGING.")
        except Exception as e:
            print(f"❌ Registration failed: {e}")