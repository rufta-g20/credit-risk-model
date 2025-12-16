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
from src.data_processing import create_full_pipeline
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
    # Log metrics with a prefix to distinguish test results
    mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    print(f"\n--- {model_name} Test Metrics ---")
    print(classification_report(y_true, y_pred))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    return metrics["roc_auc"]

# --- MODEL TRAINING AND TRACKING ---
def train_and_log_model(X_train, X_test, y_train, y_test, model, params, model_name):
    """
    Trains model, performs tuning, logs CV results, metrics, and visual artifacts (Task 5).
    """

    with mlflow.start_run(run_name=f"{model_name}_Run") as run:
        print(f"\n--- Starting MLflow Run for {model_name} ---")

        # 1. Hyperparameter Tuning with Cross-Validation
        print("Performing Grid Search for Hyperparameter Tuning...")
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=params, 
            scoring="roc_auc", 
            cv=5,  # Increased to 5-fold for better CV logging
            n_jobs=-1,
            return_train_score=True
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        
        # 2. Log Parameters and CV Results (Feedback Requirement)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("mean_cv_roc_auc", grid_search.best_score_)
        mlflow.set_tag("Model Type", model_name)

        # 3. Model Evaluation on Test Set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)
        test_auc = log_metrics(y_test, y_pred, y_prob, model_name)

        # 4. Generate and Log Visual Artifacts (Feedback Requirement)
        # We create a confusion matrix to help diagnose type I and type II errors
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "High Risk"])
        disp.plot(cmap="Blues", ax=ax)
        plt.title(f"Confusion Matrix: {model_name}")
        
        cm_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        

        # 5. Log Model Artifacts to Registry
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="model", 
            registered_model_name=model_name
        )

        print(f"MLflow Run ID: {run.info.run_id} complete.")
        return test_auc, run.info.run_id

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Credit Risk Scorecard Development")

    # Load and process data
    try:
        raw_df = pd.read_csv("./data/raw/data.csv")
    except FileNotFoundError:
        print("Error: data.csv not found.")
        exit()

    print("--- 1. Data Processing (Aggregating RFM + Temporal Features) ---")
    pipeline = create_full_pipeline()
    transformed_data = pipeline.fit_transform(raw_df)

    # Features (X) and Proxy Target (y)
    X = transformed_data.drop(columns=["is_high_risk"])
    y = transformed_data["is_high_risk"]

    # 2. Reproducible Split (Stratified to maintain risk distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Model Configurations
    models_to_run = {
        "LogisticRegression_Scorecard": {
            "model": LogisticRegression(solver="liblinear", random_state=RANDOM_STATE),
            "params": {"penalty": ["l1", "l2"], "C": [0.1, 1.0, 10.0]},
        },
        "DecisionTree_Benchmark": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {"max_depth": [5, 7, 9], "min_samples_leaf": [10, 20]},
        },
    }

    best_auc = 0
    best_model_name = ""

    # Execute Experiments
    for name, config in models_to_run.items():
        auc, run_id = train_and_log_model(
            X_train, X_test, y_train, y_test, config["model"], config["params"], name
        )

        if auc > best_auc:
            best_auc = auc
            best_model_name = name

    # --- 4. Transition Best Model to Staging (SAFE VERSION) ---
    if best_model_name:
        client = mlflow.tracking.MlflowClient()
        try:
            # Get the latest version that was just registered
            latest_version = client.get_latest_versions(best_model_name, stages=["None"])[0]
            
            client.transition_model_version_stage(
                name=best_model_name, 
                version=latest_version.version, 
                stage="Staging"
            )
            print(f"\n✅ SUCCESS: {best_model_name} V{latest_version.version} moved to STAGING.")
        except Exception as e:
            print(f"❌ Registration failed: {e}")
    else:
        print("\n⚠️ WARNING: No models were successfully trained. Skipping registration.")