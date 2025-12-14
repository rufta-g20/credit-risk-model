import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
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
    """Calculates and logs standard classification metrics."""
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
def train_and_log_model(X_train, X_test, y_train, y_test, model, params, model_name):
    """Trains model, performs tuning, and logs results to MLflow."""

    with mlflow.start_run(run_name=f"{model_name}_Run") as run:
        print(f"\n--- Starting MLflow Run for {model_name} ---")

        # 1. Hyperparameter Tuning (Instruction 4)
        print("Performing Grid Search for Hyperparameter Tuning...")

        # Use a simplified search for brevity; expand as needed
        grid_search = GridSearchCV(
            estimator=model, param_grid=params, scoring="roc_auc", cv=3, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best Parameters found: {best_params}")

        # 2. Log Parameters (Instruction 5)
        mlflow.log_params(best_params)
        mlflow.set_tag("Model Type", model_name)

        # 3. Model Evaluation (Instruction 6)
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        test_auc = log_metrics(y_test, y_pred, y_prob, model_name)

        # 4. Log Model Artifacts (Instruction 5)
        mlflow.sklearn.log_model(best_model, "model", registered_model_name=model_name)

        print(f"MLflow Run ID: {run.info.run_id}")
        return test_auc, run.info.run_id


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Credit Risk Scorecard Development")

    # Load and process data using the pipeline from Task 3/4
    try:
        raw_df = pd.read_csv("./data/raw/data.csv")
    except FileNotFoundError:
        print(
            "Error: data.csv not found. Please ensure it is in the data/raw/ directory."
        )
        exit()

    print("--- 1. Data Processing (Running Pipeline) ---")
    pipeline = create_full_pipeline()
    transformed_data = pipeline.fit_transform(raw_df)

    X = transformed_data.drop(columns=["is_high_risk"])
    y = transformed_data["is_high_risk"]

    print(f"Final feature set shape: {X.shape}")

    # 2. Data Preparation (Train-Test Split - Instruction 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(
        f"Train/Test split complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}"
    )

    # 3. Model Selection and Parameters (Instruction 3)
    models_to_run = {
        "LogisticRegression_Scorecard": {
            "model": LogisticRegression(solver="liblinear", random_state=RANDOM_STATE),
            # L1 penalty helps with feature selection and produces sparser models
            "params": {"penalty": ["l1", "l2"], "C": [0.1, 1.0, 10.0]},
        },
        "DecisionTree_Benchmark": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {"max_depth": [5, 7, 9], "min_samples_leaf": [10, 20]},
        },
    }

    best_auc = 0
    best_model_name = ""

    # Run experiments and log to MLflow
    results = {}
    for name, config in models_to_run.items():
        auc, run_id = train_and_log_model(
            X_train, X_test, y_train, y_test, config["model"], config["params"], name
        )
        results[name] = {"auc": auc, "run_id": run_id}

        if auc > best_auc:
            best_auc = auc
            best_model_name = name

    # 4. Final Best Model Registration (Instruction 5)
    print(f"\n--- Best Model Identified ---")
    print(f"Best Model: {best_model_name} with ROC-AUC: {best_auc:.4f}")

    # Register the best run as the Staging model
    # Note: MLflow automatically registers the model during log_model() above,
    # but here we confirm the best one.
    client = mlflow.tracking.MlflowClient()

    # We retrieve the latest version of the model and transition it
    try:
        latest_version = client.get_latest_versions(best_model_name, stages=["None"])[0]
        client.transition_model_version_stage(
            name=best_model_name, version=latest_version.version, stage="Staging"
        )
        print(
            f"Model '{best_model_name}' (Version {latest_version.version}) transitioned to 'Staging' in MLflow Model Registry."
        )
    except Exception as e:
        print(f"Could not transition model stage. Error: {e}")

    print("\nTask 5: Model Training and Tracking complete.")
    print("To view results, navigate to your project root and run 'mlflow ui'")
