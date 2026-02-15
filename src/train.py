import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.data_processing import ModelConfig, CustomerFeatureExtractor, ProxyTargetGenerator, create_inference_pipeline

# Initialize Config
cfg = ModelConfig() 
    
def train_and_log_model(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    model_obj: Any, 
    params: Dict[str, list], 
    name: str
) -> float:
    """Tightly typed training function with MLflow tracking."""
    
    with mlflow.start_run(run_name=f"{name}_Run"):
        # 1. Hyperparameter Tuning
        encoder = create_inference_pipeline()
        X_train_encoded = encoder.fit_transform(X_train, y_train)
        
        grid = GridSearchCV(model_obj, params, scoring="roc_auc", cv=5, n_jobs=-1)
        grid.fit(X_train_encoded, y_train)

        # 2. Build and Fit Final Production Pipeline
        final_pipe = create_inference_pipeline()
        final_pipe.steps.append(('classifier', grid.best_estimator_))
        final_pipe.fit(X_train, y_train)
        
        # 3. Evaluation
        from sklearn.metrics import roc_auc_score
        y_probs = final_pipe.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_probs)

        # 4. Logging
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("auc", auc_score)
        signature = infer_signature(X_train, grid.best_estimator_.predict(X_train_encoded[:1]))
        
        mlflow.sklearn.log_model(
            sk_model=final_pipe, 
            artifact_path="model", 
            registered_model_name=name,
            signature=signature
        )
        
        print(f"✅ {name} optimized. Test AUC: {auc_score:.4f}")
        return auc_score

if __name__ == "__main__":
    mlflow.set_experiment("Bati_Bank_Credit_Risk_V2")
    
    # Load and Extract
    raw_df = pd.read_csv("./data/raw/data.csv")
    extractor = CustomerFeatureExtractor(cfg)
    customer_df = extractor.transform(raw_df)
    
    # Target Generation (Labeling)
    labeller = ProxyTargetGenerator(cfg)
    labeled_df = labeller.generate_labels(customer_df)
    
    X = labeled_df.drop(columns=["is_high_risk"])
    y = labeled_df["is_high_risk"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y
    )

    # Define Models
    models = {
        "Logistic_Scorecard": {
            "model": LogisticRegression(solver="liblinear", class_weight="balanced"),
            "params": {"C": [0.1, 1.0, 10.0]}
        }
    }

    for name, config in models.items():
        train_and_log_model(X_train, X_test, y_train, y_test, config["model"], config["params"], name)