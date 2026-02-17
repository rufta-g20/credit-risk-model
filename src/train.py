import pandas as pd
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
from typing import Dict, Any
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from src.data_processing import ModelConfig, CustomerFeatureExtractor, ProxyTargetGenerator, create_inference_pipeline

cfg = ModelConfig() 
    
def train_and_log_model(X_train, X_test, y_train, y_test, model_obj, params, name):
    with mlflow.start_run(run_name=f"{name}_Run"):
        # 1. Hyperparameter Tuning
        encoder = create_inference_pipeline()
        X_train_encoded = encoder.fit_transform(X_train, y_train)
        X_test_encoded = encoder.transform(X_test)
        
        grid = GridSearchCV(model_obj, params, scoring="roc_auc", cv=5, n_jobs=-1)
        grid.fit(X_train_encoded, y_train)
        best_model = grid.best_estimator_

        # 2. Build Final Production Pipeline
        final_pipe = create_inference_pipeline()
        final_pipe.steps.append(('classifier', best_model))
        final_pipe.fit(X_train, y_train)
        
        # 3. SHAP Global Explainability
        # We explain the classifier using the encoded training data
        explainer = shap.LinearExplainer(best_model, X_train_encoded)
        shap_values = explainer.shap_values(X_test_encoded)
        
        # Save Global Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_encoded, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")

        # 4. Evaluation
        from sklearn.metrics import roc_auc_score
        y_probs = final_pipe.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_probs)

        # 5. Logging
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("auc", auc_score)
        
        X_sample = X_train.iloc[:1].copy()
        for col in ["F_Count", "R_Min_Days", "Hour_Mode"]:
            X_sample[col] = X_sample[col].astype(float)
        
        signature = infer_signature(X_sample, final_pipe.predict(X_sample))

        mlflow.sklearn.log_model(
            sk_model=final_pipe, 
            artifact_path="model", 
            registered_model_name=name,
            signature=signature
        )
        
        print(f"✅ {name} optimized with SHAP. Test AUC: {auc_score:.4f}")
        return auc_score

if __name__ == "__main__":
    mlflow.set_experiment("Bati_Bank_Credit_Risk_V2")
    raw_df = pd.read_csv("./data/raw/data.csv")
    customer_df = CustomerFeatureExtractor(cfg).transform(raw_df)
    labeled_df = ProxyTargetGenerator(cfg).generate_labels(customer_df)
    
    X = labeled_df.drop(columns=["is_high_risk"])
    y = labeled_df["is_high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE, stratify=y)

    models = {
        "Logistic_Scorecard": {
            "model": LogisticRegression(solver="liblinear", class_weight="balanced"),
            "params": {"C": [0.1, 1.0, 10.0]}
        }
    }

    for name, config in models.items():
        train_and_log_model(X_train, X_test, y_train, y_test, config["model"], config["params"], name)