import pandas as pd
import shap

def get_prediction_justification(pipeline, input_df: pd.DataFrame):
    """
    Generates SHAP feature importance from a native Sklearn Pipeline.
    """
    try:
        # Access steps directly from the pipeline
        classifier = pipeline.named_steps['classifier']
        encoder = pipeline.named_steps['encoder']
        
        # Transform the input
        encoded_input = encoder.transform(input_df)
        
        # LinearExplainer for Logistic Regression
        explainer = shap.LinearExplainer(classifier, encoded_input)
        shap_values = explainer.shap_values(encoded_input)
        
        # Extract top 5 impacts
        feature_names = encoded_input.columns.tolist()
        importances = dict(zip(feature_names, shap_values[0].tolist()))
        
        return dict(sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    except Exception as e:
        return {"error": f"SHAP engine error: {str(e)}"}