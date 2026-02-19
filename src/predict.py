import pandas as pd
import shap
import numpy as np

def get_prediction_justification(pipeline, input_df: pd.DataFrame):
    try:
        classifier = pipeline.named_steps['classifier']
        encoder = pipeline.named_steps['encoder']
        
        # 1. Transform the input
        encoded_input = encoder.transform(input_df)
        
        # 2. Use Explainer - We use the model coefficients directly for Logistic Regression
        feature_names = encoded_input.columns.tolist()
        coefs = classifier.coef_[0]
        
        raw_values = encoded_input.values[0]
        impacts = raw_values * coefs
        
        importances = dict(zip(feature_names, impacts.tolist()))
        
        # Filter out zero impacts to keep it clean
        significant_impacts = {k: v for k, v in importances.items() if v != 0}
        
        # Sort by absolute impact
        return dict(sorted(significant_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    except Exception as e:
        return {"error": f"SHAP engine error: {str(e)}"}