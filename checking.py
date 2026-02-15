import mlflow
model = mlflow.pyfunc.load_model(f"models:/Logistic_Scorecard/latest")
print(model.metadata.signature)