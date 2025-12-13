# Use a slim Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the MLflow runs (containing the model artifacts)
COPY src/ /app/src/

# IMPORTANT: Copy the local MLflow artifacts needed to load the model
# NOTE: This is for local testing only. Production systems would use a remote MLflow store.
COPY mlruns/ /app/mlruns/

# Expose the port for the API
EXPOSE 8000

# Command to run the application using uvicorn
# The command starts uvicorn, serving the 'app' object inside 'src.api.main' module
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]