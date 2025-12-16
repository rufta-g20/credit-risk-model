# Use a slim Python image for a smaller footprint
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (curl is needed for the Health Check)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (Filtered by your .dockerignore)
COPY . .

# Set PYTHONPATH so 'src' is recognized as a module
# This allows 'import src.api' to work correctly
ENV PYTHONPATH=/app

# Expose the port FastAPI runs on
EXPOSE 8000

# Task 6 Requirement: Health Check
# This ensures the container is marked as "unhealthy" if the API crashes
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
# We use uvicorn to point to the 'app' instance inside src/api/main.py
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]