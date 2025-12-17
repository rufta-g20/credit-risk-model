# 1. Change to Python 3.9-slim for better compatibility with xverse
FROM python:3.9-slim

WORKDIR /app

# 2. Keep curl for your HEALTHCHECK
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Explicitly install compatible versions BEFORE the rest of requirements
# This ensures xverse doesn't try to pull in a newer, breaking Pandas version
# Combine installs to reduce image layers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "pandas==1.5.3" \
    "numpy==1.23.5" \
    "xverse==1.0.5" \
    "scikit-learn==1.2.2" \
    "fastapi" "uvicorn" "pydantic" "mlflow"

# 4. Install the remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
EXPOSE 8000

# 5. Ensure your FastAPI code actually has a /health route
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]