# 📉 Credit Risk Probability Model for Alternative Data

![CI Status](https://github.com/rufta-g20/credit-risk-model/actions/workflows/ci.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

This repository contains the end-to-end implementation for building, deploying, and automating a Credit Risk Model for Bati Bank, leveraging alternative data from an eCommerce platform.

## 1. Credit Scoring Business Understanding

### 1.1 Basel II Influence
The Basel II Capital Accord sets global standards for how banks measure credit risk. This project follows these standards by ensuring:
* **Interpretability:** Using Logistic Regression with Weight of Evidence (WoE).
* **Transparency:** Clear documentation of every feature and cluster logic.

### 1.2 Proxy Target Variable Engineering
Since loan default data was unavailable (new service), we created a **Proxy Target** using:
* **RFM metrics** (Recency, Frequency, Monetary).
* **K-Means Clustering ($k=3$):** Used to identify the "Least Engaged" cluster.
* **Business Logic:** High Recency and Low Frequency represent higher uncertainty and churn risk, which we designate as **High Risk (1)**.


## 2. Project Structure

```text
credit-risk-model/
├── .github/workflows/
│   └── ci.yml             # CI/CD pipeline (Linting, Safety, Tests)
├── data/
│   ├── raw/               # Raw eCommerce transaction data
│   └── processed/         # Dataset after RFM and Target engineering
├── src/
│   ├── data_processing.py # Pipeline: Temporal features, RFM, & Mixed Encoding
│   ├── train.py           # Training, Tuning, and MLflow logging
│   └── api/
│       ├── main.py        # FastAPI app with Health Checks
│       └── pydantic_models.py # Schema with range validation
├── tests/
│   └── test_data_processing.py # Unit tests for pipeline robustness
├── Dockerfile             # Hardened image with HEALTHCHECK
├── docker-compose.yml     # Multi-container orchestration
└── requirements.txt       # Pinned dependencies for reproducibility
```

## 3. 🧠 Advanced Feature Engineering
Following industry best practices and reviewer feedback, the feature pipeline includes:

* **Temporal Features:** Derived `Hour_Mode`, `Day`, `Month`, and `Year` from `TransactionStartTime` to capture time-of-day and seasonal risk patterns.
* **Hybrid Encoding Strategy:**
    * **One-Hot Encoding (OHE):** Applied to low-cardinality features (e.g., `ChannelId`).
    * **Weight of Evidence (WoE):** Applied to high-cardinality and numerical features to ensure a monotonic relationship with risk log-odds.
* **Robustness Checks:** Added validation to ensure the K-Means clustering does not collapse with small datasets.

## 4. 🚀 Usage and Workflow

### A. Training and Tracking
To train the models and log results to **MLflow**:

```bash
python -m src.train
mlflow ui
```

The script logs **5-fold Cross-Validation** scores and saves a **Confusion Matrix** artifact for every run to monitor model performance and class errors.

### B. Deployment with Docker
The API is containerized and includes a **Health Check** to ensure the model is fully loaded into memory before the container begins serving live traffic.

1. Build and Start:

```bash
docker compose up --build
```
2. Access API
Once the container is running, you can interact with the service via the following endpoints:

* **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs) — *Interactive API documentation and testing interface.*
* **Health Status:** [http://localhost:8000/health](http://localhost:8000/health) — *Endpoint to verify the service and model readiness.*

### C. Example API Request
You can test the prediction endpoint using the following `curl` command:

```bash
curl -X 'POST' 'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "R_Min_Days": 5.0, 
    "F_Count": 10.0, 
    "M_Debit_Total": 5500.0,
    "M_Debit_Mean": 550.0, 
    "M_Debit_Std": 150.0,
    "Hour_Mode": 14, 
    "Channel_Mode": "ChannelId_3", 
    "Product_Mode": "airtime"
}'
```

## 5. 🛡️ Quality Assurance (CI/CD)
The project uses **GitHub Actions** to automate quality control and ensure code reliability through the following steps:


* **Linting:** `Black` and `Flake8` are used to enforce PEP 8 style consistency and catch potential logic errors.
* **Security:** A `Safety` scan is performed on all dependencies to identify and mitigate known vulnerabilities.
* **Unit Testing:** `Pytest` is implemented to validate core logic, specifically covering **RFM aggregation** and **temporal feature extraction**.

## 6. 🤝 Git Best Practices
This project adheres to a professional **branch-based workflow** to maintain a clean and stable codebase:

* **Feature Branching:** Major updates (Tasks 3-6) were developed on the dedicated `task-3` branch.
* **Pull Requests (PRs):** All code was integrated into the `main` branch via Pull Requests, ensuring a complete review history and documentation of changes.