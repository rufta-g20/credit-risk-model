
# 📉 Credit Risk Probability Model for Alternative Data

This repository contains the end-to-end implementation for building, deploying, and automating a Credit Risk Model for Bati Bank, leveraging alternative data from an eCommerce platform. The goal is to create a model that assigns a risk probability score to customers applying for the Buy-Now-Pay-Later (BNPL) service.


## 1. Credit Scoring Business Understanding

This section explains the key credit risk concepts behind this project and how Basel II, proxy targets, and model choices shape our approach to building a credit scoring model using alternative e-commerce data.

### 1.1 How Basel II Influences the Need for an Interpretable & Well-Documented Model

The Basel II Capital Accord sets global standards for how banks must measure and manage credit risk. Because banks must report and justify their risk estimates to regulators and calculate capital adequacy (Risk-Weighted Assets), Basel II expects models to be:

* **Transparent and Explainable:** The model must allow analysts, auditors, and regulators to understand why a customer is classified as high or low risk.
* **Traceable & Fully Documented:** Every feature, transformation, and assumption must be clearly documented to comply with validation requirements.
* **Regulatory Acceptance:** Complex “black-box” models require heavy justification and may be rejected during regulatory review.

Therefore, Basel II strongly encourages the use of interpretable models such as Logistic Regression with Weight of Evidence (WoE), which produces stable, monotonic, and explainable risk scores.

### 1.2 Why We Need a Proxy Target (and the Business Risks)

The dataset does not include an actual loan default outcome, as the Buy-Now-Pay-Later service is new. Without a real target, the model cannot learn which customers default or repay.

**Why a proxy is needed**
To train a model, we must create a target variable. We leverage the key innovation of this challenge: transforming behavioral data into a predictive risk signal. We will use:

* **RFM metrics** (Recency, Frequency, Monetary) 
* **K-Means clustering** 

to identify low-engagement customers, who are assumed to be more likely to default. These customers are labeled: **1 = high risk** and **0 = low risk**.

**Business risks of using a proxy**
The potential business risks of making predictions based on this proxy include:

1.  **Proxy may not equal real default (Model Risk):** The model predicts “likelihood of being in a disengaged cluster,” not true credit default. This is the main risk.
2.  **Wrong business decisions (Financial Impact):** A mismatch can lead to **financial loss** (approving a customer who defaults) or **lost profitable customers** (rejecting a customer who would have repaid).
3.  **Bias and unfairness:** Customers with naturally low transaction frequency may be unfairly penalized.
4.  **Regulatory concerns:** Proxy-based decisions must be justified; otherwise, the model can be challenged by risk committees or regulators.

### 1.3 Trade-offs: Simple Interpretable Models vs. Complex High-Performance Models

| Model Type | Simple, Interpretable (Logistic Regression + WoE) | Complex, High-Performance (e.g., Gradient Boosting) |
| :--- | :--- | :--- |
| **Performance** | Sometimes slightly lower predictive accuracy. | Often higher AUC/ROC and accuracy. |
| **Interpretability** | **High.** Easy to explain, produces monotonic scorecards, and is Basel-friendly. | **Low.** Harder to explain and justify in a regulated environment ("black-box"). |
| **Risk/Validation** | **Low.** Stable and robust. Minimal risk of overfitting proxy noise. | **High.** Higher risk of overfitting noisy proxy labels. Requires heavier validation and documentation. |

**Final Recommendation for This Project**
Because we are working with alternative behavioral data, a proxy risk label, and a regulated lending use-case, the most appropriate model is:

➡️ **Logistic Regression with Weight of Evidence (WoE)**

This model gives the best balance of interpretability, stability, regulatory acceptance, and ease of documentation, making it most suitable for proxy-based modeling in a financial context.


## 2. Project Structure

The project follows a standard MLOps structure to ensure engineering discipline and reproducibility:

````

credit-risk-model/
├── .github/workflows/
│   └── ci.yml             \# For CI/CD using GitHub Actions 
├── data/
│   ├── raw/               \# Raw data goes here 
│   └── processed/         \# Processed data for training 
├── notebooks/
│   └── eda.ipynb          \# Exploratory, one-off analysis 
├── src/
│   ├── **init**.py
│   ├── data\_processing.py \# Script for feature engineering 
│   ├── train.py           \# Script for model training 
│   ├── predict.py         \# Script for inference 
│   └── api/
│       ├── main.py        \# FastAPI application 
│       └── pydantic\_models.py \# Pydantic models for API 
├── tests/
│   └── test\_data\_processing.py \# Unit tests 
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

````

## 3. Key Data Fields

The analysis is based on two CSV files (`data.csv` and `Xente_Variable_Definitions.csv`) and primarily uses transaction data to engineer RFM features. Key fields include:

* **Transactionid:** Unique transaction identifier.
* **AccountId, CustomerId:** Unique identifiers for the customer.
* **ProductCategory:** Broader categories of the purchased item.
* **ChannelId:** Customer channel (web, Android, IOS, etc.).
* **Amount:** Value of the transaction (Positive for debits, negative for credits).
* **Value:** Absolute value of the amount.
* **TransactionStartTime:** Transaction timestamp.

## 4. 🚀 Usage and End-to-End Workflow

This section provides the necessary commands to train the models, run tests, and serve the API.

### A. Training and Experiment Tracking

To train the models and log results to the local `mlruns/` folder, run the training script using the module system from the project root:

```bash
# Ensure virtual environment is active (venv)
python -m src.train
# Results will be logged to mlruns/ and the best model will be registered.
```

To view the MLflow UI:

```bash
mlflow ui --host 127.0.0.1
```

### B. Running Tests and Linting

To execute all unit tests in the tests/ directory:

```bash
python -m pytest
```
To check code style using Black (or the linter you chose):

```bash
black --check src/ tests/
```

### C. Deploying and Serving the API (Local Docker)

The API loads the registered DecisionTree_Benchmark model from the Staging stage.

1. Build the Docker image:
```bash
docker-compose build
```

2. Run the API service:
```bash
docker-compose up
# The API will be available at http://localhost:8000/docs
```

3. Example API Request: You can test the /predict endpoint using the documentation at http://localhost:8000/docs.

```bash
# Example cURL request (replace JSON payload with actual data)
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "R_Min_Days": 5.0,
    "F_Count": 10.0,
    "M_Debit_Total": 5500.0,
    "M_Debit_Mean": 550.0,
    "M_Debit_Std": 150.0,
    "Channel_Mode": "APP",
    "Product_Mode": "P4"
}'
```

## 5. 🧠 Modeling Strategy and Design Decisions

This project utilizes a **score-carding approach** based on established credit risk and marketing methodologies, ensuring the final model is highly **interpretable**.

### A. Proxy Target Creation

* **Necessity:** Since true default data was unavailable, a proxy target (*is_high_risk*) was engineered to define customer segments exhibiting poor engagement.
* **Method:** We used **K-Means Clustering**($k=3$) applied to the standardized **RFM metrics** (Recency, Frequency, Monetary Value).
* **Definition of Risk:** The cluster identified by having the **Highest Average Recency** (longest time since last transaction) and **Lowest Average Frequency** was designated as the high-risk group (Class 1), representing customers who are least engaged and most likely to churn or default.

### B. Feature Transformation (Weight of Evidence - WoE)

* **Objective:** To prepare all features for the final **Logistic Regression Scorecard** model, all features were transformed using the **Weight of Evidence (WoE)** technique.
* **Benefits:**
    * **Linearity:** WoE ensures a monotonic relationship between the feature and the log-odds of the proxy target, which is critical for the linear assumptions of Logistic Regression.
    * **Interpretation:** It allows the final model weights to be directly translated into a transparent scorecard where each feature's contribution to risk is easily quantified.