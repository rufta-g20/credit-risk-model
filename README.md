
# 📉 Credit Risk Probability Model for Alternative Data

This repository contains the end-to-end implementation for building, deploying, and automating a Credit Risk Model for Bati Bank, leveraging alternative data from an eCommerce platform. The goal is to create a model that assigns a risk probability score to customers applying for the Buy-Now-Pay-Later (BNPL) service.


## 1. Credit Scoring Business Understanding

This section explains the key credit risk concepts behind this project and how Basel II, proxy targets, and model choices shape our approach to building a credit scoring model using alternative e-commerce data.

### 1.1 How Basel II Influences the Need for an Interpretable & Well-Documented Model

The Basel II Capital Accord sets global standards for how banks must measure and manage credit risk. Because banks must report and justify their risk estimates to regulators and calculate capital adequacy (Risk-Weighted Assets), Basel II expects models to be:

* [cite_start]**Transparent and Explainable:** The model must allow analysts, auditors, and regulators to understand why a customer is classified as high or low risk[cite: 29].
* **Traceable & Fully Documented:** Every feature, transformation, and assumption must be clearly documented to comply with validation requirements.
* **Regulatory Acceptance:** Complex “black-box” models require heavy justification and may be rejected during regulatory review.

Therefore, Basel II strongly encourages the use of interpretable models such as Logistic Regression with Weight of Evidence (WoE), which produces stable, monotonic, and explainable risk scores.

### 1.2 Why We Need a Proxy Target (and the Business Risks)

[cite_start]The dataset does not include an actual loan default outcome [cite: 30][cite_start], as the Buy-Now-Pay-Later service is new[cite: 4]. Without a real target, the model cannot learn which customers default or repay.

**Why a proxy is needed**
[cite_start]To train a model, we must create a target variable[cite: 30]. [cite_start]We leverage the key innovation of this challenge: transforming behavioral data into a predictive risk signal[cite: 12]. We will use:

* [cite_start]**RFM metrics** (Recency, Frequency, Monetary) [cite: 13, 52]
* [cite_start]**K-Means clustering** [cite: 53]

to identify low-engagement customers, who are assumed to be more likely to default. [cite_start]These customers are labeled: **1 = high risk** and **0 = low risk**[cite: 56].

**Business risks of using a proxy**
[cite_start]The potential business risks of making predictions based on this proxy include[cite: 30]:

1.  **Proxy may not equal real default (Model Risk):** The model predicts “likelihood of being in a disengaged cluster,” not true credit default. This is the main risk.
2.  **Wrong business decisions (Financial Impact):** A mismatch can lead to **financial loss** (approving a customer who defaults) or **lost profitable customers** (rejecting a customer who would have repaid).
3.  **Bias and unfairness:** Customers with naturally low transaction frequency may be unfairly penalized.
4.  **Regulatory concerns:** Proxy-based decisions must be justified; otherwise, the model can be challenged by risk committees or regulators.

### 1.3 Trade-offs: Simple Interpretable Models vs. Complex High-Performance Models

| Model Type | Simple, Interpretable (Logistic Regression + WoE) | Complex, High-Performance (e.g., Gradient Boosting) |
| :--- | :--- | :--- |
| **Performance** | Sometimes slightly lower predictive accuracy. | Often higher AUC/ROC and accuracy. |
| **Interpretability** | **High.** Easy to explain, produces monotonic scorecards, and is Basel-friendly. | [cite_start]**Low.** Harder to explain and justify in a regulated environment ("black-box")[cite: 31]. |
| **Risk/Validation** | **Low.** Stable and robust. Minimal risk of overfitting proxy noise. | **High.** Higher risk of overfitting noisy proxy labels. [cite_start]Requires heavier validation and documentation[cite: 31]. |

**Final Recommendation for This Project**
Because we are working with alternative behavioral data, a proxy risk label, and a regulated lending use-case, the most appropriate model is:

➡️ **Logistic Regression with Weight of Evidence (WoE)**

This model gives the best balance of interpretability, stability, regulatory acceptance, and ease of documentation, making it most suitable for proxy-based modeling in a financial context.


## 2. Project Structure

[cite_start]The project follows a standard MLOps structure to ensure engineering discipline and reproducibility[cite: 24]:

````

credit-risk-model/
├── .github/workflows/
[cite_start]│   └── ci.yml             \# For CI/CD using GitHub Actions [cite: 25]
├── data/
[cite_start]│   ├── raw/               \# Raw data goes here [cite: 25]
[cite_start]│   └── processed/         \# Processed data for training [cite: 25]
├── notebooks/
[cite_start]│   └── eda.ipynb          \# Exploratory, one-off analysis [cite: 25]
├── src/
│   ├── **init**.py
[cite_start]│   ├── data\_processing.py \# Script for feature engineering [cite: 25]
[cite_start]│   ├── train.py           \# Script for model training [cite: 25]
[cite_start]│   ├── predict.py         \# Script for inference [cite: 25]
│   └── api/
[cite_start]│       ├── main.py        \# FastAPI application [cite: 25]
[cite_start]│       └── pydantic\_models.py \# Pydantic models for API [cite: 25]
├── tests/
[cite_start]│   └── test\_data\_processing.py \# Unit tests [cite: 25]
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

````

## 3. Key Data Fields

The analysis is based on two CSV files (`data.csv` and `Xente_Variable_Definitions.csv`) and primarily uses transaction data to engineer RFM features. [cite_start]Key fields include[cite: 18, 19, 20, 21, 22]:

* **Transactionid:** Unique transaction identifier.
* **AccountId, CustomerId:** Unique identifiers for the customer.
* **ProductCategory:** Broader categories of the purchased item.
* [cite_start]**ChannelId:** Customer channel (web, Android, IOS, etc.)[cite: 20].
* [cite_start]**Amount:** Value of the transaction (Positive for debits, negative for credits)[cite: 21].
* [cite_start]**Value:** Absolute value of the amount[cite: 22].
* **TransactionStartTime:** Transaction timestamp.