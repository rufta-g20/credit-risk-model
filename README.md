# 📉 Credit Risk Probability Model for Alternative Data

## 🏢 Business Problem

Traditional credit scoring relies on historical loan data, which excludes millions of potential customers. Bati Bank aims to expand its services to the "unbanked" by using alternative eCommerce transaction data to assess creditworthiness accurately and transparently.

## 💡 Solution Overview

I developed a production-ready Credit Scoring Engine that:

* **Engineers Proxy Targets:** Uses RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to define risk labels in the absence of historical defaults.
* **Ensures Transparency:** Utilizes a Logistic Regression Scorecard with Weight of Evidence (WoE) transformation to meet Basel II interpretability standards.
* **Automates MLOps:** Features experiment tracking via MLflow, a hardened FastAPI for inference, and a Streamlit "What-If" dashboard for credit officers.

## 🚀 Recent Engineering Updates (Week 4 Plan)

| Day | Task | Status | Proof of Work |
| --- | --- | --- | --- |
| **Wed** | Code Refactoring | ✅ Done | Implementation of Type Hints & Pydantic Schemas |
| **Thu** | Logging & Error Handling | ✅ Done | Structured JSON logging in `src/api/main.py` |
| **Fri** | SHAP Integration | ✅ Done | Local & Global justifications via `src/predict.py` |
| **Sat** | Officer Dashboard | ✅ Done | Streamlit "What-If" Analysis tool |
| **Sun** | Testing & CI/CD | ✅ Done | Automated Unit tests and GitHub Actions |
| **Mon** | Final Documentation | ✅ Done | Professional README and Demo assets |

---

## 📊 Key Results

* **Operational Efficiency:** Reduced credit review time from hours to seconds via automated API inference.
* **100% Explainability:** Every prediction includes a SHAP justification, providing a clear "Why" behind every credit decision.
* **Robustness:** Achieved 100% test coverage on critical data processing functions (RFM and Temporal engineering).

---

## 📦 Project Structure

```text
credit-risk-model/ 
├── .github/workflows/
│   └── ci.yml             # CI/CD pipeline (Linting, Safety, Tests)
├── artifacts/       
│   └── shap_summary.png   # Global model explanation plot
├── assets/
│   └── demo.mp4           # System Demo Video
├── data/                        
│   └── raw/               # Raw eCommerce transaction data   
├── notebooks/ 
│   └── eda.ipynb          # Exploratory analysis & Clustering logic
├── src/
│   ├── api/
│   │   ├── main.py        # FastAPI app with structured logging
│   │   └── pydantic_models.py # Data validation schemas
│   ├── data_processing.py # RFM & WoE Pipeline
│   ├── train.py           # MLflow training & artifact logging
│   ├── predict.py         # SHAP-enabled inference logic
│   └── dashboard.py       # Streamlit What-If tool
├── tests/ 
│   └── test_data_processing.py 
├── Dockerfile             # Containerized environment
└── README.md 

```

---

## 🛠️ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/rufta-g20/credit-risk-model
cd credit-risk-model
pip install -r requirements.txt

```

### 2. Run the Lifecycle

* **Train & Track:** `python -m src.train` (View results at `mlflow ui`)
* **Launch API:** `uvicorn src.api.main:app --reload`
* **Launch Dashboard:** `streamlit run src/dashboard.py`

---

## 🧠 Technical Details

* **Data Engineering:** Temporal features (Hour, Day) and RFM metrics aggregated from raw xente transactions.
* **Algorithm:** Logistic Regression with  regularization to prevent feature saturation.
* **XAI:** SHAP (SHapley Additive exPlanations) used to calculate the contribution of each feature to the final risk probability.
* **Validation:** 5-Fold Cross-Validation tracked via MLflow.

---

## 🎥 Demo

Check out the full system walkthrough (API + MLflow + Dashboard) below:

[https://github.com/rufta-g20/credit-risk-model/blob/main/assets/demo.mp4](https://www.google.com/search?q=https://github.com/rufta-g20/credit-risk-model/blob/main/assets/demo.mp4)

---

## 🔮 Future Improvements

* **Recursive Feature Elimination:** To reduce the dominance of the `M_Debit_Total` feature found during SHAP analysis.
* **Real-time Retraining:** Integrating a drift detection mechanism to trigger retraining when customer behavior shifts.
* **Deployment:** Migrating from Localhost to a fully orchestrated Kubernetes cluster.

---

## ✍️ Author

**Rufta Gaiem**

* **LinkedIn:** [rufta-gaiem-weldegiorgis-b36426329](https://www.linkedin.com/in/rufta-gaiem-weldegiorgis-b36426329)
* **Email:** ruftagaim@gmail.com