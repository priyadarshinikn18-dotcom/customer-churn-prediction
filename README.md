# 📊 AI-Powered Customer Churn Intelligence System

<p align="center">

### Predict • Analyze • Explain • Retain

An end-to-end Machine Learning system for customer churn prediction, risk scoring, explainable AI, and retention intelligence.

**Python · XGBoost · Scikit-learn · SHAP · Streamlit**

</p>

---

## 🚀 Overview

Customer churn is a major business challenge for subscription-based companies.

This project develops an **AI-Powered Customer Churn Intelligence System** that analyzes customer demographics, services, contracts, billing information, and customer behavior to predict the probability that a customer will churn.

The system provides:

- 🎯 Churn probability
- 🚦 Low / Medium / High risk classification
- 📊 Customer risk score
- 🧠 Explainable AI using SHAP
- 💡 Retention recommendations
- 📈 Business analytics
- 🤖 Machine learning model comparison
- 🌐 Interactive Streamlit dashboard
---

## 🎯 Problem Statement

Customer retention is an important business objective because acquiring a new customer can be more expensive than retaining an existing one.

The objective of this project is to build a machine-learning system that can:

- Identify customers who are likely to churn
- Estimate the probability of churn
- Classify customers according to their risk level
- Explain why a customer is considered at risk
- Provide actionable retention recommendations

### Business Question

> **Can machine learning identify customers who are likely to churn based on their demographic, service, contract, billing, and behavioral characteristics?**

---

## 💡 Proposed Solution

This project implements a complete **Data Science → Machine Learning → Explainable AI → Deployment** pipeline.

```text
Customer Dataset
       │
       ▼
Data Inspection
       │
       ▼
Data Cleaning
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering
       │
       ▼
Train / Test Split
       │
       ▼
Model Training
       │
       ├──────────────┐
       │              │
       ▼              ▼
Logistic        Random Forest
Regression
       │              │
       └───────┬──────┘
               │
               ▼
            XGBoost
               │
               ▼
       Model Evaluation
               │
               ▼
      Final XGBoost Model
               │
        ┌──────┴──────┐
        ▼             ▼
      SHAP        Prediction
 Explainability
        │             │
        └──────┬──────┘
               ▼
      Streamlit Dashboard
               │
        ┌──────┴──────┐
        ▼             ▼
    Risk Score    Retention
                  Recommendation
                  ---

## 📊 Dataset

The project uses the **IBM Telco Customer Churn dataset**.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total Customers | **7,043** |
| Original Columns | **21** |
| Predictive Features | **19** |
| Churned Customers | **1,869** |
| Retained Customers | **5,174** |
| Churn Rate | **26.54%** |

### Target Distribution

| Churn Status | Customers | Percentage |
|---|---:|---:|
| No Churn | 5,174 | 73.46% |
| Churn | 1,869 | 26.54% |

The dataset contains demographic, service, contract, billing, and customer-behavior information.

---

## 🧹 Data Preprocessing

The preprocessing pipeline performs the following steps:

1. Dataset inspection
2. Data-type validation
3. Missing-value analysis
4. `TotalCharges` conversion from text to numeric
5. Invalid and blank value handling
6. Removal of the `customerID` identifier
7. Categorical feature encoding
8. Target encoding
9. Feature preparation
10. Stratified train/test splitting

### Train/Test Split

```text
80% Training
20% Testing

---

---

# 📈 Model Performance

Three machine-learning classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

## 🤖 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | ~80% | ~65% | ~56% | ~60% | — |
| Random Forest | 77.50% | 56.61% | 65.24% | 60.62% | 82.76% |
| **XGBoost** | **75.23%** | **52.24%** | **78.07%** | **62.59%** | **84.01%** |

---

## 🏆 Final Model — XGBoost

The final model selected for the application is **XGBoost Classifier**.

### Final Performance

| Metric | Score |
|---|---:|
| Accuracy | **75.23%** |
| Precision | **52.24%** |
| Recall | **78.07%** |
| F1 Score | **62.59%** |
| ROC-AUC | **84.01%** |

### Why XGBoost?

Accuracy was not treated as the only selection criterion because the dataset contains a significant class imbalance.

The project places additional importance on identifying customers who actually churn.

XGBoost achieved:

- 🥇 **78.07% Recall**
- 🥇 **62.59% F1 Score**
- 🥇 **84.01% ROC-AUC**

This makes XGBoost the most suitable final model for the current churn-prediction objective.

> **Important:** XGBoost does not have the highest accuracy in this comparison. Logistic Regression achieved approximately 80% accuracy. XGBoost was selected because it provided stronger churn-focused performance across recall, F1-score, and ROC-AUC.

---

## 🎯 Confusion Matrix — XGBoost

The XGBoost model produced the following confusion matrix on the test set:

```text
                 Predicted
               No Churn   Churn

Actual No         768      267
Actual Churn       82      292

---

# 🧠 Explainable AI

Machine-learning predictions can be difficult to interpret.

To improve transparency, this project uses **SHAP (SHapley Additive exPlanations)** to explain the factors influencing model predictions.

## 🔍 SHAP Workflow

```text
Customer Information
        │
        ▼
   XGBoost Model
        │
        ▼
Churn Probability
        │
        ▼
   SHAP Analysis
        │
        ▼
Feature Contributions
        │
        ▼
Prediction Explanation

---

# 🖥️ Streamlit Application

The trained XGBoost model is integrated into an interactive **Streamlit dashboard**.

The application provides a simple interface for both business analytics and individual customer prediction.

## 📊 Dashboard Features

### Business Analytics

The dashboard provides:

- Total customer count
- Churned customer count
- Retained customer count
- Overall churn rate
- Churn distribution
- Churn analysis by contract

### 🤖 Model Performance

The application displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### 🔮 Customer Prediction

Users can enter customer information and receive:

- Churn probability
- Risk score
- Risk classification
- Retention recommendations

### 🧠 Explainable AI

The application provides SHAP-based explanations showing the most influential factors behind an individual prediction.

---

# 🔄 Prediction Workflow

```text
Customer Input
      │
      ▼
Feature Encoding
      │
      ▼
Feature Alignment
      │
      ▼
XGBoost Model
      │
      ▼
Churn Probability
      │
      ▼
Risk Score
      │
      ▼
Risk Classification
      │
      ▼
SHAP Explanation
      │
      ▼
Retention Recommendation

---

# 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Explainable AI | SHAP |
| Web Application | Streamlit |
| Model Serialization | Joblib |
| Version Control | Git |
| Repository | GitHub |

---

# 📁 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── customer_churn.csv
│   └── cleaned_customer_churn.csv
│
├── models/
│   ├── feature_names.pkl
│   ├── shap_summary.png
│   └── xgboost.pkl
│
├── src/
│   ├── clean_data.py
│   ├── eda.py
│   ├── explain_model.py
│   ├── inspect_data.py
│   ├── prepare_data.py
│   └── train_model.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/priyadarshinikn18-dotcom/customer-churn-prediction.git

---

# 📌 Business Value

This system can help customer-success teams:

- Identify customers with elevated churn risk
- Prioritize retention efforts
- Understand factors influencing predictions
- Segment customers based on risk
- Support proactive customer engagement
- Improve data-driven retention decisions

The application is designed as a **decision-support system**, not an autonomous business decision-maker.

---

# ⚠️ Limitations

This project is intended for **educational, academic, and portfolio purposes**.

Current limitations include:

- Historical customer dataset
- Limited customer attributes
- Fixed risk thresholds
- Rule-based retention recommendations
- No real-time production data
- No automated model retraining
- No model-drift monitoring
- No production authentication
- No live customer database integration

Before production deployment, the model should be validated using current business data and appropriate monitoring.

---

# 🔮 Future Improvements

Potential improvements include:

### ☁️ Deployment

- Cloud deployment
- Docker containerization
- REST API
- Production hosting

### 🔄 MLOps

- MLflow experiment tracking
- Automated model retraining
- Model registry
- Model monitoring
- Data-drift detection
- Model-drift monitoring

### 📊 Advanced Analytics

- Customer segmentation
- Customer lifetime value prediction
- Retention campaign optimization
- Customer behavior forecasting

### 🧪 Machine Learning

- Hyperparameter optimization
- Cross-validation
- Ensemble learning
- Threshold optimization
- Probability calibration

### 🔐 Production

- Authentication
- Authorization
- Secure API
- Database integration
- Logging and monitoring

---

# 🎓 Skills Demonstrated

## Data Science

- Data cleaning
- Exploratory Data Analysis
- Feature engineering
- Data visualization
- Statistical analysis

## Machine Learning

- Binary classification
- Imbalanced classification
- Model comparison
- Logistic Regression
- Random Forest
- XGBoost
- Probability prediction
- Model evaluation

## Explainable AI

- SHAP
- Feature contribution analysis
- Model interpretability

## Application Development

- Streamlit
- Interactive dashboards
- Customer risk scoring
- Business recommendations

## Software Engineering

- Modular Python development
- Virtual environments
- Dependency management
- Git
- GitHub
- Reproducible machine-learning workflows

---

# 🏆 Project Highlights

- **7,043** customer records analyzed
- **3** machine-learning models evaluated
- **84.01% ROC-AUC** achieved with XGBoost
- **78.07% recall** for churn detection
- SHAP-based explainable AI
- Interactive Streamlit prediction dashboard
- Automated customer risk scoring
- Risk-based retention recommendations
- Modular and reproducible ML pipeline

---

# 👩‍💻 Author

**Priyadarshini KN**

AI/ML & Data Science Student

> Building practical machine-learning solutions with a focus on data-driven decision making and explainable AI.

---

# 📄 License

This project is intended for educational, academic, and portfolio purposes.

---