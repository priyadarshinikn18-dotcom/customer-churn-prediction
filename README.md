@'
# 📊 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts customer churn, evaluates customer risk, explains predictions using SHAP, and provides retention recommendations through a Streamlit dashboard.

## 🚀 Features

- Customer churn prediction
- Churn probability and risk scoring
- Low / Medium / High risk classification
- Logistic Regression, Random Forest and XGBoost models
- SHAP-based Explainable AI
- Customer retention recommendations
- Interactive Streamlit dashboard
- Complete data preprocessing pipeline

## 📈 Model Performance

### XGBoost — Final Model

| Metric | Score |
|---|---:|
| Accuracy | **75.23%** |
| Precision | **52.24%** |
| Recall | **78.07%** |
| F1 Score | **62.59%** |
| ROC-AUC | **84.01%** |

XGBoost was selected because it achieved the strongest performance on the key churn-focused metrics, particularly Recall, F1 Score and ROC-AUC.

## 🤖 Model Comparison

| Model | Accuracy | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | ~80% | ~56% | ~60% | — |
| Random Forest | 77.50% | 65.24% | 60.62% | 82.76% |
| **XGBoost** | **75.23%** | **78.07%** | **62.59%** | **84.01%** |

## 🧠 Explainable AI

SHAP (SHapley Additive exPlanations) is used to understand which features influence the model's churn predictions.

Key factors analyzed include:

- Contract type
- Tenure
- Monthly charges
- Total charges
- Payment method
- Internet service
- Technical support
- Online security

## 🎯 Risk Classification

| Churn Probability | Risk Level |
|---:|---|
| `< 40%` | 🟢 Low |
| `40% – 69.99%` | 🟡 Medium |
| `≥ 70%` | 🔴 High |

The application also generates a customer risk score from **0–100**.

## 💡 Retention Recommendations

The system provides risk-based recommendations such as:

- Proactive customer outreach
- Retention offers
- Contract incentives
- Technical support
- Customer engagement
- Loyalty benefits

## 📊 Dataset

The project uses the IBM Telco Customer Churn dataset.

- **Customers:** 7,043
- **Features:** 19 predictive features
- **Churned:** 1,869
- **Retained:** 5,174
- **Churn Rate:** 26.54%

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Git & GitHub

## 📁 Project Structure

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
├── requirements.txt
├── README.md
└── .gitignore