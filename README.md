#  AI-Powered Customer Churn Intelligence System

> **An end-to-end machine learning solution for predicting customer churn, analyzing customer risk, and generating explainable retention insights.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io/)
[![Scikit--learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey)](#license)

---

##  Overview

Customer churn is one of the most important business problems for subscription-based companies.

This project develops an **AI-powered customer churn intelligence system** that analyzes customer information and predicts the probability that a customer will leave a service.

Instead of producing only a binary prediction, the system provides:

* 🎯 Churn probability
* 📊 Customer risk score
* 🚦 Low / Medium / High risk classification
* 🧠 Explainable AI using SHAP
* 💡 Retention recommendations
* 📈 Interactive business analytics
* 🤖 Machine-learning model performance metrics

The application is built as a complete **Data Science → Machine Learning → Explainable AI → Deployment** pipeline.

---

#  Problem Statement

Customer acquisition is often more expensive than customer retention.

Businesses need to identify customers who are likely to churn **before they leave**, allowing customer-success teams to take preventive action.

### The problem

> **Can machine learning identify customers who are at high risk of churn based on their demographic, service, contract, billing, and usage characteristics?**

### Proposed solution

Build a machine-learning system that:

```text
Customer Data
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Churn Probability
      ↓
Risk Classification
      ↓
Explainable AI
      ↓
Retention Recommendation
```

---

#  Key Features

### 🔮 Churn Prediction

Predicts the probability that an individual customer will churn.

Example:

```text
Churn Probability: 34.83%
Risk Score:        35 / 100
Risk Level:        LOW
```

### 🚦 Risk Classification

Customers are classified into three risk categories:

| Probability | Risk      |
| ----------: | --------- |
|     `< 40%` | 🟢 Low    |
| `40–69.99%` | 🟡 Medium |
|     `≥ 70%` | 🔴 High   |

### 🧠 Explainable AI

SHAP is used to identify which features influenced an individual prediction.

This makes the model more transparent than a simple black-box prediction.

### 💡 Retention Recommendations

The application generates risk-based recommendations such as:

* Customer outreach
* Retention offers
* Contract incentives
* Technical support
* Loyalty benefits

### 📊 Analytics Dashboard

The Streamlit application provides:

* Total customers
* Churned customers
* Retained customers
* Overall churn rate
* Churn distribution
* Churn by contract
* Model performance

---

#  System Architecture

```text
                    ┌─────────────────────┐
                    │   Customer Dataset  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │   Data Cleaning     │
                    │     Pandas          │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │       EDA           │
                    │ Matplotlib/Seaborn  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │   One-Hot Encoding  │
                    └──────────┬──────────┘
                               ↓
                 ┌─────────────┴─────────────┐
                 ↓                           ↓
        Logistic Regression           Random Forest
                 │                           │
                 └─────────────┬─────────────┘
                               ↓
                           XGBoost
                               ↓
                     Model Evaluation
                               ↓
                    ┌──────────┴──────────┐
                    ↓                     ↓
              SHAP Explainability   Risk Prediction
                    │                     │
                    └──────────┬──────────┘
                               ↓
                    Streamlit Dashboard
                               ↓
                    Retention Insights
```

---

#  Dataset

The project uses the **IBM Telco Customer Churn dataset**.

### Dataset statistics

| Property            |     Value |
| ------------------- | --------: |
| Customers           | **7,043** |
| Original columns    |    **21** |
| Predictive features |    **19** |
| Target              | **Churn** |
| Churned customers   | **1,869** |
| Retained customers  | **5,174** |

### Target distribution

| Class    | Customers | Percentage |
| -------- | --------: | ---------: |
| No Churn |     5,174 |     73.46% |
| Churn    |     1,869 |     26.54% |

The target is therefore imbalanced, which is why the project evaluates models using more than accuracy alone.

---

#  Data Preprocessing

The preprocessing pipeline performs:

1. Dataset inspection
2. Data-type analysis
3. Missing-value analysis
4. `TotalCharges` conversion from text to numeric
5. Invalid/blank charge handling
6. Removal of `customerID`
7. Categorical feature encoding
8. Target encoding
9. Stratified train/test split

### Special data-cleaning case

`TotalCharges` was initially stored as a string.

Blank values were detected and converted to numeric values, with the corresponding new-customer values handled as `0`.

The cleaned dataset contains:

```text
7,043 customers
0 missing values
```

---

#  Exploratory Data Analysis

The project analyzes customer behavior through:

* Churn distribution
* Customer tenure
* Monthly charges
* Total charges
* Contract type
* Internet service
* Payment method
* Technical support
* Billing behavior
* Customer service features

The analysis helps identify patterns associated with customer churn before model training.

---

#  Machine Learning

Three classification models were evaluated.

### 1. Logistic Regression

Used as the baseline model because it provides a simple and interpretable benchmark.

### 2. Random Forest

Used to model nonlinear relationships and improve detection of churn cases.

### 3. XGBoost

Used as the final gradient-boosting model because of its strong performance on structured/tabular data.

---

#  Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

### Results

| Model               |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |       ~80% |       ~65% |       ~56% |       ~60% |          — |
| Random Forest       |     77.50% |     56.61% |     65.24% |     60.62% |     82.76% |
| **XGBoost**         | **75.23%** | **52.24%** | **78.07%** | **62.59%** | **84.01%** |

##  Selected Model: XGBoost

The final model achieved:

```text
Accuracy   : 75.23%
Precision  : 52.24%
Recall     : 78.07%
F1 Score   : 62.59%
ROC-AUC    : 84.01%
```

### Why XGBoost?

The objective is not simply to maximize accuracy.

In churn prediction, failing to identify a customer who is actually going to churn can be costly.

Therefore, the project gives significant importance to:

* Recall
* F1-score
* ROC-AUC

XGBoost achieved the strongest overall performance on these metrics.

---

#  Explainable AI — SHAP

A machine-learning model should not only provide a prediction; users should also have an understanding of **why** the prediction was made.

This project uses **SHAP (SHapley Additive exPlanations)** to analyze feature contributions.

The application provides customer-level explanations such as:

```text
Prediction
    ↓
SHAP Analysis
    ↓
Top Influencing Features
    ↓
Positive / Negative Impact
```

Example features analyzed include:

* Contract type
* Tenure
* Monthly charges
* Total charges
* Payment method
* Internet service
* Billing preferences
* Technical support
* Streaming services

The model determines their influence rather than relying on manually assigned importance.

---

#  Application

The project includes an interactive **Streamlit dashboard**.

## Dashboard

The dashboard provides:

```text
┌──────────────────────────────────────────┐
│      CUSTOMER CHURN INTELLIGENCE         │
├──────────┬──────────┬──────────┬─────────┤
│ Customers│ Churned  │ Retained │ Churn % │
├──────────┴──────────┴──────────┴─────────┤
│                                          │
│       Customer Analytics                 │
│                                          │
├──────────────────────────────────────────┤
│       Model Performance                  │
│                                          │
├──────────────────────────────────────────┤
│       Customer Prediction                │
│                                          │
├──────────────────────────────────────────┤
│       SHAP Explanation                   │
│                                          │
└──────────────────────────────────────────┘
```

---

#  Prediction Workflow

A user enters customer information such as:

```text
Tenure
Monthly Charges
Total Charges
Contract
Internet Service
Payment Method
Paperless Billing
Technical Support
```

The application then performs:

```text
User Input
    ↓
Feature Encoding
    ↓
Feature Alignment
    ↓
XGBoost
    ↓
Probability
    ↓
Risk Score
    ↓
Risk Classification
    ↓
SHAP Explanation
    ↓
Retention Recommendation
```

---

#  Retention Strategy

The application translates predictions into actionable suggestions.

### 🔴 High Risk

Potential actions:

* Proactive customer contact
* Personalized retention offer
* Long-term contract incentives
* Technical support
* Billing/service review

### 🟡 Medium Risk

Potential actions:

* Monitor customer activity
* Targeted promotional offers
* Improve engagement
* Contract upgrade incentives

### 🟢 Low Risk

Potential actions:

* Maintain service quality
* Continue engagement
* Loyalty benefits
* Monitor future behavior

> These recommendations are decision-support suggestions and should be validated against real business requirements before production use.

---

#  Technology Stack

| Category            | Technology          |
| ------------------- | ------------------- |
| Language            | Python 3.12         |
| Data Processing     | Pandas, NumPy       |
| Visualization       | Matplotlib, Seaborn |
| Machine Learning    | Scikit-learn        |
| Gradient Boosting   | XGBoost             |
| Explainable AI      | SHAP                |
| Web Application     | Streamlit           |
| Model Serialization | Joblib              |
| Version Control     | Git / GitHub        |

---

#  Repository Structure

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
```

---

#  Installation & Setup

## 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd customer-churn-prediction
```

## 2. Create the virtual environment

Windows:

```powershell
py -3.12 -m venv venv
```

## 3. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```powershell
pip install -r requirements.txt
```

## 5. Run the application

```powershell
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

#  Reproduce the ML Pipeline

The project is organized into independent stages.

### Inspect data

```powershell
python src\inspect_data.py
```

### Clean data

```powershell
python src\clean_data.py
```

### Perform EDA

```powershell
python src\eda.py
```

### Prepare features

```powershell
python src\prepare_data.py
```

### Train models

```powershell
python src\train_model.py
```

### Generate SHAP analysis

```powershell
python src\explain_model.py
```

---

#  Screenshots

Add screenshots of the application here.

Recommended screenshots:

### Dashboard

```text
docs/dashboard.png
```

### Prediction

```text
docs/prediction.png
```

### SHAP Explanation

```text
docs/shap-explanation.png
```

Example Markdown:

```markdown
![Dashboard](docs/dashboard.png)

![Prediction](docs/prediction.png)

![SHAP Explanation](docs/shap-explanation.png)
```

---

#  Example Prediction

Example input:

```text
Tenure:              2 months
Monthly Charges:     95
Total Charges:       190
Contract:            Month-to-month
Internet Service:    Fiber optic
Payment Method:      Electronic check
Paperless Billing:   Yes
Tech Support:        No
```

Example output from the application:

```text
Churn Probability: 34.83%
Risk Score:        35/100
Risk Level:        LOW
```

The output is generated by the trained model and changes according to the customer input.

---

#  Limitations

This is a portfolio and educational project rather than a production banking or telecom decision system.

Current limitations include:

* Historical dataset
* Limited customer features
* Fixed model thresholds
* Rule-based retention recommendations
* No real-time production data
* No automated model retraining
* No model-drift monitoring
* No production-grade authentication

The model should be validated on current business data before being used for real customer decisions.

---

#  Future Improvements

Potential future development includes:

* ☁️ Cloud deployment
* 🔄 Automated model retraining
* 📦 MLflow experiment tracking
* 🐳 Docker deployment
* ⚙️ CI/CD with GitHub Actions
* 📡 Real-time churn prediction
* 👥 Customer segmentation
* 📈 Model drift monitoring
* ⚖️ Fairness and bias analysis
* 🧪 Hyperparameter optimization
* 📊 A/B testing of retention strategies
* 🔐 Production authentication and authorization

---

# 🎓 Skills Demonstrated

This project demonstrates practical experience in:

### Data Science

* Data cleaning
* Exploratory Data Analysis
* Statistical analysis
* Feature engineering
* Data visualization

### Machine Learning

* Classification
* Model comparison
* Imbalanced-data evaluation
* XGBoost
* Model evaluation

### Explainable AI

* SHAP
* Feature contribution analysis
* Model interpretability

### Application Development

* Streamlit
* Interactive prediction interface
* Risk scoring
* Business recommendations

### Software Engineering

* Modular Python scripts
* Virtual environments
* Dependency management
* Git/GitHub
* Reproducible workflows

---

#  Author

Priyadarshini KN

AI/ML & Data Science Student

> Building practical machine-learning solutions with a focus on data-driven decision making and explainable AI.

---

#  License

This project is intended for educational, academic, and portfolio purposes.
