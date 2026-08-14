import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load("models/xgboost.pkl")
feature_names = joblib.load("models/feature_names.pkl")

df = pd.read_csv("data/cleaned_customer_churn.csv")

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# ============================================================
# HEADER
# ============================================================

st.title("📊 AI-Powered Customer Churn Intelligence")
st.caption(
    "Machine Learning • XGBoost • Explainable AI • Customer Retention"
)

st.divider()

# ============================================================
# BUSINESS METRICS
# ============================================================

total_customers = len(df)
churned_customers = (df["Churn"] == "Yes").sum()
retained_customers = (df["Churn"] == "No").sum()

churn_rate = (churned_customers / total_customers) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

with col3:
    st.metric(
        "Retained Customers",
        f"{retained_customers:,}"
    )

with col4:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

st.divider()

# ============================================================
# ANALYTICS
# ============================================================

st.header("📈 Customer Analytics")

col1, col2 = st.columns(2)

with col1:

    churn_data = df["Churn"].value_counts()

    st.subheader("Churn Distribution")

    st.bar_chart(churn_data)


with col2:

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.subheader("Churn by Contract")

    st.bar_chart(contract_churn)

st.divider()

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("🤖 Model Performance")

test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

with col5:
    st.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}"
    )

st.info(
    "Recall is particularly important for churn prediction because "
    "missing a customer who is likely to churn can result in customer loss."
)

st.divider()

# ============================================================
# CUSTOMER PREDICTION
# ============================================================

st.header("🔮 Customer Churn Prediction")

st.write(
    "Enter customer information to estimate the probability of churn."
)

# ------------------------------------------------------------
# CUSTOMER DETAILS
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col3:

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

# ------------------------------------------------------------
# BILLING DETAILS
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

with col3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# ------------------------------------------------------------
# CHARGES
# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=10.0
    )

# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Convert categorical variables
input_encoded = pd.get_dummies(
    input_data,
    drop_first=True
)

# Match training feature structure
input_encoded = input_encoded.reindex(
    columns=feature_names,
    fill_value=0
)

st.divider()

# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Predict Churn Risk",
    use_container_width=True
):

    probability = model.predict_proba(
        input_encoded
    )[0][1]

    churn_percentage = probability * 100

    # Risk classification
    if churn_percentage >= 70:
        risk_level = "HIGH"
        risk_message = (
            "High churn risk. Immediate retention action is recommended."
        )

    elif churn_percentage >= 40:
        risk_level = "MEDIUM"
        risk_message = (
            "Medium churn risk. Monitor this customer closely."
        )

    else:
        risk_level = "LOW"
        risk_message = (
            "Low churn risk. Customer appears relatively stable."
        )

    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("🎯 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Churn Probability",
            f"{churn_percentage:.2f}%"
        )

    with col2:

        st.metric(
            "Risk Level",
            risk_level
        )

    with col3:

        st.metric(
            "Risk Score",
            f"{churn_percentage:.0f}/100"
        )

    # Risk message
    if risk_level == "HIGH":

        st.error(
            f"🔴 {risk_message}"
        )

    elif risk_level == "MEDIUM":

        st.warning(
            f"🟡 {risk_message}"
        )

    else:

        st.success(
            f"🟢 {risk_message}"
        )

    # ========================================================
    # RETENTION RECOMMENDATION
    # ========================================================

    st.subheader("💡 Recommended Retention Actions")

    if risk_level == "HIGH":

        recommendations = [
            "Contact the customer proactively",
            "Offer a personalized retention discount",
            "Consider a long-term contract offer",
            "Provide additional technical support",
            "Review billing and service issues"
        ]

    elif risk_level == "MEDIUM":

        recommendations = [
            "Monitor customer activity",
            "Offer a targeted promotional plan",
            "Improve customer support engagement",
            "Consider contract upgrade incentives"
        ]

    else:

        recommendations = [
            "Maintain current service quality",
            "Continue customer engagement",
            "Offer loyalty benefits",
            "Monitor future behavior"
        ]

    for recommendation in recommendations:

        st.write(
            f"• {recommendation}"
        )

    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.subheader("🧠 Explainable AI")

    st.write(
        "The following analysis shows which customer features "
        "influenced the model's prediction."
    )

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            input_encoded
        )

        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1]

        feature_importance = pd.DataFrame({
            "Feature": input_encoded.columns,
            "Impact": shap_array[0]
        })

        feature_importance["Absolute Impact"] = (
            feature_importance["Impact"].abs()
        )

        feature_importance = feature_importance.sort_values(
            "Absolute Impact",
            ascending=False
        ).head(10)

        feature_importance = feature_importance.sort_values(
            "Impact"
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.barh(
            feature_importance["Feature"],
            feature_importance["Impact"]
        )

        ax.set_xlabel(
            "SHAP Impact on Churn Prediction"
        )

        ax.set_title(
            "Top Factors Influencing This Prediction"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    except Exception as e:

        st.warning(
            f"SHAP explanation could not be generated: {e}"
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Customer Churn Intelligence System | "
    "Python • XGBoost • SHAP • Streamlit"
)