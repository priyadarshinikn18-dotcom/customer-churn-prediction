import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load processed test data
X_test = pd.read_csv("data/X_test.csv")

# Load trained XGBoost model
model = joblib.load("models/xgboost.pkl")

print("Creating SHAP explanations...")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the test dataset
shap_values = explainer.shap_values(X_test)

# Create SHAP summary plot
plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

# Save plot
plt.savefig(
    "models/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP analysis completed.")
print("Saved to: models/shap_summary.png")