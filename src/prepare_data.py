import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

# Convert target variable into numbers
# No = 0, Yes = 1
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True)

# Save processed data
X.to_csv("data/X_processed.csv", index=False)
y.to_csv("data/y_processed.csv", index=False)

# Display results
print("===== DATA PREPARATION COMPLETE =====")
print("Original features:", df.shape[1] - 1)
print("Processed features:", X.shape[1])
print("Number of samples:", X.shape[0])

print("\n===== TARGET DISTRIBUTION =====")
print(y.value_counts())

print("\n===== PROCESSED DATA SHAPE =====")
print(X.shape)

print("\nProcessed data saved successfully.")
print("X: data/X_processed.csv")
print("y: data/y_processed.csv")