import pandas as pd

# Load raw dataset
df = pd.read_csv("data/customer_churn.csv")

print("Original shape:", df.shape)

# Convert TotalCharges from text to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Replace invalid/blank TotalCharges with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Remove customerID because it is only an identifier
df = df.drop(columns=["customerID"])

# Save cleaned dataset
df.to_csv("data/cleaned_customer_churn.csv", index=False)

print("Cleaned shape:", df.shape)
print("Missing values after cleaning:")
print(df.isnull().sum())

print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

print("\nCleaned dataset saved to:")
print("data/cleaned_customer_churn.csv")