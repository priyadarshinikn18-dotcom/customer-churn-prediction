import pandas as pd

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== CHURN DISTRIBUTION =====")
print(df["Churn"].value_counts())

print("\n===== CHURN PERCENTAGE =====")
print(df["Churn"].value_counts(normalize=True) * 100)