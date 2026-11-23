
import pandas as pd
import os

# Load dataset
file_path = "data/smartphone_usage.csv"

if not os.path.exists(file_path):
    print("Dataset not found! Check the data folder.")
    raise SystemExit

df = pd.read_csv(file_path)

# 1. Dataset dimensions
print("\n========== DATASET SHAPE ==========")
print("Rows and columns:", df.shape)

# 2. First five records
print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

# 3. Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 4. Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 5. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe(include="all").to_string())

# 6. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 7. Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

# 8. Unique values in each column
print("\n========== UNIQUE VALUES ==========")
for column in df.columns:
    print(column, ":", df[column].nunique())

print("\nDataset inspection completed!")


# 9. Dataset shape and feature names
print("\n========== DATASET DETAILS ==========")
print("Total records:", df.shape[0])
print("Total columns:", df.shape[1])
print("Features:", df.columns.tolist())

# 10. Target class distribution
print("\n========== CLASS DISTRIBUTION ==========")
print(df["addicted"].value_counts(dropna=False))

# 11. Class distribution in percentage
print("\n========== CLASS PERCENTAGE ==========")
print(df["addicted"].value_counts(normalize=True) * 100)

# 12. Target data type
print("\n========== TARGET INFORMATION ==========")
print(df["addicted"].dtype)
print("Unique target values:", df["addicted"].unique())

