
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/smartphone_usage.csv")

# Remove unnecessary index column
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Select important features
features = [
    "daily_screen_time",
    "night_usage",
    "app_sessions",
    "social_media_usage"
]

target = "addicted"

# Keep only required columns
df = df[features + [target]].copy()

# Convert features to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert target labels into binary values
df[target] = (
    df[target].astype(str).str.strip().str.lower()
    .map({"not addicted": 0, "addicted": 1})
)

# Handle missing values
df = df.dropna()

# Remove duplicate records
df = df.drop_duplicates()

# Separate features and target
X = df[features]
y = df[target].astype(int)

print("\n========== PREPARED DATASET ==========")
print("Total records:", len(df))
print("Features:", features)

print("\nClass distribution:")
print(y.value_counts())

# Plot class distribution
os.makedirs("outputs", exist_ok=True)

y.value_counts().sort_index().rename(
    index={0: "Low Risk", 1: "High Risk"}
).plot(kind="bar", title="Smartphone Usage Risk Distribution")

plt.xlabel("Risk Category")
plt.ylabel("Number of Users")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")
plt.show()

# Stratified train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Fit scaler ONLY on training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
os.makedirs("data/processed", exist_ok=True)

pd.DataFrame(
    X_train_scaled, columns=features
).to_csv("data/processed/X_train.csv", index=False)

pd.DataFrame(
    X_test_scaled, columns=features
).to_csv("data/processed/X_test.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True) * 100)

print("\nPreprocessing completed successfully!")

import joblib

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")