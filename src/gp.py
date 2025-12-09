"""
CSC3034 Assignment 2 - Telco Customer Churn Prediction with MLP

This script:
1. Loads and explores the Telco Customer Churn dataset.
2. Cleans and preprocesses the data (handle missing values, encode categoricals, scale features).
3. Splits the data into train / validation / test sets.
4. Trains a Multi-Layer Perceptron (MLP) neural network.
5. Evaluates the model using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
"""

# ======================
# 1. Imports
# ======================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6, 5)

RANDOM_STATE = 42  # for reproducibility

# ======================
# 2. Load dataset
# ======================

# Change this path if needed
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("\nFirst 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nChurn distribution (counts):")
print(df["Churn"].value_counts())
print("\nChurn distribution (proportions):")
print(df["Churn"].value_counts(normalize=True))


# ======================
# 3. Data cleaning & preprocessing
# ======================

# 3.1 Handle TotalCharges - convert to numeric, coerce errors, fill NaNs
print("\nHandling TotalCharges column...")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
missing_tc = df["TotalCharges"].isna().sum()
print(f"Number of missing TotalCharges after conversion: {missing_tc}")

# Use median to fill missing TotalCharges
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# 3.2 Drop non-informative ID column
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# 3.3 Separate features (X) and target (y)
print("\nSeparating features and target...")
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})  # encode target as 0/1

print("Feature columns:", list(X.columns))
print("Target classes:", y.unique())

# 3.4 One-hot encode categorical features
print("\nEncoding categorical variables with one-hot encoding...")
X_encoded = pd.get_dummies(X, drop_first=True)  # drop_first to avoid dummy trap
print("Shape before encoding:", X.shape)
print("Shape after encoding:", X_encoded.shape)

# ======================
# 4. Train / Validation / Test split
# ======================

print("\nSplitting data into Train / Validation / Test sets...")

# First split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X_encoded,
    y,
    test_size=0.4,
    random_state=RANDOM_STATE,
    stratify=y
)

# Second split: split temp into 50% val, 50% test (so 20/20 overall)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("Train shape:", X_train.shape, "Target:", y_train.shape)
print("Val   shape:", X_val.shape, "Target:", y_val.shape)
print("Test  shape:", X_test.shape, "Target:", y_test.shape)

# ======================
# 5. Feature scaling
# ======================

print("\nScaling features with StandardScaler...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ======================
# 6. Build and train the MLP model
# ======================

print("\nBuilding and training MLPClassifier...")

# Example architecture: 2 hidden layers with 64 and 32 neurons
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=200,
    random_state=RANDOM_STATE
)

mlp.fit(X_train_scaled, y_train)

print("Training complete.")
print("Number of iterations run:", mlp.n_iter_)

# Optional: plot loss curve
plt.figure()
plt.plot(mlp.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

# ======================
# 7. Evaluation on Validation Set
# ======================

print("\n=== Validation Set Performance ===")
val_pred = mlp.predict(X_val_scaled)
val_proba = mlp.predict_proba(X_val_scaled)[:, 1]

print("Classification report (Validation):")
print(classification_report(y_val, val_pred, digits=4))

val_cm = confusion_matrix(y_val, val_pred)
print("Confusion matrix (Validation):")
print(val_cm)

val_auc = roc_auc_score(y_val, val_proba)
print("ROC-AUC (Validation):", val_auc)

# Plot confusion matrix for validation set
disp_val = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=["No Churn", "Churn"])
disp_val.plot(cmap="Blues")
plt.title("Confusion Matrix - Validation Set")
plt.tight_layout()
plt.show()

# ======================
# 8. Final Evaluation on Test Set
# ======================

print("\n=== Test Set Performance ===")
y_pred = mlp.predict(X_test_scaled)
y_proba = mlp.predict_proba(X_test_scaled)[:, 1]

print("Classification report (Test):")
print(classification_report(y_test, y_pred, digits=4))

test_cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (Test):")
print(test_cm)

test_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC (Test):", test_auc)

# Plot confusion matrix for test set
disp_test = ConfusionMatrixDisplay(confusion_matrix=test_cm, display_labels=["No Churn", "Churn"])
disp_test.plot(cmap="Blues")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.show()

print("\nDone.")
