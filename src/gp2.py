"""
Telco Customer Churn Prediction - Logistic Regression vs MLP (Neural Network)

Requirements implemented:
- Uses WA_Fn-UseC_-Telco-Customer-Churn.csv
- Drops customerID
- Converts TotalCharges to numeric, fills missing with median
- Encodes Churn: "Yes" -> 1, "No" -> 0
- One-hot encodes all categorical features
- 60/20/20 Train/Validation/Test split with stratification
- StandardScaler on features
- Model A: Logistic Regression (baseline)
- Model B: MLPClassifier (Neural Network)
- Prints accuracy, precision, recall, F1, confusion matrix, ROC-AUC
- Plots:
    * Loss curve for MLP
    * Confusion matrix heatmaps for both models
- Side-by-side metric comparison
- Printed discussion of results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)


# ---------------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------------

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("=== Telco Customer Churn Prediction ===\n")
print(f"Loading dataset from: {DATA_PATH}\n")

df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head(), "\n")

print("Dataset info:")
print(df.info(), "\n")

print("Churn distribution:")
print(df["Churn"].value_counts())
print("\nChurn distribution (proportions):")
print(df["Churn"].value_counts(normalize=True), "\n")


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

print("=== Preprocessing ===")

# Drop customerID
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])
    print("Dropped 'customerID' column.")

# Convert TotalCharges to numeric and fill missing with median
print("\nHandling 'TotalCharges' column...")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
missing_tc = df["TotalCharges"].isna().sum()
print(f"Missing values in TotalCharges after conversion: {missing_tc}")

median_tc = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
print(f"Filled missing TotalCharges with median: {median_tc:.2f}")

# Encode target: Yes -> 1, No -> 0
print("\nEncoding target variable 'Churn' (Yes->1, No->0)...")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
y = df["Churn"]

# Features
X = df.drop(columns=["Churn"])

# Identify categorical columns and apply one-hot encoding
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
print(f"\nCategorical feature columns ({len(categorical_cols)}): {categorical_cols}")

print("Applying one-hot encoding to categorical features...")
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"Feature shape before encoding: {X.shape}")
print(f"Feature shape after encoding:  {X_encoded.shape}\n")


# ---------------------------------------------------------------------------
# 3. Train / Validation / Test Split (60 / 20 / 20)
# ---------------------------------------------------------------------------

print("=== Train / Validation / Test Split (60/20/20) ===")

X_train, X_temp, y_train, y_temp = train_test_split(
    X_encoded,
    y,
    test_size=0.4,          # remaining 40% -> temp (will be split into val/test)
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,          # 50% of 40% = 20% of total
    stratify=y_temp,
    random_state=42
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples\n")


# ---------------------------------------------------------------------------
# 4. Feature Scaling
# ---------------------------------------------------------------------------

print("=== Feature Scaling with StandardScaler ===")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("Scaling complete.\n")


# ---------------------------------------------------------------------------
# 5. Helper Function for Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X, y_true, model_name="Model"):
    """
    Evaluate a classifier on given data and return metrics dictionary.
    Also prints classification report, confusion matrix, and ROC-AUC.
    """
    print(f"\n=== Evaluation for {model_name} ===")

    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        # Fallback in case predict_proba is not available
        # (not expected for LogisticRegression or MLPClassifier)
        y_proba = y_pred

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Churn", "Churn"],
        digits=4
    )
    print("Classification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (rows: true, cols: predicted):")
    print(cm)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = np.nan
    print(f"ROC-AUC: {roc_auc:.4f}\n")

    # Plot confusion matrix heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

    # Extract key metrics for churn class (label 1)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    churn_metrics = report_dict["1"]  # label '1' is churn

    metrics = {
        "accuracy": report_dict["accuracy"],
        "precision_churn": churn_metrics["precision"],
        "recall_churn": churn_metrics["recall"],
        "f1_churn": churn_metrics["f1-score"],
        "roc_auc": roc_auc
    }

    return metrics


# ---------------------------------------------------------------------------
# 6. Model A - Logistic Regression (Baseline)
# ---------------------------------------------------------------------------

print("=== Training Model A: Logistic Regression (Baseline) ===")

log_reg = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)
log_reg.fit(X_train_scaled, y_train)

print("Logistic Regression training complete.\n")

# Evaluate on test set
metrics_logreg = evaluate_model(
    log_reg,
    X_test_scaled,
    y_test,
    model_name="Logistic Regression (Test Set)"
)


# ---------------------------------------------------------------------------
# 7. Model B - MLPClassifier (Neural Network)
# ---------------------------------------------------------------------------

print("=== Training Model B: MLPClassifier (Neural Network) ===")

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=200,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

print(f"MLP training complete. Iterations run: {mlp.n_iter_}\n")

# Plot training loss curve
plt.figure()
plt.plot(mlp.loss_curve_)
plt.title("MLP Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

# Evaluate on test set
metrics_mlp = evaluate_model(
    mlp,
    X_test_scaled,
    y_test,
    model_name="MLP Neural Network (Test Set)"
)


# ---------------------------------------------------------------------------
# 8. Side-by-Side Metric Comparison
# ---------------------------------------------------------------------------

print("=== Side-by-Side Metric Comparison (Test Set) ===")

comparison_df = pd.DataFrame.from_dict(
    {
        "Logistic Regression": metrics_logreg,
        "MLP Neural Network": metrics_mlp
    },
    orient="index"
)

print(comparison_df, "\n")


# ---------------------------------------------------------------------------
# 9. Discussion / Analytical Summary
# ---------------------------------------------------------------------------

print("=== Analytical Summary ===")

# Decide which model is better primarily using ROC-AUC, then recall for churn
better_model = "MLP Neural Network" if metrics_mlp["roc_auc"] >= metrics_logreg["roc_auc"] else "Logistic Regression"

print(f"Chosen best model based on ROC-AUC: {better_model}\n")

def fmt(x):
    return f"{x:.4f}"

print("Summary of key metrics for the Churn class (label = 1) on the TEST set:")
print(f"- Logistic Regression: accuracy={fmt(metrics_logreg['accuracy'])}, "
      f"precision={fmt(metrics_logreg['precision_churn'])}, "
      f"recall={fmt(metrics_logreg['recall_churn'])}, "
      f"f1={fmt(metrics_logreg['f1_churn'])}, "
      f"ROC-AUC={fmt(metrics_logreg['roc_auc'])}")

print(f"- MLP Neural Network: accuracy={fmt(metrics_mlp['accuracy'])}, "
      f"precision={fmt(metrics_mlp['precision_churn'])}, "
      f"recall={fmt(metrics_mlp['recall_churn'])}, "
      f"f1={fmt(metrics_mlp['f1_churn'])}, "
      f"ROC-AUC={fmt(metrics_mlp['roc_auc'])}\n")

# Qualitative discussion printed for the report
print("Discussion:")
print("- Both models achieve reasonable accuracy on the test set.")
print("- Logistic Regression serves as a simple linear baseline.")
print("- The MLP Neural Network can model more complex, non-linear relationships "
      "between customer features and churn behaviour.")

if metrics_mlp["recall_churn"] > metrics_logreg["recall_churn"]:
    print("- The Neural Network has higher RECALL for the churn class, meaning it is better "
          "at correctly identifying customers who are likely to churn.")
else:
    print("- Logistic Regression has higher RECALL for the churn class, meaning it is slightly better "
          "at catching churners, although it may sacrifice precision.")

if metrics_mlp["precision_churn"] > metrics_logreg["precision_churn"]:
    print("- The Neural Network also offers higher PRECISION for churn predictions, "
          "so when it predicts churn, it is more likely to be correct.")
else:
    print("- Logistic Regression offers higher PRECISION for churn predictions, "
          "so its churn predictions contain fewer false alarms.")

print("- ROC-AUC provides an overall measure of ranking quality between churners and non-churners. "
      f"In this experiment, {better_model} achieves the higher ROC-AUC, so it is selected "
      "as the preferred model for this assignment.")

print("- The confusion matrices show that non-churn customers (majority class) are classified more "
      "accurately than churn customers, reflecting the original class imbalance in the dataset.")

print("\nFinal Choice for Assignment:")
print(f"- The **{better_model}** is chosen as the main model, with Logistic Regression used as a benchmark.")
print("- These results and figures (loss curve and confusion matrices) can be directly used "
      "in the Result Evaluation section of the report.")
