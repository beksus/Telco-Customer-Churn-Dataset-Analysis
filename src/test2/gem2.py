# ==========================================
# CSC3034 Assignment 2 - High Distinction Code
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Imbalanced-Learn (for SMOTE)
from imblearn.over_sampling import SMOTE

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set Plot Style
plt.style.use('ggplot')

# ==========================================
# 1. DATA ACQUISITION & CLEANING
# ==========================================
print("Loading Data...")
try:
    # Load dataset
    df = pd.read_csv('../WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Drop irrelevant ID column
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # FIX: TotalCharges has whitespace " " strings. Force convert to numeric.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with missing values (created by the step above)
    df.dropna(inplace=True)

    print(f"Data Cleaned. Final Shape: {df.shape}")

except FileNotFoundError:
    print("ERROR: File not found. Please upload 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")

# ==========================================
# 2. ENCODING & SPLITTING
# ==========================================
# Target Encoding (Yes -> 1, No -> 0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-Hot Encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

# Define Features (X) and Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Save feature names for later plotting
feature_names = X.columns

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. HANDLING IMBALANCE (SMOTE) & SCALING
# ==========================================
# This addresses the "Real-world dataset" issue of class imbalance
print("Applying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale the data (Fit on Resampled Train, Transform on Test)
# Note: We fit the scaler on the resampled data so the model sees a normalized distribution
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

print(f"Original Train Shape: {X_train.shape}")
print(f"Resampled Train Shape: {X_train_resampled.shape}")

# ==========================================
# 4. BENCHMARK MODEL (Logistic Regression)
# ==========================================
# Rubric: "Comparisons to benchmarks"
print("\n--- Training Benchmark Model (Logistic Regression) ---")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train_resampled)

# Benchmark Evaluation
y_pred_log = log_model.predict(X_test_scaled)
print("\n[Benchmark] Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))

# VISUALIZATION: Feature Importance (Why do customers churn?)
# We map coefficients back to feature names
plt.figure(figsize=(10, 6))
importance = pd.Series(log_model.coef_[0], index=feature_names)
importance.nlargest(10).plot(kind='barh', color='#4c72b0')
plt.title('Top 10 Features Driving Churn (Benchmark Analysis)')
plt.xlabel('Coefficient Magnitude')
plt.tight_layout()
plt.show()

# ==========================================
# 5. NEURAL NETWORK (The Main Model)
# ==========================================
print("\n--- Building Neural Network ---")
model = Sequential()

# Input Layer & Hidden Layer 1
model.add(Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]))
model.add(Dropout(0.3)) # Prevent overfitting

# Hidden Layer 2
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))

# Output Layer (Sigmoid for Binary Classification)
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early Stopping (Stop if validation loss doesn't improve for 10 epochs)
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# Train
history = model.fit(X_train_scaled, y_train_resampled,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)

# ==========================================
# 6. EVALUATION & VISUALIZATION
# ==========================================
# Predict
y_pred_nn_probs = model.predict(X_test_scaled)
y_pred_nn = (y_pred_nn_probs > 0.5).astype(int)

# 1. Classification Report
print("\n[Final Model] Neural Network Report:")
print(classification_report(y_test, y_pred_nn))

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_nn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Neural Network Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 3. Training History (Accuracy & Loss)
plt.figure(figsize=(14, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# ==========================================
# 7. FINAL SUMMARY (Recall Comparison)
# ==========================================
# Calculate Recalls for Class 1 (Churn)
from sklearn.metrics import recall_score
recall_log = recall_score(y_test, y_pred_log)
recall_nn = recall_score(y_test, y_pred_nn)

print("\n" + "="*40)
print("FINAL ANALYSIS SUMMARY")
print("="*40)
print(f"Logistic Regression Recall (Churn): {recall_log:.4f}")
print(f"Neural Network Recall (Churn):      {recall_nn:.4f}")
print("-" * 40)
if recall_nn > recall_log:
    print("Conclusion: The Neural Network outperformed the Benchmark in detecting churners.")
else:
    print("Conclusion: The Benchmark performed comparably to the Neural Network.")
print("="*40)