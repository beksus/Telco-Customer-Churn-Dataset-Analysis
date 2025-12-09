# python
# Colab-ready Telco Churn: Logistic Regression benchmark + Keras NN with SMOTE preprocessing

# Install required packages for Colab (uncomment if needed)
# !pip install -q imbalanced-learn seaborn matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# For Colab file upload convenience
try:
    from google.colab import files
    uploaded = files.upload()
    fname = next(iter(uploaded))
except Exception:
    fname = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# 1. LOAD & CLEAN
df = pd.read_csv(fname)
# Drop customerID
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Force convert TotalCharges to numeric, drop NaNs created by coercion
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)

# Encode Churn
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode all other categorical features (drop_first=True)
# Identify categorical columns (object dtype)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Separate features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# 1. Splitting: 80% train / 20% test, stratify to keep class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. HANDLING IMBALANCE: Apply SMOTE to training data only
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 2. Scale features using StandardScaler (fit on training resampled set)
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 3. BENCHMARK: Train Logistic Regression on the same processed data
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res_scaled, y_train_res)

# Predict on test set
y_pred_lr = lr.predict(X_test_scaled)

print("Logistic Regression - Classification Report (Test set):")
print(classification_report(y_test, y_pred_lr, digits=4))

# Extract feature coefficients and plot Top 10 drivers of churn (highest positive coefficients)
feature_names = X.columns.tolist()
coefs = lr.coef_.flatten()
coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
# Select positive coefficients (increase odds of churn) and take top 10 by magnitude
top10 = coef_df.sort_values(by='coef', ascending=False).head(10).iloc[::-1]  # reverse for horizontal bar plot

plt.figure(figsize=(8,6))
sns.barplot(x='coef', y='feature', data=top10, palette='viridis')
plt.title('Top 10 Positive Drivers of Churn (Logistic Regression)')
plt.xlabel('Coefficient (positive -> higher churn odds)')
plt.tight_layout()
plt.show()

# 4. NEURAL NETWORK IMPLEMENTATION
# set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

input_dim = X_train_res_scaled.shape[1]

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train_res_scaled, y_train_res,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2
)

# 5. EVALUATION & VISUALIZATION
# Predict on test set
y_proba_nn = model.predict(X_test_scaled).ravel()
y_pred_nn = (y_proba_nn >= 0.5).astype(int)

print("\nNeural Network - Accuracy (Test):", accuracy_score(y_test, y_pred_nn))
print("Neural Network - Classification Report (Test set):")
print(classification_report(y_test, y_pred_nn, digits=4))

# Confusion Matrix heatmap for NN
cm = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No','Yes'], yticklabels=['No','Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Neural Network')
plt.tight_layout()
plt.show()

# Plot Training vs Validation Loss and Accuracy side-by-side
hist = history.history
epochs = range(1, len(hist['loss'])+1)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, hist['loss'], label='Train Loss')
plt.plot(epochs, hist['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, hist['accuracy'], label='Train Acc')
plt.plot(epochs, hist['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 6. FINAL COMPARISON: Recall (Neural Network vs Logistic Regression)
recall_nn = recall_score(y_test, y_pred_nn)
recall_lr = recall_score(y_test, y_pred_lr)

print(f"Final Recall comparison -> Neural Network Recall: {recall_nn:.4f} | Logistic Regression Recall: {recall_lr:.4f}")

if recall_nn > recall_lr:
    print("Neural Network has higher recall than Logistic Regression on the test set.")
elif recall_nn < recall_lr:
    print("Logistic Regression has higher recall than the Neural Network on the test set.")
else:
    print("Neural Network and Logistic Regression have equal recall on the test set.")
