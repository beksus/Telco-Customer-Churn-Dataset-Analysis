# python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# --- 1. Load data ---
DATA_PATH = "../WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

# --- 2. Basic cleaning ---
# Convert TotalCharges (some values are blank strings) to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Fill NaNs in TotalCharges with median (alternatively could use MonthlyCharges * tenure)
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Drop customerID (identifier)
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# Replace service-specific placeholders
df.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)

# --- 3. Encode categorical variables ---
# Keep copy of original columns for later feature names
orig_cols = df.columns.tolist()

# Map binary columns (Yes/No) to 1/0, map gender to 1/0
object_cols = [c for c in df.columns if df[c].dtype == 'object']
for c in object_cols:
    unique_vals = set(df[c].dropna().unique())
    if unique_vals <= {'Yes', 'No'}:
        df[c] = df[c].map({'Yes': 1, 'No': 0})
    elif unique_vals <= {'Male', 'Female'}:
        df[c] = df[c].map({'Male': 1, 'Female': 0})
    else:
        # one-hot encode remaining categorical variable (drop_first to avoid multicollinearity)
        df = pd.get_dummies(df, columns=[c], drop_first=True)

# --- 4. Split features / target ---
y = df['Churn'].astype(int)
X = df.drop(columns=['Churn'])

# Standardize numeric features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train/test split (80/20) stratified on churn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# --- 5. Handle class imbalance with SMOTE on training set only ---
sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"Before SMOTE: {np.bincount(y_train)}  After SMOTE: {np.bincount(y_train_res)}")

# --- 6. Build the neural network (Keras) ---
input_dim = X_train_res.shape[1]
def build_model(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model

model = build_model(input_dim)
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Train
history = model.fit(
    X_train_res, y_train_res,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=2
)

# --- 7. Training plots ---
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# --- 8. Evaluate on test set ---
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("Neural Network - Test classification report:")
print(classification_report(y_test, y_pred, digits=4))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Compute F1, precision, recall explicitly
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# --- 9. Save the trained model ---
MODEL_PATH = "../telco_churn_model.h5"
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# --- 10. Baseline: Logistic Regression for comparison ---
log = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)
log.fit(X_train, y_train)  # train baseline on original (imbalanced) training set but with class_weight
y_pred_log = log.predict(X_test)

print("\nLogistic Regression - Test classification report:")
print(classification_report(y_test, y_pred_log, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_log))

# --- 11. Permutation importance (on logistic regression baseline) ---
print("\nPermutation importance (Logistic Regression) - top features:")
perm = permutation_importance(log, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=2)
feat_importances = pd.Series(perm.importances_mean, index=X.columns)
top_feats = feat_importances.sort_values(ascending=False).head(15)
print(top_feats)

# Optional: plot top permutation importances
plt.figure(figsize=(8,6))
sns.barplot(x=top_feats.values, y=top_feats.index)
plt.title('Top permutation importances (Logistic Regression)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# --- End ---
