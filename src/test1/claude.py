"""
Telco Customer Churn Prediction using Neural Networks
======================================================
This script implements a deep learning model to predict customer churn
using the Telco Customer Churn dataset.

Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv
Task: Binary Classification (Churn: Yes/No)
"""

# ============================================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("TELCO CUSTOMER CHURN PREDICTION - NEURAL NETWORK MODEL")
print("=" * 80)
print()

# ============================================================================
# 2. DATA LOADING
# ============================================================================

print("STEP 1: Loading Data")
print("-" * 80)

# Load the dataset
df = pd.read_csv('../WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"Dataset shape: {df.shape}")
print(f"Total samples: {df.shape[0]}")
print(f"Total features: {df.shape[1] - 1} (excluding target)")
print()

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())
print()

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\nSTEP 2: Exploratory Data Analysis")
print("-" * 80)

# Check data types
print("Data types:")
print(df.dtypes)
print()

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print()

# Check target distribution
print("Target variable distribution:")
print(df['Churn'].value_counts())
print(f"\nChurn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")
print()

# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================

print("\nSTEP 3: Data Preprocessing")
print("-" * 80)

# 4.1: Drop customerID column (not useful for prediction)
print("Dropping 'customerID' column...")
df = df.drop('customerID', axis=1)

# 4.2: Handle TotalCharges - convert to numeric and handle missing values
print("Handling 'TotalCharges' column...")
# TotalCharges might have spaces instead of NaN for missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values with median
missing_count = df['TotalCharges'].isnull().sum()
if missing_count > 0:
    print(f"  - Found {missing_count} missing values in TotalCharges")
    median_value = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_value, inplace=True)
    print(f"  - Filled with median: {median_value:.2f}")
print()

# 4.3: Separate numerical and categorical features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in df.columns if col not in numerical_features + ['Churn']]

print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features[:5]}... (showing first 5)")
print()

# 4.4: Encode target variable (Churn)
print("Encoding target variable 'Churn'...")
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("  - 'Yes' → 1, 'No' → 0")
print()

# 4.5: One-Hot Encoding for categorical variables
print("Applying One-Hot Encoding to categorical features...")
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
print(f"  - Original features: {len(df.columns)}")
print(f"  - After encoding: {len(df_encoded.columns)}")
print()

# 4.6: Separate features (X) and target (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"Final feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print()

# ============================================================================
# 5. TRAIN-TEST SPLIT
# ============================================================================

print("\nSTEP 4: Train-Test Split")
print("-" * 80)

# Split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0] / len(X):.1%})")
print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0] / len(X):.1%})")
print(f"Training set churn rate: {y_train.mean():.2%}")
print(f"Test set churn rate: {y_test.mean():.2%}")
print()

# ============================================================================
# 6. FEATURE SCALING
# ============================================================================

print("\nSTEP 5: Feature Scaling (Standardization)")
print("-" * 80)

# Standardize features to have mean=0 and variance=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized using StandardScaler")
print(f"  - Mean of training features: ~{X_train_scaled.mean():.4f}")
print(f"  - Std of training features: ~{X_train_scaled.std():.4f}")
print()

# ============================================================================
# 7. HANDLE CLASS IMBALANCE USING SMOTE
# ============================================================================

print("\nSTEP 6: Handling Class Imbalance with SMOTE")
print("-" * 80)

print(f"Before SMOTE:")
print(f"  - Class 0 (No Churn): {(y_train == 0).sum()} samples")
print(f"  - Class 1 (Churn): {(y_train == 1).sum()} samples")
print(f"  - Imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAfter SMOTE:")
print(f"  - Class 0 (No Churn): {(y_train_resampled == 0).sum()} samples")
print(f"  - Class 1 (Churn): {(y_train_resampled == 1).sum()} samples")
print(f"  - Balanced ratio: {(y_train_resampled == 0).sum() / (y_train_resampled == 1).sum():.2f}:1")
print()

# ============================================================================
# 8. BUILD NEURAL NETWORK MODEL
# ============================================================================

print("\nSTEP 7: Building Neural Network Model")
print("-" * 80)

# Get input dimension
input_dim = X_train_resampled.shape[1]

# Build Sequential model
model = Sequential([
    # Input layer (implicit) + First hidden layer
    Dense(64, activation='relu', input_dim=input_dim, name='hidden_layer_1'),
    Dropout(0.3, name='dropout_1'),

    # Second hidden layer
    Dense(32, activation='relu', name='hidden_layer_2'),
    Dropout(0.3, name='dropout_2'),

    # Output layer
    Dense(1, activation='sigmoid', name='output_layer')
], name='ChurnPredictionModel')

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

# Display model architecture
print("Model Architecture:")
model.summary()
print()

# ============================================================================
# 9. TRAIN THE MODEL
# ============================================================================

print("\nSTEP 8: Training the Model")
print("-" * 80)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    '../best_churn_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# Train the model
print("Training in progress...")
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

print("\nTraining completed!")
print(f"Total epochs trained: {len(history.history['loss'])}")
print()

# ============================================================================
# 10. EVALUATE THE MODEL
# ============================================================================

print("\nSTEP 9: Model Evaluation")
print("-" * 80)

# Evaluate on test set
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    X_test_scaled, y_test, verbose=0
)

print("Test Set Performance:")
print(f"  - Loss: {test_loss:.4f}")
print(f"  - Accuracy: {test_accuracy:.4f}")
print(f"  - Precision: {test_precision:.4f}")
print(f"  - Recall: {test_recall:.4f}")
print(f"  - AUC: {test_auc:.4f}")
print()

# Make predictions
y_pred_prob = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Calculate F1-score manually
precision = test_precision
recall = test_recall
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"  - F1-Score: {f1_score:.4f}")
print()

# Classification Report
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# ============================================================================
# 11. CONFUSION MATRIX
# ============================================================================

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()
print("Interpretation:")
print(f"  - True Negatives (Correctly predicted No Churn): {cm[0, 0]}")
print(f"  - False Positives (Incorrectly predicted Churn): {cm[0, 1]}")
print(f"  - False Negatives (Incorrectly predicted No Churn): {cm[1, 0]}")
print(f"  - True Positives (Correctly predicted Churn): {cm[1, 1]}")
print()

# ============================================================================
# 12. VISUALIZATION
# ============================================================================

print("\nSTEP 10: Generating Visualizations")
print("-" * 80)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training and Validation Loss
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Training and Validation Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            cbar_kws={'label': 'Count'})
axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted Label', fontsize=12)
axes[1, 0].set_ylabel('True Label', fontsize=12)

# Plot 4: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
axes[1, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('False Positive Rate', fontsize=12)
axes[1, 1].set_ylabel('True Positive Rate', fontsize=12)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('churn_model_evaluation.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'churn_model_evaluation.png'")
plt.show()

# ============================================================================
# 13. SAVE THE MODEL
# ============================================================================

print("\nSTEP 11: Saving the Model")
print("-" * 80)

# Save the final model
model.save('telco_churn_model_final.h5')
print("Model saved as 'telco_churn_model_final.h5'")
print("Best model (from early stopping) saved as 'best_churn_model.h5'")
print()

# ============================================================================
# 14. SUMMARY AND CONCLUSIONS
# ============================================================================

print("\n" + "=" * 80)
print("MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nKey Results:")
print(f"  ✓ Test Accuracy: {test_accuracy:.2%}")
print(f"  ✓ Test Precision: {test_precision:.2%}")
print(f"  ✓ Test Recall: {test_recall:.2%}")
print(f"  ✓ Test F1-Score: {f1_score:.2%}")
print(f"  ✓ Test AUC: {test_auc:.4f}")
print("\nDeliverables:")
print("  ✓ Trained model: telco_churn_model_final.h5")
print("  ✓ Best model checkpoint: best_churn_model.h5")
print("  ✓ Evaluation plots: churn_model_evaluation.png")
print("\nNotes:")
print("  - SMOTE was applied to balance the training data")
print("  - Early stopping prevented overfitting")
print("  - Dropout layers (0.3) were used for regularization")
print("  - Model achieved good performance on imbalanced test set")
print("=" * 80)