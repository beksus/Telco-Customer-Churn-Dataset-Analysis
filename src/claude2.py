"""
Telco Customer Churn Prediction - Neural Network with Comprehensive Evaluation
==============================================================================
This script implements a deep learning model with comprehensive evaluation including:
- Multiple performance metrics
- Baseline model comparison (Logistic Regression)
- Feature importance analysis
- Detailed discussion of model strengths and weaknesses

Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv
Task: Binary Classification (Churn: Yes/No)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
    accuracy_score
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("=" * 90)
print("TELCO CUSTOMER CHURN PREDICTION - COMPREHENSIVE EVALUATION")
print("=" * 90)
print()

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("STEP 1: Data Loading and Exploration")
print("-" * 90)

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Total samples: {df.shape[0]}")
print(f"Total features: {df.shape[1] - 1} (excluding target)")
print()

# Check target distribution
print("Target variable distribution:")
churn_dist = df['Churn'].value_counts()
print(churn_dist)
print(f"\nChurn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")
print(f"Class imbalance ratio: {churn_dist['No'] / churn_dist['Yes']:.2f}:1")
print()

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("STEP 2: Data Preprocessing")
print("-" * 90)

# Convert TotalCharges to numeric (handle blank strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill NaNs in TotalCharges with median
missing_before = df['TotalCharges'].isna().sum()
if missing_before > 0:
    print(f"Found {missing_before} missing values in TotalCharges")
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    print(f"Filled with median: {df['TotalCharges'].median():.2f}")
print()

# Drop customerID (identifier column)
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)
    print("Dropped 'customerID' column")

# Standardize service-specific placeholders
df.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
print("Standardized service placeholder values")
print()

# ============================================================================
# STEP 3: FEATURE ENCODING
# ============================================================================
print("STEP 3: Feature Encoding")
print("-" * 90)

# Identify categorical columns
object_cols = [c for c in df.columns if df[c].dtype == 'object']
print(f"Found {len(object_cols)} categorical columns")

# Encode binary columns and apply one-hot encoding for multi-class
for c in object_cols:
    unique_vals = set(df[c].dropna().unique())
    if unique_vals <= {'Yes', 'No'}:
        df[c] = df[c].map({'Yes': 1, 'No': 0})
    elif unique_vals <= {'Male', 'Female'}:
        df[c] = df[c].map({'Male': 1, 'Female': 0})
    else:
        # One-hot encode remaining categorical variables
        df = pd.get_dummies(df, columns=[c], drop_first=True)

print(f"Final feature count after encoding: {df.shape[1] - 1}")
print()

# ============================================================================
# STEP 4: FEATURE SCALING AND TRAIN-TEST SPLIT
# ============================================================================
print("STEP 4: Feature Scaling and Data Splitting")
print("-" * 90)

# Separate features and target
y = df['Churn'].astype(int)
X = df.drop(columns=['Churn'])

# Store feature names for later analysis
feature_names = X.columns.tolist()

# Standardize numeric features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
print(f"Standardized numerical features: {num_cols}")

# Train/test split (80/20) stratified on churn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0] / len(X):.1%})")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0] / len(X):.1%})")
print(f"Training churn rate: {y_train.mean():.2%}")
print(f"Test churn rate: {y_test.mean():.2%}")
print()

# ============================================================================
# STEP 5: HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================================
print("STEP 5: Handling Class Imbalance with SMOTE")
print("-" * 90)

print(f"Before SMOTE: Class 0={np.bincount(y_train)[0]}, Class 1={np.bincount(y_train)[1]}")
print(f"Imbalance ratio: {np.bincount(y_train)[0] / np.bincount(y_train)[1]:.2f}:1")

# Apply SMOTE on training set only
sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print(f"After SMOTE: Class 0={np.bincount(y_train_res)[0]}, Class 1={np.bincount(y_train_res)[1]}")
print(f"Balanced ratio: {np.bincount(y_train_res)[0] / np.bincount(y_train_res)[1]:.2f}:1")
print()

# ============================================================================
# STEP 6: BUILD NEURAL NETWORK MODEL
# ============================================================================
print("STEP 6: Building Neural Network Model")
print("-" * 90)

input_dim = X_train_res.shape[1]


def build_model(input_dim):
    """
    Build a Sequential neural network for binary classification.

    Architecture:
    - Input layer (implicit)
    - Hidden layer 1: 128 neurons + ReLU + Dropout(0.3)
    - Hidden layer 2: 64 neurons + ReLU + Dropout(0.3)
    - Output layer: 1 neuron + Sigmoid
    """
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu', name='hidden_1'),
        Dropout(0.3, name='dropout_1'),
        Dense(64, activation='relu', name='hidden_2'),
        Dropout(0.3, name='dropout_2'),
        Dense(1, activation='sigmoid', name='output')
    ], name='ChurnNN')

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model


model = build_model(input_dim)
print("Neural Network Architecture:")
model.summary()
print()

# ============================================================================
# STEP 7: TRAIN NEURAL NETWORK
# ============================================================================
print("STEP 7: Training Neural Network")
print("-" * 90)

# Define callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_churn_nn_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# Train the model
print("Training in progress...")
history = model.fit(
    X_train_res, y_train_res,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print(f"\nTraining completed after {len(history.history['loss'])} epochs")
print()

# ============================================================================
# STEP 8: EVALUATE NEURAL NETWORK ON TEST SET
# ============================================================================
print("STEP 8: Neural Network Evaluation on Test Set")
print("-" * 90)

# Predictions
y_prob_nn = model.predict(X_test, verbose=0).ravel()
y_pred_nn = (y_prob_nn >= 0.5).astype(int)

# Metrics
nn_accuracy = accuracy_score(y_test, y_pred_nn)
nn_precision = precision_score(y_test, y_pred_nn)
nn_recall = recall_score(y_test, y_pred_nn)
nn_f1 = f1_score(y_test, y_pred_nn)
nn_auc = roc_auc_score(y_test, y_prob_nn)

print("Neural Network - Test Set Performance:")
print(f"  Accuracy:  {nn_accuracy:.4f}")
print(f"  Precision: {nn_precision:.4f}")
print(f"  Recall:    {nn_recall:.4f}")
print(f"  F1-Score:  {nn_f1:.4f}")
print(f"  ROC-AUC:   {nn_auc:.4f}")
print()

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred_nn, target_names=['No Churn', 'Churn'], digits=4))

# Confusion matrix
cm_nn = confusion_matrix(y_test, y_pred_nn)
print("Confusion Matrix:")
print(cm_nn)
print(f"  True Negatives:  {cm_nn[0, 0]}")
print(f"  False Positives: {cm_nn[0, 1]}")
print(f"  False Negatives: {cm_nn[1, 0]}")
print(f"  True Positives:  {cm_nn[1, 1]}")
print()

# ============================================================================
# STEP 9: BASELINE MODEL - LOGISTIC REGRESSION
# ============================================================================
print("STEP 9: Baseline Model - Logistic Regression")
print("-" * 90)

# Train logistic regression with class weighting (no SMOTE needed)
log_reg = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    random_state=RANDOM_STATE
)
log_reg.fit(X_train, y_train)

# Predictions
y_pred_log = log_reg.predict(X_test)
y_prob_log = log_reg.predict_proba(X_test)[:, 1]

# Metrics
log_accuracy = accuracy_score(y_test, y_pred_log)
log_precision = precision_score(y_test, y_pred_log)
log_recall = recall_score(y_test, y_pred_log)
log_f1 = f1_score(y_test, y_pred_log)
log_auc = roc_auc_score(y_test, y_prob_log)

print("Logistic Regression - Test Set Performance:")
print(f"  Accuracy:  {log_accuracy:.4f}")
print(f"  Precision: {log_precision:.4f}")
print(f"  Recall:    {log_recall:.4f}")
print(f"  F1-Score:  {log_f1:.4f}")
print(f"  ROC-AUC:   {log_auc:.4f}")
print()

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred_log, target_names=['No Churn', 'Churn'], digits=4))

# Confusion matrix
cm_log = confusion_matrix(y_test, y_pred_log)
print("Confusion Matrix:")
print(cm_log)
print()

# ============================================================================
# STEP 10: MODEL COMPARISON AND ANALYSIS
# ============================================================================
print("STEP 10: Comparative Analysis - Neural Network vs Logistic Regression")
print("-" * 90)

# Create comparison table
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Neural Network': [nn_accuracy, nn_precision, nn_recall, nn_f1, nn_auc],
    'Logistic Regression': [log_accuracy, log_precision, log_recall, log_f1, log_auc],
    'Difference (NN - LR)': [
        nn_accuracy - log_accuracy,
        nn_precision - log_precision,
        nn_recall - log_recall,
        nn_f1 - log_f1,
        nn_auc - log_auc
    ]
})

print("\nPerformance Comparison Table:")
print(comparison_df.to_string(index=False))
print()

# Determine better model
if nn_f1 > log_f1:
    better_model = "Neural Network"
    improvement = ((nn_f1 - log_f1) / log_f1) * 100
else:
    better_model = "Logistic Regression"
    improvement = ((log_f1 - nn_f1) / nn_f1) * 100

print(f"Best performing model: {better_model}")
print(f"F1-Score improvement: {improvement:.2f}%")
print()

# ============================================================================
# STEP 11: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("STEP 11: Feature Importance Analysis")
print("-" * 90)

print("Computing permutation importance (Logistic Regression baseline)...")
perm = permutation_importance(
    log_reg, X_test, y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

feat_importances = pd.Series(perm.importances_mean, index=feature_names)
top_feats = feat_importances.sort_values(ascending=False).head(15)

print("\nTop 15 Most Important Features:")
for idx, (feat, importance) in enumerate(top_feats.items(), 1):
    print(f"  {idx:2d}. {feat:30s} : {importance:.6f}")
print()

# ============================================================================
# STEP 12: COMPREHENSIVE VISUALIZATIONS
# ============================================================================
print("STEP 12: Generating Comprehensive Visualizations")
print("-" * 90)

# Create comprehensive figure with 6 subplots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Training History - Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_title('Neural Network: Training Loss', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training History - Accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax2.set_title('Neural Network: Training Accuracy', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Model Comparison Bar Chart
ax3 = fig.add_subplot(gs[0, 2])
metrics_comparison = comparison_df[['Metric', 'Neural Network', 'Logistic Regression']].set_index('Metric')
metrics_comparison.plot(kind='bar', ax=ax3, width=0.8)
ax3.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score')
ax3.set_ylim([0, 1])
ax3.legend(['Neural Network', 'Logistic Regression'])
ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 4: Confusion Matrix - Neural Network
ax4 = fig.add_subplot(gs[1, 0])
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            cbar_kws={'label': 'Count'})
ax4.set_title('Confusion Matrix: Neural Network', fontsize=12, fontweight='bold')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')

# Plot 5: Confusion Matrix - Logistic Regression
ax5 = fig.add_subplot(gs[1, 1])
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Greens', ax=ax5,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            cbar_kws={'label': 'Count'})
ax5.set_title('Confusion Matrix: Logistic Regression', fontsize=12, fontweight='bold')
ax5.set_xlabel('Predicted')
ax5.set_ylabel('Actual')

# Plot 6: ROC Curves Comparison
ax6 = fig.add_subplot(gs[1, 2])
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_prob_nn)
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
ax6.plot(fpr_nn, tpr_nn, linewidth=2, label=f'Neural Network (AUC={nn_auc:.3f})')
ax6.plot(fpr_log, tpr_log, linewidth=2, label=f'Logistic Regression (AUC={log_auc:.3f})')
ax6.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax6.set_title('ROC Curve Comparison', fontsize=12, fontweight='bold')
ax6.set_xlabel('False Positive Rate')
ax6.set_ylabel('True Positive Rate')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Feature Importance (spans bottom row)
ax7 = fig.add_subplot(gs[2, :])
top_feats_plot = top_feats.sort_values()
sns.barplot(x=top_feats_plot.values, y=top_feats_plot.index, ax=ax7, palette='viridis')
ax7.set_title('Top 15 Feature Importances (Permutation - Logistic Regression)',
              fontsize=12, fontweight='bold')
ax7.set_xlabel('Importance Score')
ax7.set_ylabel('Feature')

plt.savefig('comprehensive_churn_evaluation.png', dpi=300, bbox_inches='tight')
print("Comprehensive visualization saved as 'comprehensive_churn_evaluation.png'")
plt.show()

# ============================================================================
# STEP 13: SAVE MODELS
# ============================================================================
print("\nSTEP 13: Saving Models")
print("-" * 90)

# Save neural network
model.save('telco_churn_neural_network.h5')
print("Neural Network saved as 'telco_churn_neural_network.h5'")
print("Best model checkpoint saved as 'best_churn_nn_model.h5'")

# Save logistic regression (optional)
import joblib

joblib.dump(log_reg, 'telco_churn_logistic_regression.pkl')
print("Logistic Regression saved as 'telco_churn_logistic_regression.pkl'")
print()

# ============================================================================
# STEP 14: COMPREHENSIVE DISCUSSION AND CONCLUSIONS
# ============================================================================
print("\n" + "=" * 90)
print("COMPREHENSIVE EVALUATION SUMMARY")
print("=" * 90)

print("\n1. MODEL PERFORMANCE SUMMARY:")
print("-" * 90)
print(f"Neural Network:")
print(f"  • Accuracy:  {nn_accuracy:.2%} | Precision: {nn_precision:.2%} | Recall: {nn_recall:.2%}")
print(f"  • F1-Score:  {nn_f1:.4f} | ROC-AUC: {nn_auc:.4f}")
print()
print(f"Logistic Regression (Baseline):")
print(f"  • Accuracy:  {log_accuracy:.2%} | Precision: {log_precision:.2%} | Recall: {log_recall:.2%}")
print(f"  • F1-Score:  {log_f1:.4f} | ROC-AUC: {log_auc:.4f}")
print()

print("\n2. STRENGTHS OF THE NEURAL NETWORK MODEL:")
print("-" * 90)
strengths = [
    "Non-linear pattern recognition: Can capture complex, non-linear relationships between features",
    f"High recall ({nn_recall:.2%}): Better at identifying customers who will churn (fewer false negatives)",
    f"Strong ROC-AUC ({nn_auc:.4f}): Excellent discrimination between churn and non-churn customers",
    "SMOTE integration: Effectively handles class imbalance in training data",
    "Dropout regularization: Prevents overfitting with 0.3 dropout rate",
    "Early stopping: Automatically prevents overtraining by monitoring validation loss",
    "Scalability: Can be extended with more layers/neurons for larger datasets"
]
for i, strength in enumerate(strengths, 1):
    print(f"  {i}. {strength}")
print()

print("\n3. WEAKNESSES OF THE NEURAL NETWORK MODEL:")
print("-" * 90)
weaknesses = [
    "Black box nature: Difficult to interpret how individual features contribute to predictions",
    "Computational cost: Requires more training time and computational resources than logistic regression",
    f"Precision trade-off: Lower precision ({nn_precision:.2%}) means more false positives (predicting churn when customer stays)",
    "Hyperparameter sensitivity: Performance depends on architecture choices (layers, neurons, dropout rate)",
    "Data requirements: Needs larger datasets to fully leverage deep learning capabilities",
    "Risk of overfitting: Despite regularization, still prone to overfitting on small datasets"
]
for i, weakness in enumerate(weaknesses, 1):
    print(f"  {i}. {weakness}")
print()

print("\n4. COMPARISON WITH BASELINE (LOGISTIC REGRESSION):")
print("-" * 90)
if nn_f1 > log_f1:
    print(f"✓ Neural Network OUTPERFORMS Logistic Regression by {improvement:.2f}% in F1-Score")
    print(f"  • Key advantages: Better at capturing non-linear relationships in customer behavior")
    print(f"  • Higher recall: More effective at identifying actual churners")
else:
    print(f"✗ Logistic Regression OUTPERFORMS Neural Network by {improvement:.2f}% in F1-Score")
    print(f"  • Possible reasons: Dataset may have predominantly linear relationships")
    print(f"  • Baseline is simpler, more interpretable, and sufficient for this problem")

print()
print("Trade-offs observed:")
print(
    f"  • NN Recall: {nn_recall:.4f} vs LR Recall: {log_recall:.4f} → NN catches {(nn_recall - log_recall) * 100:.1f}% more churners")
print(
    f"  • NN Precision: {nn_precision:.4f} vs LR Precision: {log_precision:.4f} → NN has {abs(nn_precision - log_precision) * 100:.1f}% different false positive rate")
print()

print("\n5. KEY INSIGHTS FROM FEATURE IMPORTANCE ANALYSIS:")
print("-" * 90)
print(f"Top 3 Most Important Features:")
for i, (feat, imp) in enumerate(top_feats.head(3).items(), 1):
    print(f"  {i}. {feat}: {imp:.6f}")
print()
print("Interpretation:")
print("  • These features have the strongest predictive power for customer churn")
print("  • Business teams should focus retention efforts on customers with these characteristics")
print("  • Contract type and tenure are often key indicators of customer loyalty")
print()

print("\n6. BUSINESS RECOMMENDATIONS:")
print("-" * 90)
recommendations = [
    f"Deploy the Neural Network model for churn prediction (F1={nn_f1:.4f}, Recall={nn_recall:.2%})",
    "Focus retention campaigns on high-risk customers identified by the model",
    f"Accept {cm_nn[0, 1]} false positives as acceptable cost to capture {cm_nn[1, 1]} true churners",
    "Monitor model performance monthly and retrain with new data quarterly",
    "Investigate top features from importance analysis to understand churn drivers",
    "Consider ensemble methods (combining NN + LR) for potentially better performance"
]
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")
print()

print("\n7. LIMITATIONS AND FUTURE WORK:")
print("-" * 90)
limitations = [
    "Limited to historical data - cannot predict impact of new business initiatives",
    "Class imbalance may still affect real-world predictions despite SMOTE",
    "Model doesn't incorporate time-series patterns or customer interaction sequences",
    "External factors (market conditions, competitor actions) not included"
]
print("Current Limitations:")
for i, lim in enumerate(limitations, 1):
    print(f"  {i}. {lim}")
print()

future_work = [
    "Experiment with deeper architectures (3-4 hidden layers) and different activation functions",
    "Try ensemble methods (Random Forest, XGBoost) and model stacking",
    "Incorporate temporal features (customer lifetime value trends, usage patterns over time)",
    "Develop customer segmentation models to create targeted retention strategies",
    "Implement model explainability techniques (SHAP, LIME) for better interpretability"
]
print("Future Improvements:")
for i, fw in enumerate(future_work, 1):
    print(f"  {i}. {fw}")
print()

print("=" * 90)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 90)
print("\nDeliverables Generated:")
print("  ✓ Neural Network model: telco_churn_neural_network.h5")
print("  ✓ Best checkpoint: best_churn_nn_model.h5")
print("  ✓ Baseline model: telco_churn_logistic_regression.pkl")
print("  ✓ Comprehensive visualizations: comprehensive_churn_evaluation.png")
print("  ✓ Detailed performance analysis and comparisons")
print("  ✓ Feature importance analysis")
print("  ✓ Strengths, weaknesses, and recommendations")
print("=" * 90)