# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn for preprocessing and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight

# TensorFlow/Keras for the Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Set style for plots
plt.style.use('ggplot')

# ==========================================
# 2. DATA ACQUISITION & CLEANING
# ==========================================
# Load the dataset
# NOTE: Replace 'WA_Fn-UseC_-Telco-Customer-Churn.csv' with your actual file name
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Drop customerID as it is not needed for prediction
df.drop('customerID', axis=1, inplace=True)

# SPECIFIC CLEANING TASK mentioned in your group report:
# "strip the whitespace in total charges and turn it into a number"
# We force errors='coerce' to turn non-numeric strings (like empty spaces) into NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for null values created by the previous step and remove them
print(f"Missing values before dropping: {df.isnull().sum().sum()}")
df.dropna(inplace=True)

# ==========================================
# 3. DATA PREPROCESSING & ENCODING
# ==========================================
# The dataset has many categorical variables that need encoding

# 3a. Target Variable (Churn) - Binary Encoding
# Yes -> 1, No -> 0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3b. Feature Encoding
# We use "Get Dummies" (One-Hot Encoding) for categorical features
# drop_first=True avoids the "dummy variable trap" (multicollinearity)
df = pd.get_dummies(df, drop_first=True)

# Separation of Features (X) and Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# ==========================================
# 4. DATA SPLITTING [cite: 51]
# ==========================================
# Splitting into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. FEATURE SCALING
# ==========================================
# Neural networks converge faster when data is scaled
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# ==========================================
# 6. NEURAL NETWORK ARCHITECTURE [cite: 53]
# ==========================================
model = Sequential()

# Input Layer & 1st Hidden Layer
# units=16 is a hyperparameter you can discuss in the report
# activation='relu' is standard for hidden layers
model.add(Dense(units=16, activation='relu', input_dim=X_train.shape[1]))

# 2nd Hidden Layer
model.add(Dense(units=8, activation='relu'))

# Dropout Layer (Optional but good for preventing overfitting)
model.add(Dropout(0.2))

# Output Layer
# units=1 because it is binary classification (Churn Yes/No)
# activation='sigmoid' is required for binary output (0 to 1 probability)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# 7. MODEL TRAINING (UPDATED FOR IMBALANCE)
# ==========================================

# Calculate weights: This tells the model how much to "pay attention" to each class.
# Since Churn (1) is rare, it will get a higher weight.
class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Convert to a dictionary format that Keras expects: {0: weight_0, 1: weight_1}
class_weights_dict = dict(enumerate(class_weights_vals))
print(f"Class Weights Calculated: {class_weights_dict}")

# Train the model with class_weight parameter added
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    class_weight=class_weights_dict,  # <--- THIS IS THE KEY UPDATE
                    verbose=1)

# ==========================================
# 8. EVALUATION & RESULTS [cite: 56, 82]
# ==========================================
# Predict on the test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int) # Convert probability to 0 or 1

# Print Metrics
print("\n--- Model Evaluation ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Training History Visualization (Loss & Accuracy)
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()