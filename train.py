# Train the Model (train_model.py)
# Import required libraries
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
data = pd.read_csv("data/breast_cancer_data.csv")
data=data.fillna(data.mean())
data=data.dropna()
print(data.dtypes)
# Step 2: Encode target labels
# Convert 'M' -> 1 (Malignant), 'B' -> 0 (Benign)
label_encoder = LabelEncoder()
data["diagnosis"] = label_encoder.fit_transform(data["diagnosis"])

# Step 3: Select features (all numeric features) and target
X = data.drop(columns=["id", "diagnosis"])  # Features
y = data["diagnosis"]                        # Target

# Step 4: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Scale features
# Scaling ensures all features contribute equally to the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train SVM classifier
# Linear kernel, probability=True for confidence scores
svm_model = SVC(kernel="linear", C=1.0, random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)

# Step 7: Evaluate model performance
y_pred = svm_model.predict(X_test_scaled)
print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 8: Save model, scaler, and label encoder for deployment
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Model, scaler, and label encoder saved successfully.")



# Explanation of Tools and Steps:
# pandas: Load and manipulate CSV data.
# LabelEncoder: Convert target labels (M/B) to numeric codes.
# train_test_split: Split data into training (80%) and testing (20%).
# StandardScaler: Normalize numeric features for better SVM performance.
# SVC: Support Vector Machine classifier for tumor classification.
# joblib: Save model and preprocessing objects for later use.
# accuracy_score & classification_report: Evaluate model performance.