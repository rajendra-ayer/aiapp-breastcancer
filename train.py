import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import joblib
import os

# ------------------------------
# Step 1: Load data
# ------------------------------
data = pd.read_csv("data/breast_cancer_data.csv")

# Step 2: Drop useless columns (like unnamed indexes if exist)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Step 3: Fill only numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Step 4: Drop rows with any remaining NaN (safety)
data = data.dropna()

# Step 5: Encode target
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Step 6: Define features and target
X = data.drop(['id', 'diagnosis'], axis=1, errors='ignore')
y = data['diagnosis']

# Step 7: Double-check for NaN before training
if X.isnull().sum().sum() > 0:
    print("⚠️ Warning: Still found NaN values, replacing with column means.")
    X = X.fillna(X.mean())

# Step 8: Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Verify scaling result
if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
    raise ValueError(" NaN values found after scaling. Check dataset preprocessing!")

# Step 10: Train SVM model
svm_model = SVC(probability=True, kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Step 11: Save model artifacts
os.makedirs("models", exist_ok=True)
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Model, scaler, and encoder saved successfully!")
