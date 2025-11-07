import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import joblib
import os

# ----------------------
# 1. Load data
# ----------------------
data = pd.read_csv("data/breast_cancer_data.csv")
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]  # drop unnamed cols

# ----------------------
# 2. Fill missing numeric values
# ----------------------
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
data = data.dropna()  # drop any remaining NaN

# ----------------------
# 3. Encode target
# ----------------------
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# ----------------------
# 4. Features and target
# ----------------------
X = data.drop(['id', 'diagnosis'], axis=1, errors='ignore')
y = data['diagnosis']

# Save feature names for the app
os.makedirs("models", exist_ok=True)
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

# ----------------------
# 5. Train/test split and scaling
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------
# 6. Train SVM model
# ----------------------
model = SVC(probability=True, kernel='rbf', random_state=42)
model.fit(X_train_scaled, y_train)

# ----------------------
# 7. Save model artifacts
# ----------------------
joblib.dump(model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Model, scaler, encoder, and feature names saved successfully!")
