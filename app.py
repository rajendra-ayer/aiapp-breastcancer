import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------
# 1. Page setup
# ----------------------
st.title("Breast Cancer Classifier")
st.write("Predict whether a tumor is malignant or benign.")

# ----------------------
# 2. Load model, scaler, encoder, and feature names
# ----------------------
model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")
feature_names = joblib.load("models/feature_names.pkl")  # exact features used in training

# ----------------------
# 3. Load dataset (optional, for min/max/mean)
# ----------------------
data = pd.read_csv("data/breast_cancer_data.csv")

# ----------------------
# 4. Input fields for all features
# ----------------------
st.write("Enter tumor features:")

inputs = []
for f in feature_names:
    if f in data.columns:
        min_val = float(data[f].min())
        max_val = float(data[f].max())
        mean_val = float(data[f].mean())
    else:
        min_val, max_val, mean_val = 0.0, 10.0, 5.0

    value = st.number_input(f, min_value=min_val, max_value=max_val, value=mean_val)
    inputs.append(value)

# ----------------------
# 5. Predict button
# ----------------------
if st.button("Predict"):
    X = np.array([inputs])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    label = encoder.inverse_transform([pred])[0]

    # ----------------------
    # 6. Show result
    # ----------------------
    st.subheader("Prediction Result")
    if label == "M":
        st.write(f"Tumor is Malignant ({prob[1]*100:.1f}% confidence)")
    else:
        st.write(f"Tumor is Benign ({prob[0]*100:.1f}% confidence)")
