
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Step 1: Page configuration
st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")
st.title("Breast Cancer Classification App")
st.write("Predict whether a tumor is malignant or benign using all numeric features.")

# Step 2: Load trained model, scaler, and label encoder
model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Step 3: Load feature names from dataset
data = pd.read_csv("data/breast_cancer_data.csv")
feature_names = [col for col in data.columns if col not in ["id", "diagnosis"]]

# Step 4: Sidebar inputs for all features
st.sidebar.header("Enter Tumor Features")
input_data = []
for feature in feature_names:
    value = st.sidebar.number_input(feature, value=float(data[feature].mean()))
    input_data.append(value)

input_array = np.array([input_data])
input_scaled = scaler.transform(input_array)

# Step 5: Predict button
if st.sidebar.button("Predict"):
    prediction = model.predict(input_scaled)[0]            # Predict class
    probability = model.predict_proba(input_scaled)[0]     # Get confidence
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # Step 6: Display results
    st.subheader("Prediction Result")
    if predicted_label == "M":
        st.write(f"Tumor is Malignant ({probability[1]*100:.1f}% confidence)")
    else:
        st.write(f"Tumor is Benign ({probability[0]*100:.1f}% confidence)")

# Explanation of Steps and Tools:
# Streamlit: Create an interactive web app.
# Sidebar: Users can input all numeric features.
# joblib: Load the saved model, scaler, and label encoder.
# Scale user inputs before prediction.
# Display predicted class and confidence.
