import streamlit as st
import requests
import os

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction Dashboard")

api_url = os.environ.get("API_URL", "http://localhost:8000")

st.header("Single Customer Prediction")

col1, col2, col3 = st.columns(3)
features = {}
for i in range(10):
    if i % 3 == 0:
        with col1:
            features[f"feature_{i}"] = st.number_input(f"Feature {i}", value=0.0)
    elif i % 3 == 1:
        with col2:
            features[f"feature_{i}"] = st.number_input(f"Feature {i}", value=0.0)
    else:
        with col3:
            features[f"feature_{i}"] = st.number_input(f"Feature {i}", value=0.0)

if st.button("Predict"):
    try:
        response = requests.post(f"{api_url}/predict", json=features)
        if response.status_code == 200:
            result = response.json()
            prob = result['churn_probability']
            st.success(f"Churn Probability: {prob:.2%}")
            if prob > 0.5:
                st.error("High risk of churn!")
            else:
                st.info("Low risk of churn.")
        else:
            st.error(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {api_url}. Is it running?")
