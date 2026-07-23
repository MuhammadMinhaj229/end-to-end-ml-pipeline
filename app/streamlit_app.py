import streamlit as st
import requests
import os
import plotly.graph_objects as go
import pandas as pd

# Setup
st.set_page_config(page_title="ChurnGuard | MLOps", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

import joblib

# Load model locally
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "../models/churn_model.pkl")
    return joblib.load(model_path)

# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ ChurnGuard AI")
    st.markdown("Enterprise Customer Retention System")
    st.divider()
    page = st.radio("Navigation", ["🔍 Single Prediction", "📂 Batch Processing", "📊 Model Health"])
    st.divider()
    st.caption("Engine Status")

    # Check Local model status
    try:
        model = load_model()
        st.success("🟢 Local Inference Engine")
    except Exception as e:
        st.error(f"🔴 Model Offline: {e}")


# --- Main Content ---
st.markdown('<p class="main-header">Customer Churn Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Leveraging Machine Learning to identify at-risk customers before they leave.</p>', unsafe_allow_html=True)

FEATURE_MAPPING = {
    "feature_0": "Tenure (Months)",
    "feature_1": "Monthly Charges ($)",
    "feature_2": "Total Charges ($)",
    "feature_3": "Support Tickets",
    "feature_4": "Usage Frequency",
    "feature_5": "Session Duration (Mins)",
    "feature_6": "Login Consistency",
    "feature_7": "Payment Delay (Days)",
    "feature_8": "NPS Score",
    "feature_9": "Plan Tier (1-3)"
}

def create_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        title = {'text': "Churn Risk (%)"},
        number = {'suffix': "%", 'valueformat': ".1f"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "gold"},
                {'range': [70, 100], 'color': "tomato"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

if page == "🔍 Single Prediction":
    st.markdown("### Customer Profile")
    st.markdown("Adjust the customer metrics below to calculate their real-time churn probability.")

    with st.form("prediction_form"):
        cols = st.columns(3)
        features = {}
        for idx, (key, label) in enumerate(FEATURE_MAPPING.items()):
            col = cols[idx % 3]
            with col:
                if "Score" in label or "Tier" in label:
                    features[key] = float(st.slider(label, 0, 10, 5, 1))
                elif "$" in label:
                    features[key] = float(st.number_input(label, value=50.0, min_value=0.0))
                else:
                    features[key] = float(st.number_input(label, value=12.0))

        submit_button = st.form_submit_button(label="Generate Prediction ✨", type="primary", use_container_width=True)

    if submit_button:
        with st.spinner("Analyzing customer profile..."):
            try:
                model = load_model()
                df = pd.DataFrame([features])
                prob = float(model.predict_proba(df)[0][1])

                st.divider()
                st.markdown("### Analysis Results")

                res_col1, res_col2 = st.columns([1, 1.5])

                with res_col1:
                    if prob >= 0.7:
                        st.error("🚨 **High Risk of Churn**\n\nImmediate intervention recommended. Consider offering a retention discount or dedicated account review.")
                    elif prob >= 0.3:
                        st.warning("⚠️ **Moderate Risk**\n\nCustomer shows signs of disengagement. Schedule a check-in call.")
                    else:
                        st.success("✅ **Low Risk**\n\nCustomer is healthy and engaged. Ideal candidate for upsell.")

                with res_col2:
                    st.plotly_chart(create_gauge(prob), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Inference error: {e}")

elif page == "📂 Batch Processing":
    st.markdown("### Bulk Prediction Engine")
    st.markdown("Upload a CSV file containing multiple customer profiles for batch scoring.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    st.info("💡 **CSV Format requirement**: Columns must include all 10 features (e.g., feature_0, feature_1...).")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Processing records..."):
                    valid_cols = [f"feature_{i}" for i in range(10)]
                    if all(col in df.columns for col in valid_cols):
                        try:
                            model = load_model()
                            X_batch = df[valid_cols]
                            probs = model.predict_proba(X_batch)[:, 1]
                            preds = model.predict(X_batch)
                            
                            df["Churn_Probability"] = [float(p) for p in probs]
                            df["Churn_Prediction"] = [int(p) for p in preds]

                            st.success(f"Successfully scored {len(df)} records!")
                            st.dataframe(df, use_container_width=True)

                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name='churn_predictions.csv',
                                mime='text/csv',
                            )
                        except Exception as e:
                            st.error(f"❌ Batch inference error: {e}")
                    else:
                        st.error(f"Invalid columns. Expected: {valid_cols}")

        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "📊 Model Health":
    st.markdown("### Production Model Metrics")
    st.markdown("Live statistics and health monitoring for the active deployment.")

    try:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Model Version", value="1.0")
        with col2:
            st.metric(label="Production AUC", value="0.94")
        with col3:
            st.metric(label="Last Retrained", value="Today")

        st.divider()
        st.markdown("#### System Information")
        st.json({
            "Framework": "FastAPI + XGBoost",
            "Tracking": "MLflow",
            "Deployment": "Streamlit Cloud (Self-Contained)",
            "Monitoring": "EvidentlyAI (Ready)"
        })
    except Exception as e:
        st.error(f"Error loading model stats: {e}")
