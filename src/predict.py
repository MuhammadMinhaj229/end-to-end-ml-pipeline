import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List

model_path = os.path.join(os.path.dirname(__file__), "../models/churn_model.pkl")
model = None

FEATURE_NAMES = [
    "Tenure (Months)",
    "Monthly Charges ($)",
    "Total Charges ($)",
    "Support Tickets",
    "Usage Frequency",
    "Session Duration (Mins)",
    "Login Consistency",
    "Payment Delay (Days)",
    "NPS Score",
    "Plan Tier"
]

def load_model():
    global model
    if model is None:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            raise Exception("Model not found. Train the model first.")

def determine_playbook(prob: float, features: dict) -> tuple:
    """Assigns risk tier, primary driver, and actionable playbook."""
    if prob >= 0.70:
        tier = "Critical Risk"
        if features.get("feature_3", 0) > 3 or features.get("feature_7", 0) > 10:
            driver = "High Support Friction & Payment Delay"
            playbook = "Dispatch Customer Success Executive; schedule priority technical audit."
        else:
            driver = "Sharp Engagement Deceleration"
            playbook = "Trigger executive retention discount and re-onboarding concierge."
    elif prob >= 0.40:
        tier = "Elevated Risk"
        driver = "Sub-optimal Platform Adoption"
        playbook = "Deliver proactive feature adoption webinar and account check-in."
    else:
        tier = "Healthy Account"
        driver = "Stable Consistent Activity"
        playbook = "Standard quarterly business review and expansion cross-sell."
    return tier, driver, playbook

def predict(features: dict) -> dict:
    load_model()
    # Ensure ordered columns
    cols = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame([features])[cols]
    prob = float(model.predict_proba(df)[0][1])
    pred = int(model.predict(df)[0])

    tier, driver, playbook = determine_playbook(prob, features)

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": pred,
        "risk_tier": tier,
        "top_risk_driver": driver,
        "recommended_playbook": playbook
    }

def predict_batch(features_list: list) -> list:
    load_model()
    cols = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(features_list)[cols]
    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    results = []
    for feats, prob, pred in zip(features_list, probs, preds):
        tier, driver, playbook = determine_playbook(float(prob), feats)
        results.append({
            "churn_probability": round(float(prob), 4),
            "churn_prediction": int(pred),
            "risk_tier": tier,
            "top_risk_driver": driver,
            "recommended_playbook": playbook
        })
    return results
