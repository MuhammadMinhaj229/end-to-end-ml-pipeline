import joblib
import pandas as pd
import os

model_path = os.path.join(os.path.dirname(__file__), "../models/churn_model.pkl")
model = None

def load_model():
    global model
    if model is None:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            raise Exception("Model not found. Train the model first.")

def predict(features: dict) -> dict:
    load_model()
    df = pd.DataFrame([features])
    prob = model.predict_proba(df)[0][1]
    pred = int(model.predict(df)[0])
    return {"churn_probability": float(prob), "churn_prediction": pred}

def predict_batch(features_list: list) -> list:
    load_model()
    df = pd.DataFrame(features_list)
    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    results = []
    for prob, pred in zip(probs, preds):
        results.append({"churn_probability": float(prob), "churn_prediction": int(pred)})
    return results
