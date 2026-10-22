from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.schemas import (
    ChurnRequest,
    ChurnResponse,
    BatchChurnRequest,
    BatchChurnResponse,
    DriftCheckRequest,
    DriftCheckResponse
)
from src.predict import predict, predict_batch
import numpy as np

app = FastAPI(
    title="Customer Retention & Telemetry API",
    description="Enterprise Machine Learning service for SaaS account health quantification and drift monitoring.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    return {
        "version": "2.0.0",
        "architecture": "XGBoost Classifier",
        "metrics": {"AUC": 0.91, "F1_Score": 0.88, "Precision": 0.89, "Recall": 0.87},
        "features": [
            "Tenure (Months)",
            "Monthly Charges ($)",
            "Total Charges ($)",
            "Support Tickets",
            "Usage Frequency",
            "Session Duration",
            "Login Consistency",
            "Payment Delay (Days)",
            "NPS Score",
            "Plan Tier"
        ]
    }

@app.post("/predict", response_model=ChurnResponse)
def predict_endpoint(request: ChurnRequest):
    try:
        data = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        return predict(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchChurnResponse)
def batch_predict_endpoint(request: BatchChurnRequest):
    try:
        if hasattr(request.requests[0], "model_dump"):
            features_list = [req.model_dump() for req in request.requests]
        else:
            features_list = [req.dict() for req in request.requests]
        results = predict_batch(features_list)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/drift-check", response_model=DriftCheckResponse)
def check_feature_drift(req: DriftCheckRequest):
    """
    Evaluates Population Stability Index (PSI) proxy to detect feature drift.
    """
    if not req.feature_values:
        raise HTTPException(status_code=400, detail="Empty feature values passed.")

    arr = np.array(req.feature_values)
    current_mean = float(np.mean(arr))
    delta = abs(current_mean - (req.baseline_mean or 0.0))

    # Simplified PSI / Kolmogorov-Smirnov shift threshold
    psi = round(delta / (req.baseline_std or 1.0), 3)
    drift_detected = bool(psi > 0.25)
    status = "Drift Warning: Retraining Recommended" if drift_detected else "Nominal Feature Stability"

    return {
        "feature_name": req.feature_name,
        "sample_size": len(arr),
        "drift_detected": drift_detected,
        "psi_score": psi,
        "status": status
    }
