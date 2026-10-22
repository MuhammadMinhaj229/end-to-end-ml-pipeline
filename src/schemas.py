from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChurnRequest(BaseModel):
    feature_0: float = Field(..., description="Tenure (Months)")
    feature_1: float = Field(..., description="Monthly Charges (USD)")
    feature_2: float = Field(..., description="Total Charges (USD)")
    feature_3: float = Field(..., description="Support Tickets Opened")
    feature_4: float = Field(..., description="Usage Frequency Index")
    feature_5: float = Field(..., description="Session Duration (Minutes)")
    feature_6: float = Field(..., description="Login Consistency Ratio")
    feature_7: float = Field(..., description="Payment Delay (Days)")
    feature_8: float = Field(..., description="Net Promoter Score (NPS)")
    feature_9: float = Field(..., description="Plan Tier (1-3)")

class BatchChurnRequest(BaseModel):
    requests: List[ChurnRequest]

class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_tier: Optional[str] = "Moderate"
    recommended_playbook: Optional[str] = "Standard Success Review"
    top_risk_driver: Optional[str] = "Usage Frequency"

class BatchChurnResponse(BaseModel):
    predictions: List[ChurnResponse]

class DriftCheckRequest(BaseModel):
    feature_values: List[float]
    feature_name: str = "feature_0"
    baseline_mean: Optional[float] = 0.0
    baseline_std: Optional[float] = 1.0

class DriftCheckResponse(BaseModel):
    feature_name: str
    sample_size: int
    drift_detected: bool
    psi_score: float
    status: str
