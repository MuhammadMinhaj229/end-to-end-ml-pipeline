from pydantic import BaseModel
from typing import List

class ChurnRequest(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float

class BatchChurnRequest(BaseModel):
    requests: List[ChurnRequest]

class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int

class BatchChurnResponse(BaseModel):
    predictions: List[ChurnResponse]
