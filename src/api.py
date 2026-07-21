from fastapi import FastAPI, HTTPException
from src.schemas import ChurnRequest, ChurnResponse, BatchChurnRequest, BatchChurnResponse
from src.predict import predict, predict_batch

app = FastAPI(title="Churn Prediction API", description="API for Customer Churn Prediction")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    return {"version": "1.0", "metrics": {"AUC": 0.89}}

@app.post("/predict", response_model=ChurnResponse)
def predict_endpoint(request: ChurnRequest):
    try:
        # Pydantic v2 support
        if hasattr(request, 'model_dump'):
            result = predict(request.model_dump())
        else:
            result = predict(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchChurnResponse)
def batch_predict_endpoint(request: BatchChurnRequest):
    try:
        if hasattr(request.requests[0], 'model_dump'):
            features_list = [req.model_dump() for req in request.requests]
        else:
            features_list = [req.dict() for req in request.requests]
        results = predict_batch(features_list)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
