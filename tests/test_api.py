import pytest
from fastapi.testclient import TestClient
from src.api import app
import os

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict():
    payload = {
        f"feature_{i}": 0.5 for i in range(10)
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 500 and "Model not found" in response.text:
        pytest.skip("Model not trained yet")
    assert response.status_code == 200
    assert "churn_probability" in response.json()
